from __future__ import annotations

import textwrap
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Iterator, Callable, Sequence

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from PIL import Image
from detectron2.data.catalog import Metadata
from ignite.engine import Engine
from ignite.metrics import Metric
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.utils import scatter_
from torchvision.ops import box_iou

from ..structures import ImageSize, Vocabulary, matched_boxlist_union
from ..utils.utils import NamedEnumMixin


def mean_average_precision(annotations, scores) -> float:
    """Computes the mean average precision (mAP) for a multi-class multi-label scenario.

    In object detection mAP is the mean AP across all classes, i.e.
    for each class c, compute AP[c] using all samples in the dataset, then take the average across classes.

    Not to bo confounded with:
    for each sample s, compute the AP[s] considering all classes, then take the average across samples.
    """

    # The correct behavior (for each class compute the AP using all samples, then average across classes)
    # corresponds to the `macro` aggregation from scikit-learn.
    # However, if we are given a small batch it is possible to have a column of all 0s in the annotations matrix,
    # i.e. none of the samples is positive for that class. It's best to pass `average=None` so that per-class
    # APs are returned and then compute the mean manually skipping nan values.

    with np.errstate(invalid="ignore"):
        average_precisions = sklearn.metrics.average_precision_score(
            y_true=annotations, y_score=scores, average=None
        )

    if np.isnan(average_precisions).all():
        return 0

    return np.nanmean(average_precisions).item()


class MeanAveragePrecisionBatch(object):
    def __call__(self, engine: Engine):
        y_true = engine.state.output["target"]
        y_score = engine.state.output["output"]
        avg_precision = mean_average_precision(y_true, y_score)

        # Return tensor so that ignite.metrics.Average takes batch size into account
        B = len(y_true)
        avg_precision = torch.full((B, 1), fill_value=avg_precision)
        engine.state.output[f"pc/mAP"] = avg_precision


class MeanAveragePrecisionEpoch(Metric):
    y_true: List[torch.Tensor]
    y_score: List[torch.Tensor]

    def reset(self):
        self.y_true = []
        self.y_score = []

    def update(self, output):
        y_true, y_score = output
        self.y_true.append(y_true)
        self.y_score.append(y_score)

    def compute(self):
        return mean_average_precision(torch.cat(self.y_true), torch.cat(self.y_score))


def precision_at(annotations, scores, sizes):
    """Precision@x

    - rank the relationships by their score and keep the top x
    - compute how many of those retrieved relationships are actually relevant

    Args:
        annotations: tensor of shape [num_samples, num_relationships] and values {0, 1}
        scores: tensor of shape [num_samples, num_relationships] and float scores

    ::

                  # ( relevant items retrieved )
      Precision = ------------------------------ = P ( relevant | retrieved )
                      # ( retrieved items )
    """
    result = {}
    # Sorted labels are the indexes that would sort y_score, e.g.
    # [[ 10, 3, 4, ....., 5, 41 ],
    #  [  1, 2, 6, ....., 8, 78 ]]
    # means that for the first image class 10 is the top scoring class
    sorted_labels = torch.argsort(scores, dim=1, descending=True)

    # One could do this to get the sorted scores
    # sorted_scores = torch.gather(y_scores, index=sorted_labels, dim=1)

    # Use these indexes to index into y_true, but keep only max(sizes) columns
    annotations_of_top_max_s = torch.gather(
        annotations, index=sorted_labels[:, : max(sizes)], dim=1
    )

    # cumsum[i, j] = number of relevant items within the top j+1 retrieved items
    # Cast to float to avoid int/int division.
    cumsum = annotations_of_top_max_s.cumsum(dim=1).float()

    # Given a size s, `cumsum[i, s-1] / s` gives the precision for sample i.
    # If we end up doing 0 / 0, it simply means that in the top-s documents
    # there was no relevant document, so precision should be 0.
    # Then we take the batch mean.
    for s in sizes:
        precision_per_sample = cumsum[:, (s - 1)] / s
        precision_per_sample[torch.isnan(precision_per_sample)] = 0.
        result[s] = precision_per_sample.mean(dim=0).item()

    return result


def recall_at(annotations, scores, sizes):
    """Recall@x

    - rank the relationships by their score and keep the top x
    - compute how many of the relevant relationships are among the retrieved

    ::

                # ( relevant items retrieved )
      Recall  = ------------------------------ = P ( retrieved | relevant )
                     # ( relevant items )

    Args:
        annotations: tensor of shape [num_samples, num_relationships] and values {0, 1}
        scores: tensor of shape [num_samples, num_relationships] and float scores

    References:

        `"Visual Relationship Detection with Language Priors" (Lu et al.) <https://arxiv.org/abs/1608.00187>`_
        describes `recall@x` as:

            The evaluation metrics we report is recall @ 100 and recall @ 50.
            Recall @ x computes the fraction of times the correct relationship is predicted
            in the top x confident relationship predictions. Since we have 70 predicates and
            an average of 18 objects per image, the total possible number of relationship
            predictions is 100×70×100, which implies that the random guess will result in a
            recall @ 100 of 0.00014.

        `Weakly-supervised learning of visual relations (Peyre et al.) <https://arxiv.org/abs/1707.09472>`_
        cites the previous paper and describes `recall@x` as:

            We compute recall @ x which corresponds to the proportion of ground truth
            pairs among the x top scored candidate pairs in each image.
    """

    result = {}
    # Sorted labels are the indexes that would sort y_score, e.g.
    # [[ 10, 3, 4, ....., 5, 41 ],
    #  [  1, 2, 6, ....., 8, 78 ]]
    # means that for the first image class 10 is the top scoring class
    sorted_labels = torch.argsort(scores, dim=1, descending=True)

    # One could do this to get the sorted scores
    # sorted_scores = torch.gather(y_scores, index=sorted_labels, dim=1)

    # Use these indexes to index into y_true, but keep only max(self.sizes) columns
    annotations_of_top_max_s = torch.gather(
        annotations, index=sorted_labels[:, : max(sizes)], dim=1
    )

    # cumsum[i, j] = number of relevant items within the top j+1 retrieved items
    # Cast to float to avoid int/int division later.
    cumsum = annotations_of_top_max_s.cumsum(dim=1).float()

    # Divide each row (a sample) by the total number of relevant document for
    # that row, to get the recall per sample.
    # If for a specific sample there are 0 relevant documents we get NaN in the division.
    # Last, take the batch mean skipping NaN values.
    # If we get a batch where all documents have 0 relevant documents, it's a problem.
    num_relevant_per_sample = annotations.sum(dim=1, keepdims=True)
    recall_per_sample = cumsum / num_relevant_per_sample
    for s in sizes:
        recall_at_s_per_sample = recall_per_sample[:, s - 1]
        finite = torch.isfinite(recall_at_s_per_sample)
        result[s] = recall_at_s_per_sample[finite].mean(dim=0).item()

    return result


class RecallAtBatch(object):
    """Recall@x over the output of the last batch"""

    def __init__(self, sizes: Tuple[int, ...] = (10, 30, 50)):
        self._sorted_sizes = list(sorted(sizes))

    def __call__(self, engine: Engine):
        y_true = engine.state.output["target"]
        y_score = engine.state.output["output"]
        recalls = recall_at(y_true, y_score, self._sorted_sizes)

        # Return tensors so that ignite.metrics.Average takes batch size into account
        B = len(y_true)
        engine.state.output["recalls"] = {}
        for k, r in recalls.items():
            r = torch.full((B, 1), fill_value=r)
            engine.state.output["recalls"][f"pc/recall_at_{k}"] = r


class RecallAtEpoch(Metric):
    """Recall@x by accumulating outputs over epochs"""

    _y_true: List[torch.Tensor]
    _y_score: List[torch.Tensor]

    def __init__(
            self,
            sizes: Tuple[int, ...] = (10, 30, 50),
            output_transform=lambda x: x,
            device=None,
    ):
        self._sorted_sizes = list(sorted(sizes))
        super(RecallAtEpoch, self).__init__(output_transform, device)

    def reset(self):
        self._y_true = []
        self._y_score = []

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_true, y_score = output
        self._y_true.append(y_true)
        self._y_score.append(y_score)

    def compute(self):
        y_true = torch.cat(self._y_true, dim=0)
        y_score = torch.cat(self._y_score, dim=0)
        return recall_at(y_true, y_score, self._sorted_sizes)

    def completed(self, engine, name):
        result = self.compute()
        for k, v in result.items():
            engine.state.metrics[f"{name}_{k}"] = v


class PredicatePredictionLogger(object):
    def __init__(
            self,
            grid: Tuple[int, int],
            data_root: Union[str, Path],
            tag: str,
            logger: SummaryWriter,
            global_step_fn: Callable[[], int],
            metadata: Metadata,
            save_dir: Optional[Union[str, Path]] = None,
    ):
        """

        Args:
            grid:
            img_dir: directory where the images will be opened from
            tag:
            logger: tensorboard logger for the images
            global_step_fn:
            save_dir: optional destination for .jpg images
        """
        self.tag = tag
        self.grid = grid
        self.logger = logger
        self.global_step_fn = global_step_fn
        self.predicate_vocabulary = Vocabulary(metadata.predicate_classes)

        self.img_dir = Path(data_root).expanduser().resolve() / metadata.image_root
        if not self.img_dir.is_dir():
            raise ValueError(f"Image dir must exist: {self.img_dir}")

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = Path(self.logger.logdir).expanduser().resolve()
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, engine: Engine):
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")

        global_step = self.global_step_fn()

        predicate_probs = engine.state.output["output"].sigmoid()
        targets_bce = engine.state.output["target"]
        filenames = engine.state.batch[2]

        fig, axes = plt.subplots(*self.grid, figsize=(16, 12), dpi=50)
        axes_iter: Iterator[plt.Axes] = axes.flat

        for target, pred, filename, ax in zip(
                targets_bce, predicate_probs, filenames, axes_iter
        ):
            # Some images are black and white, make sure they are read as RBG
            image = Image.open(self.img_dir.joinpath(filename)).convert("RGB")
            img_size = ImageSize(image.size[1], image.size[0])
            image = np.asarray(image)

            recall_at_5 = recall_at(target[None, :], pred[None, :], (5,))[5]
            mAP = mean_average_precision(target[None, :], pred[None, :])

            ax.imshow(image)
            ax.set_title(
                f"{Path(filename).name[:-4]} mAP {mAP:.1%} R@5 {recall_at_5:.1%}"
            )

            target_str = self.predicate_vocabulary.get_str(
                target.nonzero().flatten()
            ).tolist()
            ax.text(
                0.05,
                0.95,
                "\n".join(target_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="square", facecolor="white", alpha=0.8),
            )

            top_5 = torch.argsort(pred, descending=True)[:5]
            prediction_str = [
                f"{score:.1%} {str}"
                for score, str in zip(
                    pred[top_5], self.predicate_vocabulary.get_str(top_5)
                )
            ]
            ax.text(
                0.65,
                0.95,
                "\n".join(prediction_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="square", facecolor="white", alpha=0.8),
            )

            ax.tick_params(
                which="both",
                **{
                    k: False
                    for k in (
                        "bottom",
                        "top",
                        "left",
                        "right",
                        "labelbottom",
                        "labeltop",
                        "labelleft",
                        "labelright",
                    )
                },
            )
            ax.set_xlim(0, img_size.width)
            ax.set_ylim(img_size.height, 0)

        fig.tight_layout()

        if self.save_dir is not None:
            import io

            with io.BytesIO() as buff:
                fig.savefig(
                    buff, format="png", facecolor="white", bbox_inches="tight", dpi=50
                )
                pil_img = Image.open(buff).convert("RGB")
                plt.close(fig)
            save_path = self.save_dir.joinpath(f"{global_step}.{self.tag}.jpg")
            pil_img.save(save_path, "JPEG")
            self.logger.add_image(
                f"{self.tag}",
                np.moveaxis(np.asarray(pil_img), 2, 0),
                global_step=global_step,
            )
        else:
            self.logger.add_figure(
                f"{self.tag}", fig, global_step=global_step, close=True
            )


class VisualRelationPredictionLogger(object):
    def __init__(
            self,
            grid: Tuple[int, int],
            data_root: Union[str, Path],
            tag: str,
            logger: SummaryWriter,
            top_x_relations: int,
            global_step_fn: Callable[[], int],
            metadata: Metadata,
            save_dir: Optional[Union[str, Path]] = None,
    ):
        """

        Args:
            grid:
            img_dir: directory where the images will be opened from
            tag:
            logger: tensorboard logger for the images
            global_step_fn:
            save_dir: optional destination for .jpg images
        """
        self.tag = tag
        self.grid = grid
        self.logger = logger
        self.top_x_relations = top_x_relations
        self.global_step_fn = global_step_fn
        self.object_vocabulary = Vocabulary(metadata.thing_classes)
        self.predicate_vocabulary = Vocabulary(metadata.predicate_classes)

        self.img_dir = Path(data_root).expanduser().resolve() / metadata.image_root
        if not self.img_dir.is_dir():
            raise ValueError(f"Image dir must exist: {self.img_dir}")

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = Path(self.logger.logdir).expanduser().resolve()
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, engine: Engine):
        global_step = self.global_step_fn()
        targets = engine.state.batch[1]
        filenames = engine.state.batch[2]
        relations = engine.state.output["relations"]

        self._log_relations(relations, targets, filenames, global_step)

    def _log_relations(
            self,
            relations: Batch,
            targets: Batch,
            filenames: Sequence[str],
            global_step: int,
    ):
        # import matplotlib.pyplot as plt
        # plt.switch_backend('Agg')

        text = ""

        pred_node_offsets = [0] + relations.n_nodes[:-1].cumsum(dim=0).tolist()
        pred_relation_scores = torch.split_with_sizes(
            relations.relation_scores, relations.n_edges.tolist()
        )
        pred_predicate_scores = torch.split_with_sizes(
            relations.predicate_scores, relations.n_edges.tolist()
        )
        pred_predicate_classes = torch.split_with_sizes(
            relations.predicate_classes, relations.n_edges.tolist()
        )
        pred_relation_indexes = torch.split_with_sizes(
            relations.relation_indexes, relations.n_edges.tolist(), dim=1
        )

        gt_node_offsets = [0] + targets.n_nodes[:-1].cumsum(dim=0).tolist()
        gt_predicate_classes = torch.split_with_sizes(
            targets.predicate_classes, targets.n_edges.tolist()
        )
        gt_relation_indexes = torch.split_with_sizes(
            targets.relation_indexes, targets.n_edges.tolist(), dim=1
        )

        for b in range(min(relations.num_graphs, self.grid[0] * self.grid[1])):
            buffer = (
                f"{filenames[b]}\n"
                f"- input instances {relations.n_nodes[b].item()}\n"
                f"- (subj, obj) pairs {(relations.n_nodes[b] * (relations.n_nodes[b] - 1)).item()}\n\n"
            )

            top_x_relations = set()
            count_retrieved = 0
            buffer += f"Top {relations.n_edges[b].item()} predicted relations:\n"
            for i in range(relations.n_edges[b].item()):
                node_offset = pred_node_offsets[b]
                score = pred_relation_scores[b][i].item()

                subj_idx = pred_relation_indexes[b][0, i].item()
                obj_idx = pred_relation_indexes[b][1, i].item()
                predicate_score = pred_predicate_scores[b][i].item()
                predicate_class = pred_predicate_classes[b][i].item()
                predicate_str = self.predicate_vocabulary.get_str(predicate_class)

                subj_class = relations.object_classes[
                    pred_relation_indexes[b][0, i]
                ].item()
                obj_class = relations.object_classes[
                    pred_relation_indexes[b][1, i]
                ].item()
                subj_box = (
                    relations.object_boxes[pred_relation_indexes[b][0, i]]
                        .cpu()
                        .int()
                        .numpy()
                )
                obj_box = (
                    relations.object_boxes[pred_relation_indexes[b][1, i]]
                        .cpu()
                        .int()
                        .numpy()
                )
                subj_str = self.object_vocabulary.get_str(subj_class)
                obj_str = self.object_vocabulary.get_str(obj_class)

                top_x_relations.add(
                    (subj_class, subj_idx, predicate_class, obj_idx, obj_class)
                )
                buffer += (
                    f"{i + 1:3d} {score:.1e} : "
                    f"({subj_idx - node_offset:3d}) {subj_str:<14} "
                    f"{predicate_str:^14} "
                    f"{obj_str:>14} ({obj_idx - node_offset:3d})   "
                    f"{str(subj_box):<25} {predicate_score:>6.1%} {str(obj_box):>25}\n"
                )

            buffer += f"\nGround-truth relations:\n"
            for j in range(targets.n_edges[b].item()):
                node_offset = gt_node_offsets[b]

                subj_idx = gt_relation_indexes[b][0, j].item()
                obj_idx = gt_relation_indexes[b][1, j].item()
                predicate_class = gt_predicate_classes[b][j].item()
                predicate_str = self.predicate_vocabulary.get_str(predicate_class)

                subj_class = targets.object_classes[gt_relation_indexes[b][0, j]].item()
                obj_class = targets.object_classes[gt_relation_indexes[b][1, j]].item()
                subj_box = (
                    targets.object_boxes[gt_relation_indexes[b][0, j]]
                        .cpu()
                        .int()
                        .numpy()
                )
                obj_box = (
                    targets.object_boxes[gt_relation_indexes[b][1, j]]
                        .cpu()
                        .int()
                        .numpy()
                )
                subj_str = self.object_vocabulary.get_str(subj_class)
                obj_str = self.object_vocabulary.get_str(obj_class)

                # Assume the input boxes are from GT, not detectron, otherwise we'd have to match by IoU
                # TODO add matching by IoU
                retrieved = (
                                subj_class,
                                subj_idx,
                                predicate_class,
                                obj_idx,
                                obj_class,
                            ) in top_x_relations
                if retrieved:
                    count_retrieved += 1
                buffer += (
                    f'{"  OK️" if retrieved else "    "}        : '
                    f"({subj_idx - node_offset:3d}) {subj_str:<14} "
                    f"{predicate_str:^14} "
                    f"{obj_str:>14} ({obj_idx - node_offset:3d})   "
                    f"{str(subj_box):<25}        {str(obj_box):>25}\n"
                )

            buffer += f"\nRecall@{self.top_x_relations}: {count_retrieved / targets.n_edges[b].item():.2%}\n\n"

            text += textwrap.indent(buffer, "    ", lambda line: True) + "---\n\n"

        self.logger.add_text(
            f"Visual relations ({self.tag})", text, global_step=global_step
        )

        # fig, axes = plt.subplots(*self.grid, figsize=(16, 12), dpi=50)
        # axes_iter: Iterator[plt.Axes] = axes.flat

        # for target, pred, filename, ax in zip(targets_bce, predicate_probs, filenames, axes_iter):
        #     image = cv2.imread(self.img_dir.joinpath(filename).as_posix())
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     img_size = ImageSize(*image.shape[:2])
        #
        #     recall_at_5 = recall_at(target[None, :], pred[None, :], (5,))[5]
        #     mAP = mean_average_precision(target[None, :], pred[None, :])
        #
        #     ax.imshow(image)
        #     ax.set_title(f'{filename[:-4]} mAP {mAP:.1%} R@5 {recall_at_5:.1%}')
        #
        #     target_str = self.predicate_vocabulary.get_str(target.nonzero().flatten()).tolist()
        #     ax.text(
        #         0.05, 0.95,
        #         '\n'.join(target_str),
        #         transform=ax.transAxes,
        #         fontsize=11,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
        #     )
        #
        #     top_5 = torch.argsort(pred, descending=True)[:5]
        #     prediction_str = [f'{score:.1%} {str}' for score, str
        #                       in zip(pred[top_5], self.predicate_vocabulary.get_str(top_5))]
        #     ax.text(
        #         0.65, 0.95,
        #         '\n'.join(prediction_str),
        #         transform=ax.transAxes,
        #         fontsize=11,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
        #     )
        #
        #     ax.tick_params(which='both', **{k: False for k in ('bottom', 'top', 'left', 'right',
        #                                                        'labelbottom', 'labeltop', 'labelleft', 'labelright')})
        #     ax.set_xlim(0, img_size.width)
        #     ax.set_ylim(img_size.height, 0)
        #
        # fig.tight_layout()
        #
        # if self.save_dir is not None:
        #     import io
        #     from PIL import Image
        #     with io.BytesIO() as buff:
        #         fig.savefig(buff, format='png', facecolor='white', bbox_inches='tight', dpi=50)
        #         pil_img = Image.open(buff).convert('RGB')
        #         plt.close(fig)
        #     save_path = self.save_dir.joinpath(f'{global_step}.{self.tag}.jpg')
        #     pil_img.save(save_path, 'JPEG')
        #     self.logger.add_image(f'{self.tag}', np.moveaxis(np.asarray(pil_img), 2, 0), global_step=global_step)
        # else:
        #     self.logger.add_figure(f'{self.tag}', fig, global_step=global_step, close=True)


class HoiClassificationMeanAvgPrecision(Metric):
    """Mean Average Precision over the 600 human-object interactions defined in HICO.

    A human-object interaction is a triplet (person, predicate, object). This mAP is
    only computed at image level, i.e. the interaction is either present or not.
    If multiple visual relations with matching predicate and object are predicted
    for a single image we keep the prediction with the highest score.
    Scores are used to rank interactions and compute AP thresholds.
    """

    gt: List[Dict[Tuple[int, int], bool]]
    pred: List[Dict[Tuple[int, int], float]]

    _required_output_keys = ("relations", "targets")

    def reset(self):
        self.gt = []
        self.pred = []

    def update(self, output):
        relations = output[0]
        targets = output[1]

        sizes = relations.n_edges.tolist()
        for subjs, preds, objs, rel_scores in zip(
                torch.split_with_sizes(
                    relations.object_classes[relations.relation_indexes[0]], sizes
                ),
                torch.split_with_sizes(relations.predicate_classes, sizes),
                torch.split_with_sizes(
                    relations.object_classes[relations.relation_indexes[1]], sizes
                ),
                torch.split_with_sizes(relations.relation_scores, sizes),
        ):
            graph_hois = {}
            for subj, pred, obj, hoi_score in zip(subjs, preds, objs, rel_scores):
                if subj.item() != 0:
                    continue
                hoi = (pred.item(), obj.item())
                if hoi_score.item() > graph_hois.get(hoi, -1):
                    graph_hois[hoi] = hoi_score.item()
            self.pred.append(graph_hois)

        sizes = targets.n_edges.tolist()
        for subjs, preds, objs in zip(
                torch.split_with_sizes(
                    targets.object_classes[targets.relation_indexes[0]], sizes
                ),
                torch.split_with_sizes(targets.predicate_classes, sizes),
                torch.split_with_sizes(
                    targets.object_classes[targets.relation_indexes[1]], sizes
                ),
        ):
            graph_hois = {}
            for subj, pred, obj in zip(subjs, preds, objs):
                if subj.item() != 0:
                    continue
                hoi = (pred.item(), obj.item())
                graph_hois[hoi] = True
            self.gt.append(graph_hois)

    def compute(self):
        from xib.datasets.hico_det.metadata import HOI

        # These dataframes have shape [num_images, num_hois]
        gt = (
            pd.DataFrame(self.gt)
                .fillna(False, downcast={object: bool})
                .reindex(columns=HOI, fill_value=0)
        )
        pred = pd.DataFrame(self.pred).fillna(0).reindex(columns=HOI, fill_value=0)

        return mean_average_precision(gt.values.astype(np.int), pred.values)


class HoiDetectionMeanAvgPrecision(Metric):
    """Mean Average Precision over the 600 human-object interactions defined in HICO.

    A human-object interaction is a triplet (person, predicate, object).
    Subj and obj of a triplet must overlap with subj and obj of a ground-truth triplet
    to count as true positives.
    """

    # Update strategy:
    # - iterate one image at the time,
    #   considering the predicted relations and ground-truth relations
    #     - group GT relations by HOI and iterate over every group
    #         - filter predicted relations based on the current GT hoi
    #             - iterate over predicted boxes in descending order of score,
    #               greedily match the current predicted box with the GT box that
    #               has the largest overlap, but at least .5 IoU
    #             - for each predicted relation, keep track of whether it was
    #               matched (true positive) or not (false positives)
    #         - store all scores and all matches for each hoi in a long list
    #         - also keep a count of how many GT boxes are there for
    #           each HOI (true positives + false negatives)
    #
    # Compute strategy:
    # - one hoi at the time
    #   - take the list of all scores and the list that says
    #     whether a box matched or not
    #   - sort in ascending order according to the scores
    #   - compute precision and recall at every position,
    #     add a [1] and a [0] at the respective ends
    #   - compute AP using 11 point interpolated precision

    hoi_dicts = Dict[Tuple[int, int], Dict[str, Union[List[np.ndarray], int]]]

    _required_output_keys = ("relations", "targets")

    def reset(self):
        from xib.datasets.hico_det.metadata import HOI
        self.hoi_dicts = {
            hoi: {
                'scores': [],
                'true_positives': [],
                'total_relevant': 0,
            }
            for hoi in HOI
        }

    def update(self, output):
        predictions = output[0]
        targets = output[1]

        pr_sizes = predictions.n_edges.tolist()
        gt_sizes = targets.n_edges.tolist()

        # Predictions: subj_cls, subj_box, pred_cls, obj_cls, obj_box, scores (desc)
        predictions = (pd.DataFrame({
            'subj_cls': subj_cls.numpy(),
            'subj_box': subj_box.numpy().tolist(),
            'pred_cls': pred_cls.numpy(),
            'obj_cls': obj_cls.numpy(),
            'obj_box': obj_box.numpy().tolist(),
            'scores': scores.numpy(),
        }) for subj_cls, subj_box, pred_cls, obj_cls, obj_box, scores in zip(
            torch.split(predictions.object_classes[predictions.relation_indexes[0]], pr_sizes),
            torch.split(predictions.object_boxes[predictions.relation_indexes[0]], pr_sizes),
            torch.split(predictions.predicate_classes, pr_sizes),
            torch.split(predictions.object_classes[predictions.relation_indexes[1]], pr_sizes),
            torch.split(predictions.object_boxes[predictions.relation_indexes[1]], pr_sizes),
            torch.split(predictions.relation_scores, pr_sizes)
        ))

        # Targets: subj_cls, subj_box, pred_cls, obj_cls, obj_box
        targets = (pd.DataFrame({
            'subj_cls': subj_cls.numpy(),
            'subj_box': subj_box.numpy().tolist(),
            'pred_cls': pred_cls.numpy(),
            'obj_cls': obj_cls.numpy(),
            'obj_box': obj_box.numpy().tolist(),
        }) for subj_cls, subj_box, pred_cls, obj_cls, obj_box in zip(
            torch.split(targets.object_classes[targets.relation_indexes[0]], gt_sizes),
            torch.split(targets.object_boxes[targets.relation_indexes[0]], gt_sizes),
            torch.split(targets.predicate_classes, gt_sizes),
            torch.split(targets.object_classes[targets.relation_indexes[1]], gt_sizes),
            torch.split(targets.object_boxes[targets.relation_indexes[1]], gt_sizes),
        ))

        for pr, gt in zip(predictions, targets):
            for hoi, gt_hoi in gt.groupby(['subj_cls', 'pred_cls', 'obj_cls']):
                self.hoi_dicts[hoi[1:]]['total_relevant'] += len(gt_hoi)

                pr_hoi = pr.query(f"subj_cls == {hoi[0]} and pred_cls == {hoi[1]} and obj_cls == {hoi[2]}")
                if len(pr_hoi) == 0:
                    continue

                # Match predicted boxes with gt boxes keeping track of matches (true positives)
                ious = torch.min(
                    box_iou(
                        torch.tensor(pr_hoi.subj_box.tolist()),
                        torch.tensor(gt_hoi.subj_box.tolist())
                    ),
                    box_iou(
                        torch.tensor(pr_hoi.obj_box.tolist()),
                        torch.tensor(gt_hoi.obj_box.tolist())
                    )
                )

                pr_matched = np.full(len(pr_hoi), fill_value=False)
                gt_matched_total = 0

                for pr_i in range(len(pr_hoi)):
                    iou_best, gt_j = ious[pr_i, :].max(dim=0)

                    if iou_best > .5:
                        pr_matched[pr_i] = True
                        ious[:, gt_j] = -1
                        gt_matched_total += 1

                        if gt_matched_total == len(gt_hoi):
                            break

                self.hoi_dicts[hoi[1:]]['scores'].append(pr_hoi.scores.values)
                self.hoi_dicts[hoi[1:]]['true_positives'].append(pr_matched)

    def compute(self):
        from xib.datasets.hico_det.metadata import HOI

        ap_dict = {hoi: np.nan for hoi in HOI}

        for hoi, hoi_dict in self.hoi_dicts.items():
            if len(hoi_dict['scores']) == 0:
                if hoi_dict['total_relevant'] > 0:
                    ap_dict[hoi] = 0
                continue
            scores = np.concatenate(hoi_dict['scores'])
            true_positives = np.concatenate(hoi_dict['true_positives'])
            total_relevant = hoi_dict['total_relevant']

            # Sort all scores in descending order
            sort_idx = np.argsort(scores)[::-1]
            true_positives = true_positives[sort_idx]
            cumsum = true_positives.cumsum()

            # This way, recall will monotonically increase
            # and precision will wiggle a bit
            precisions = cumsum / np.arange(1, len(true_positives) + 1)
            recalls = cumsum / total_relevant
            # ap = self._sklearn_ap(precisions, recalls)
            # ap = self._monotonically_decreasing_ap(precisions, recalls)
            ap = self._11_points_interpolated_precision(precisions, recalls)

            ap_dict[hoi] = ap

        return np.nanmean(list(ap_dict.values()))

    @staticmethod
    def _sklearn_ap(precisions, recalls):
        # For the numerical integration to work, we need
        # recalls in descending order and precisions
        # in ascending order, just like the return value
        # of sklearn.metrics.precision_recall_curve
        precisions = precisions[::-1]
        recalls = recalls[::-1]

        precisions = precisions.tolist() + [1.]
        recalls = recalls.tolist() + [0.]
        ap = -np.sum(np.diff(recalls) * np.array(precisions)[:-1])
        return ap

    @staticmethod
    def _monotonically_decreasing_ap(precisions, recalls):
        precisions = precisions[::-1]
        recalls = recalls[::-1]

        interpolated_precisions = np.maximum.accumulate(precisions).tolist() + [1.]
        recalls = recalls.tolist() + [0.]
        ap = -np.sum(np.diff(recalls) * np.array(interpolated_precisions)[:-1])
        return ap

    @staticmethod
    def _11_points_interpolated_precision(precisions, recalls):
        ap = 0.
        for t in np.linspace(0, 1, num=11, endpoint=True):
            recall_mask = recalls >= t
            if recall_mask.any():
                p = np.max(precisions[recall_mask])
                ap += p / 11
            # else:
            #     p = 0
            #     ap += p / 11
        return ap


class VisualRelationRecallAt(object):
    class Mode(NamedEnumMixin, Enum):
        def __init__(self, short_name):
            self.short_name = short_name

    def __init__(
            self, type: Union[str, VisualRelationRecallAt.Mode], steps: Tuple[int, ...]
    ):
        if type == "predicate":
            self._compute_matches = self._predicate_detection
        elif type == "phrase":
            self._compute_matches = self._phrase_detection
        elif type == "relationship":
            self._compute_matches = self._relationship_detection
        else:
            raise ValueError(f"Invalid visual relations matching type: {type}")

        self.matching_type = type
        self.steps = torch.tensor(sorted(steps))

    def __call__(self, engine: Engine):
        targets = engine.state.batch[1]

        predictions = engine.state.output["relations"]
        matches = self._compute_matches(predictions, targets)
        recall_at = self._recall_at(predictions, targets, matches)

        for k, r in recall_at.items():
            r = torch.full((predictions.num_graphs, 1), fill_value=r)
            engine.state.output[f"vr/{self.matching_type}/recall_at_{k}"] = r

    @staticmethod
    def _predicate_detection(predictions: Batch, targets: Batch) -> torch.Tensor:
        """Computes matches based on "predicate detection" mode."""
        # [E_t, 5]
        gt_matrix = torch.stack(
            [
                # subject_idx, object_idx
                targets.batch[targets.relation_indexes[0]],
                targets.batch[targets.relation_indexes[1]],
                # subject_class, predicate_class, object_class
                targets.object_classes[targets.relation_indexes[0]],
                targets.predicate_classes,
                targets.object_classes[targets.relation_indexes[1]],
            ],
            dim=1,
        )

        # [E_p, 5]
        pred_matrix = torch.stack(
            [
                # subject_idx, object_idx
                predictions.batch[predictions.relation_indexes[0]],
                predictions.batch[predictions.relation_indexes[1]],
                # subject_class, predicate_class, object_class
                predictions.object_classes[predictions.relation_indexes[0]],
                predictions.predicate_classes,
                predictions.object_classes[predictions.relation_indexes[1]],
            ],
            dim=1,
        )

        # Block matrix [E_p, E_t]
        matches = (gt_matrix[None, :, :] == pred_matrix[:, None, :]).all(dim=2)

        return matches

    @staticmethod
    def _phrase_detection(predictions: Batch, targets: Batch) -> torch.Tensor:
        """Computes matches based on "phrase detection" mode."""
        # [E_p, 4]
        pred_matrix = torch.stack(
            [
                # graph_idx
                predictions.batch[predictions.relation_indexes[0]],
                # subject_class, predicate_class, object_class
                predictions.object_classes[predictions.relation_indexes[0]],
                predictions.predicate_classes,
                predictions.object_classes[predictions.relation_indexes[1]],
            ],
            dim=1,
        )

        # [E_t, 4]
        gt_matrix = torch.stack(
            [
                # graph_idx
                targets.batch[targets.relation_indexes[0]],
                # subject_class, predicate_class, object_class
                targets.object_classes[targets.relation_indexes[0]],
                targets.predicate_classes,
                targets.object_classes[targets.relation_indexes[1]],
            ],
            dim=1,
        )

        # Block matrix [E_p, E_t]
        matches_class = (gt_matrix[None, :, :] == pred_matrix[:, None, :]).all(dim=2)

        # [E_p, 4]
        pred_union_boxes = matched_boxlist_union(
            predictions.object_boxes[predictions.relation_indexes[0]],
            predictions.object_boxes[predictions.relation_indexes[1]],
        )

        # [E_t, 4]
        gt_union_boxes = matched_boxlist_union(
            targets.object_boxes[targets.relation_indexes[0]],
            targets.object_boxes[targets.relation_indexes[1]],
        )

        # Full matrix [E_p, E_t]
        iou_union = box_iou(pred_union_boxes, gt_union_boxes)

        # Block matrix [E_p, E_t]
        matches = matches_class & (iou_union > 0.5)

        return matches

    @staticmethod
    def _relationship_detection(predictions: Batch, targets: Batch) -> torch.Tensor:
        """Computes matches based on "relationship detection" mode"""
        # [E_p, 4]
        pred_matrix = torch.stack(
            [
                # graph_idx
                predictions.batch[predictions.relation_indexes[0]],
                # subject_class, predicate_class, object_class
                predictions.object_classes[predictions.relation_indexes[0]],
                predictions.predicate_classes,
                predictions.object_classes[predictions.relation_indexes[1]],
            ],
            dim=1,
        )

        # [E_t, 4]
        gt_matrix = torch.stack(
            [
                # graph_idx
                targets.batch[targets.relation_indexes[0]],
                # subject_class, predicate_class, object_class
                targets.object_classes[targets.relation_indexes[0]],
                targets.predicate_classes,
                targets.object_classes[targets.relation_indexes[1]],
            ],
            dim=1,
        )

        # Block matrix [E_p, E_t]
        matches_class = (gt_matrix[None, :, :] == pred_matrix[:, None, :]).all(dim=2)

        # Two full matrices [E_p, E_t]
        iou_subject = box_iou(
            predictions.object_boxes[predictions.relation_indexes[0]],
            targets.object_boxes[targets.relation_indexes[0]],
        )
        iou_object = box_iou(
            predictions.object_boxes[predictions.relation_indexes[1]],
            targets.object_boxes[targets.relation_indexes[1]],
        )

        # Block matrix [E_p, E_t]
        matches = matches_class & (iou_subject > 0.5) & (iou_object > 0.5)

        return matches

    def _recall_at(
            self, predictions: Batch, targets: Batch, matches: torch.Tensor
    ) -> Dict[int, float]:
        # matches.argmax(dim=0) will return the last index if no True value is found.
        # We can use matches.any(dim=0) to ignore those cases.
        # Also, we must account for the row offset in the matches matrix.
        gt_retrieved = matches.any(dim=0)

        offset = (
                predictions.n_edges.cumsum(dim=0).repeat_interleave(targets.n_edges)
                - predictions.n_edges[0]
        )
        gt_retrieved_rank = matches.int().argmax(dim=0) - offset

        # [K, E_t]
        gt_retrieved_at = (
                                  gt_retrieved_rank[None, :] < self.steps[:, None]
                          ) & gt_retrieved[None, :]

        # [K, num_graphs]
        gt_relation_to_graph_assignment = targets.batch[targets.relation_indexes[0]]
        recall_at_per_graph = scatter_(
            "mean",
            gt_retrieved_at.float(),
            index=gt_relation_to_graph_assignment,
            dim=1,
            dim_size=targets.num_graphs,
        )

        # [K]
        recall_at = recall_at_per_graph.mean(dim=1)

        return {k: v for k, v in zip(self.steps.numpy(), recall_at.numpy())}
