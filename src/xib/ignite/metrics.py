from __future__ import annotations

import textwrap
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Iterator, Callable, Sequence

import cv2
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from detectron2.data.catalog import Metadata
from ignite.engine import Events, Engine
from ignite.metrics import Metric
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.utils import scatter_
from torchvision.ops import box_iou

from ..structures import ImageSize, Vocabulary, matched_boxlist_union
from ..utils.utils import NamedEnumMixin


class BatchMetric(Metric, ABC):
    """A metric computed independently for every batch."""

    def attach(self, engine, name):
        # Reset at the beginning of every iteration
        if not engine.has_event_handler(self.started, Events.ITERATION_STARTED):
            engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        # Update at the after every iteration
        if not engine.has_event_handler(
            self.iteration_completed, Events.ITERATION_COMPLETED
        ):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, self.iteration_completed
            )
        # Copy metric to engine.state.metrics after every iteration
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


def mean_average_precision(annotations, scores) -> float:
    """Computes the mean average precision (mAP) for a multi-class multi-label scenario.

    In object detection mAP is the average AP across all classes, i.e.
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

    return np.nanmean(average_precisions).item()


class MeanAveragePrecisionBatch(BatchMetric):
    avg_precision: float

    def reset(self):
        self.avg_precision = float("NaN")

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_true, y_score = output
        self.avg_precision = mean_average_precision(y_true, y_score)

    def compute(self):
        return self.avg_precision


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
    # Then we take the batch mean.
    for s in sizes:
        result[s] = (cumsum[:, (s - 1)] / s).mean(dim=0).item()

    return result


def recall_at(annotations, scores, sizes):
    """Recall@x

    - rank the relationships by their score and keep the top x
    - compute how many of the relevant relationships are among the retrieved

    ::

                # ( relevant items retrieved )
      Recall  = ------------------------------ = P ( retrieved | relevant )
                     # ( relevant items )

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
    # Cast to float to avoid int/int division.
    cumsum = annotations_of_top_max_s.cumsum(dim=1).float()

    # Divide each row by the total number of relevant document for that row to get the recall per sample.
    # Then take the batch mean.
    num_rel = annotations.sum(dim=1, keepdims=True)
    recall = cumsum / num_rel
    for s in sizes:
        result[s] = recall[:, s - 1].mean(axis=0).item()

    return result


class RecallAtBatch(BatchMetric):
    """Recall@x over the output of the last batch"""

    _recall_at: Dict[int, Optional[float]]

    def __init__(
        self,
        sizes: Tuple[int, ...] = (10, 30, 50),
        output_transform=lambda x: x,
        device=None,
    ):
        self._sorted_sizes = list(sorted(sizes))
        super(RecallAtBatch, self).__init__(output_transform, device)

    def reset(self):
        self._recall_at = {s: None for s in self._sorted_sizes}

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_true, y_score = output
        self._recall_at.update(recall_at(y_true, y_score, self._sorted_sizes))

    def compute(self):
        return self._recall_at

    def completed(self, engine, name):
        result = self.compute()
        for k, v in result.items():
            engine.state.metrics[f"{name}_{k}"] = v


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
            image = cv2.imread(self.img_dir.joinpath(filename).as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_size = ImageSize(*image.shape[:2])

            recall_at_5 = recall_at(target[None, :], pred[None, :], (5,))[5]
            mAP = mean_average_precision(target[None, :], pred[None, :])

            ax.imshow(image)
            ax.set_title(f"{filename[:-4]} mAP {mAP:.1%} R@5 {recall_at_5:.1%}")

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
            from PIL import Image

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
        relations = engine.state.output["relations"]
        targets = engine.state.batch[1]
        filenames = engine.state.batch[2]

        for mode in ("with_obj_scores", "no_obj_scores"):
            self._log_relations(relations[mode], targets, filenames, global_step)

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


class HOImAP(Metric):
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
        relations = output[0]["with_obj_scores"]
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

        gt = (
            pd.DataFrame(self.gt)
            .fillna(False, downcast={object: bool})
            .reindex(columns=HOI, fill_value=0)
        )
        pred = pd.DataFrame(self.pred).fillna(0).reindex(columns=HOI, fill_value=0)

        return mean_average_precision(gt.values.astype(np.int), pred.values)


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

        for mode in ("with_obj_scores", "no_obj_scores"):
            predictions = engine.state.output["relations"][mode]
            matches = self._compute_matches(predictions, targets)
            recall_at = self._recall_at(predictions, targets, matches)
            for k, r in recall_at.items():
                engine.state.output[f"{self.matching_type}/{mode}/recall_at_{k}"] = r

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
