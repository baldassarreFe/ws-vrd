import json
import textwrap
from pathlib import Path
from typing import Tuple, Union, Callable, Optional, Sequence

import torch
import numpy as np
import pandas as pd
from detectron2.data.catalog import Metadata
from ignite.engine import Engine
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch

from xib.structures import Vocabulary
from xib.structures.relations import split_vr_batch


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


class VisualRelationPredictionExporter(object):
    _required_output_keys = ("targets", "relations", "filenames")

    def __init__(self, dest: Union[str, Path], mode: str, iou_threshold: float = 0.5):
        if mode == "GT":
            self.mode = mode
            self.modes = ("pred_det",)
        elif mode == "D2":
            self.mode = mode
            self.modes = ("phrase_det", "relation_det")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.iou_threshold = iou_threshold
        self.dest = Path(dest).expanduser().resolve() / f"detected_triplets_{mode}.json"
        if self.dest.is_file():
            raise FileExistsError(f"Refusing to overwrite existing file: {self.dest}")
        super(VisualRelationPredictionExporter, self).__init__()

    def __call__(self, engine: Engine):
        targets = engine.state.batch[1]
        filenames = engine.state.batch[2]
        predictions = engine.state.output["relations"]

        self._export_relations(predictions, targets, filenames)

    def _export_relations(self, predictions, targets, filenames):
        with open(self.dest, "a") as f:
            for pred, filename in zip(split_vr_batch(predictions), filenames):
                annotations = [
                    {
                        "box_idx": idx,
                        "category_id": cls.item(),
                        "bbox": box.tolist(),
                        "score": sc.item(),
                    }
                    for idx, (cls, box, sc) in enumerate(
                        zip(pred.object_classes, pred.object_boxes, pred.object_scores)
                    )
                ]
                relations = [
                    {
                        "category_id": pred.item(),
                        "subject_idx": subj_idx.item(),
                        "object_idx": obj_idx.item(),
                        "relation_score": rel_sc.item(),
                        "predicate_score": pred_sc.item(),
                    }
                    for pred, subj_idx, obj_idx, rel_sc, pred_sc in zip(
                        pred.predicate_classes,
                        pred.relation_indexes[0],
                        pred.relation_indexes[1],
                        pred.relation_scores,
                        pred.predicate_scores,
                    )
                ]

                data_dict = {
                    "file_name": filename,
                    "height": pred.object_image_size[0, 0].item(),
                    "width": pred.object_image_size[0, 1].item(),
                    "boxes_origin": self.mode,
                    "annotations": annotations,
                    "relations": relations,
                }

                f.write(json.dumps(data_dict) + "\n")
