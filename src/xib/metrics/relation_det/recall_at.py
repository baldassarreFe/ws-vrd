from typing import Union, Tuple, Dict

import torch
from ignite.engine import Engine
from torch_geometric.data import Batch
from torch_geometric.utils import scatter_
from torchvision.ops import box_iou

from xib.structures import matched_boxlist_union


class VisualRelationRecallAt(object):
    def __init__(self, type: Union[str], steps: Tuple[int, ...]):
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
