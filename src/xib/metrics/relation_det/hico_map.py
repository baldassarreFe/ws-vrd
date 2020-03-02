from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ignite.metrics import Metric
from torchvision.ops import box_iou

from ..pred_class.mean_avg_prec import mean_average_precision


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
            .reindex(columns=HOI.keys(), fill_value=0)
        )
        pred = (
            pd.DataFrame(self.pred).fillna(0).reindex(columns=HOI.keys(), fill_value=0)
        )

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
            hoi: {"scores": [], "true_positives": [], "total_relevant": 0}
            for hoi in HOI.keys()
        }

    def update(self, output):
        predictions = output[0]
        targets = output[1]

        pr_sizes = predictions.n_edges.tolist()
        gt_sizes = targets.n_edges.tolist()

        # Predictions: subj_cls, subj_box, pred_cls, obj_cls, obj_box, scores (desc)
        predictions = (
            pd.DataFrame(
                {
                    "subj_cls": subj_cls.numpy(),
                    "subj_box": subj_box.numpy().tolist(),
                    "pred_cls": pred_cls.numpy(),
                    "obj_cls": obj_cls.numpy(),
                    "obj_box": obj_box.numpy().tolist(),
                    "scores": scores.numpy(),
                }
            )
            for subj_cls, subj_box, pred_cls, obj_cls, obj_box, scores in zip(
                torch.split(
                    predictions.object_classes[predictions.relation_indexes[0]],
                    pr_sizes,
                ),
                torch.split(
                    predictions.object_boxes[predictions.relation_indexes[0]], pr_sizes
                ),
                torch.split(predictions.predicate_classes, pr_sizes),
                torch.split(
                    predictions.object_classes[predictions.relation_indexes[1]],
                    pr_sizes,
                ),
                torch.split(
                    predictions.object_boxes[predictions.relation_indexes[1]], pr_sizes
                ),
                torch.split(predictions.relation_scores, pr_sizes),
            )
        )

        # Targets: subj_cls, subj_box, pred_cls, obj_cls, obj_box
        targets = (
            pd.DataFrame(
                {
                    "subj_cls": subj_cls.numpy(),
                    "subj_box": subj_box.numpy().tolist(),
                    "pred_cls": pred_cls.numpy(),
                    "obj_cls": obj_cls.numpy(),
                    "obj_box": obj_box.numpy().tolist(),
                }
            )
            for subj_cls, subj_box, pred_cls, obj_cls, obj_box in zip(
                torch.split(
                    targets.object_classes[targets.relation_indexes[0]], gt_sizes
                ),
                torch.split(
                    targets.object_boxes[targets.relation_indexes[0]], gt_sizes
                ),
                torch.split(targets.predicate_classes, gt_sizes),
                torch.split(
                    targets.object_classes[targets.relation_indexes[1]], gt_sizes
                ),
                torch.split(
                    targets.object_boxes[targets.relation_indexes[1]], gt_sizes
                ),
            )
        )

        for pr, gt in zip(predictions, targets):
            for hoi, gt_hoi in gt.groupby(["subj_cls", "pred_cls", "obj_cls"]):
                self.hoi_dicts[hoi[1:]]["total_relevant"] += len(gt_hoi)

                pr_hoi = pr.query(
                    f"subj_cls == {hoi[0]} and pred_cls == {hoi[1]} and obj_cls == {hoi[2]}"
                )
                if len(pr_hoi) == 0:
                    continue

                # Match predicted boxes with gt boxes keeping track of matches (true positives)
                ious = torch.min(
                    box_iou(
                        torch.tensor(pr_hoi.subj_box.tolist()),
                        torch.tensor(gt_hoi.subj_box.tolist()),
                    ),
                    box_iou(
                        torch.tensor(pr_hoi.obj_box.tolist()),
                        torch.tensor(gt_hoi.obj_box.tolist()),
                    ),
                )

                pr_matched = np.full(len(pr_hoi), fill_value=False)
                gt_matched_total = 0

                for pr_i in range(len(pr_hoi)):
                    iou_best, gt_j = ious[pr_i, :].max(dim=0)

                    if iou_best > 0.5:
                        pr_matched[pr_i] = True
                        ious[:, gt_j] = -1
                        gt_matched_total += 1

                        if gt_matched_total == len(gt_hoi):
                            break

                self.hoi_dicts[hoi[1:]]["scores"].append(pr_hoi.scores.values)
                self.hoi_dicts[hoi[1:]]["true_positives"].append(pr_matched)

    def compute(self):
        from xib.datasets.hico_det.metadata import HOI

        # Initialize all APs to 0
        ap_dict = {hoi: 0 for hoi in HOI.keys()}

        for hoi, hoi_dict in self.hoi_dicts.items():
            if len(hoi_dict["scores"]) == 0:
                if hoi_dict["total_relevant"] == 0:
                    ap_dict[hoi] = np.nan
                continue
            scores = np.concatenate(hoi_dict["scores"])
            true_positives = np.concatenate(hoi_dict["true_positives"])
            total_relevant = hoi_dict["total_relevant"]

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

        aps = np.array(list(ap_dict.values()))
        rare = np.array(list(HOI.values())) < 10
        assert rare.sum() == 138

        return {
            "all": np.nanmean(aps),
            "rare": np.nanmean(aps[rare]),
            "nonrare": np.nanmean(aps[~rare]),
        }

    def completed(self, engine, name):
        result = self.compute()
        for k, v in result.items():
            engine.state.metrics[f"{name}/{k}"] = v

    @staticmethod
    def _sklearn_ap(precisions, recalls):
        # For the numerical integration to work, we need
        # recalls in descending order and precisions
        # in ascending order, just like the return value
        # of sklearn.metrics.precision_recall_curve
        precisions = precisions[::-1]
        recalls = recalls[::-1]

        precisions = precisions.tolist() + [1.0]
        recalls = recalls.tolist() + [0.0]
        ap = -np.sum(np.diff(recalls) * np.array(precisions)[:-1])
        return ap

    @staticmethod
    def _monotonically_decreasing_ap(precisions, recalls):
        precisions = precisions[::-1]
        recalls = recalls[::-1]

        interpolated_precisions = np.maximum.accumulate(precisions).tolist() + [1.0]
        recalls = recalls.tolist() + [0.0]
        ap = -np.sum(np.diff(recalls) * np.array(interpolated_precisions)[:-1])
        return ap

    @staticmethod
    def _11_points_interpolated_precision(precisions, recalls):
        ap = 0.0
        for t in np.linspace(0, 1, num=11, endpoint=True):
            recall_mask = recalls >= t
            if recall_mask.any():
                p = np.max(precisions[recall_mask])
                ap += p / 11
            # else:
            #     p = 0
            #     ap += p / 11
        return ap
