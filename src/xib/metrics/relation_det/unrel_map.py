from typing import Dict, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from ignite.metrics import Metric
from torchvision.ops import box_iou

from xib.structures import matched_boxlist_union


class UnRelDetectionMeanAvgPrecision(Metric):
    """Mean Average Precision over the triplets defined in UnRel.

    Cases:
    - GT boxes
      - match by subj,pred,obj classes only
    - D2 boxes (match by subj,pred,obj classes):
      - iou(subj_det, subj_true) > iou_threshold
      - iou(subj_det, subj_true) > iou_threshold
      - iou(subj_det, subj_true) > iou_threshold
    """

    unrel_dicts = Dict[
        Tuple[int, int, int], Dict[str, Union[Dict[str, List[np.ndarray]], int]]
    ]

    _required_output_keys = ("relations", "targets")

    def __init__(self, mode: str, iou_threshold: float = 0.3):
        if mode == "GT":
            self.mode = mode
            self.modes = ("GT",)
        elif mode == "D2":
            self.mode = mode
            self.modes = ("D2_subj", "D2_union", "D2_subj_obj")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.iou_threshold = iou_threshold
        super(UnRelDetectionMeanAvgPrecision, self).__init__()

    def reset(self):
        from xib.datasets.unrel.metadata import UNUSUAL_RELATIONS

        self.unrel_dicts = {
            unrel_spo: {
                "scores": [],
                "true_positives": {mode: [] for mode in self.modes},
                "total_relevant": 0,
            }
            for unrel_spo in UNUSUAL_RELATIONS
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
            for spo, gt_spo in gt.groupby(["subj_cls", "pred_cls", "obj_cls"]):
                if spo not in self.unrel_dicts:
                    # Detected relationship but not an unusual one from UnRel
                    continue

                self.unrel_dicts[spo]["total_relevant"] += len(gt_spo)

                pr_spo = pr.query(
                    f"subj_cls == {spo[0]} and pred_cls == {spo[1]} and obj_cls == {spo[2]}"
                )
                if len(pr_spo) == 0:
                    continue

                # Match predicted boxes with gt boxes keeping track of matches (true positives)
                if self.mode == "GT":
                    ious_subj = box_iou(
                        torch.tensor(pr_spo.subj_box.tolist()),
                        torch.tensor(gt_spo.subj_box.tolist()),
                    )
                    ious_obj = box_iou(
                        torch.tensor(pr_spo.obj_box.tolist()),
                        torch.tensor(gt_spo.obj_box.tolist()),
                    )
                    ious_subj_obj = torch.min(ious_subj, ious_obj)
                elif self.mode == "D2":
                    pr_union = matched_boxlist_union(
                        torch.tensor(pr_spo.subj_box.tolist()),
                        torch.tensor(pr_spo.obj_box.tolist()),
                    )
                    gt_union = matched_boxlist_union(
                        torch.tensor(gt_spo.subj_box.tolist()),
                        torch.tensor(gt_spo.obj_box.tolist()),
                    )
                    ious_union = box_iou(pr_union, gt_union)
                    ious_subj = box_iou(
                        torch.tensor(pr_spo.subj_box.tolist()),
                        torch.tensor(gt_spo.subj_box.tolist()),
                    )
                    ious_obj = box_iou(
                        torch.tensor(pr_spo.obj_box.tolist()),
                        torch.tensor(gt_spo.obj_box.tolist()),
                    )
                    ious_subj_obj = torch.min(ious_subj, ious_obj)
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")

                pr_matched = {
                    mode: np.full(len(pr_spo), fill_value=False) for mode in self.modes
                }

                for pr_i in range(len(pr_spo)):
                    if self.mode == "GT":
                        iou_best, gt_j = ious_subj_obj[pr_i, :].max(dim=0)
                        if iou_best >= 0.99:
                            pr_matched["GT"][pr_i] = True
                            ious_subj_obj[:, gt_j] = -1
                    elif self.mode == "D2":
                        # D2 union
                        iou_best, gt_j = ious_subj[pr_i, :].max(dim=0)
                        if iou_best > 0.3:
                            pr_matched["D2_subj"][pr_i] = True
                            ious_subj[:, gt_j] = -1

                        # D2 union
                        iou_best, gt_j = ious_union[pr_i, :].max(dim=0)
                        if iou_best > 0.3:
                            pr_matched["D2_union"][pr_i] = True
                            ious_union[:, gt_j] = -1

                        # D2 subj obj
                        iou_best, gt_j = ious_subj_obj[pr_i, :].max(dim=0)
                        if iou_best > 0.3:
                            pr_matched["D2_subj_obj"][pr_i] = True
                            ious_subj_obj[:, gt_j] = -1
                    else:
                        raise ValueError(f"Invalid mode: {self.mode}")

                self.unrel_dicts[spo]["scores"].append(pr_spo.scores.values)
                for mode, tps in pr_matched.items():
                    self.unrel_dicts[spo]["true_positives"][mode].append(tps)

    def compute(self):
        from xib.datasets.unrel.metadata import UNUSUAL_RELATIONS

        # Initialize all APs to 0
        ap_dict = {spo: {mode: 0 for mode in self.modes} for spo in UNUSUAL_RELATIONS}

        for spo, spo_dict in self.unrel_dicts.items():
            if len(spo_dict["scores"]) == 0:
                if spo_dict["total_relevant"] == 0:
                    for mode in self.modes:
                        ap_dict[spo][mode] = np.nan
                continue
            scores = np.concatenate(spo_dict["scores"])
            total_relevant = spo_dict["total_relevant"]

            for mode in self.modes:
                true_positives = np.concatenate(spo_dict["true_positives"][mode])

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

                ap_dict[spo][mode] = ap

        aps = {
            mode: np.nanmean([ap[mode] for ap in ap_dict.values()])
            for mode in self.modes
        }

        return aps

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
