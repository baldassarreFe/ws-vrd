from typing import List

import numpy as np
import sklearn.metrics
import torch
from ignite.engine import Engine
from ignite.metrics import Metric


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


class PredicateClassificationMeanAveragePrecisionBatch(object):
    def __call__(self, engine: Engine):
        y_true = engine.state.output["target"]
        y_score = engine.state.output["output"]
        avg_precision = mean_average_precision(y_true, y_score)

        # Return tensor so that ignite.metrics.Average takes batch size into account
        B = len(y_true)
        avg_precision = torch.full((B, 1), fill_value=avg_precision)
        engine.state.output[f"pc/mAP"] = avg_precision


class PredicateClassificationMeanAveragePrecisionEpoch(Metric):
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
