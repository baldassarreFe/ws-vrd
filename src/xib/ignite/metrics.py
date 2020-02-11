from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Iterator, Callable

import torch
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from ignite.engine import Events, Engine
from ignite.metrics import Metric
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch

from ..datasets.hico_det import HicoDet, HicoDetSample


class NoInputMetric(Metric, ABC):
    """A metric that does not require an input to be computed"""

    def __init__(self):
        super(NoInputMetric, self).__init__(output_transform=NoInputMetric._do_nothing)

    @abstractmethod
    def update(self, _: None):
        pass

    @staticmethod
    def _do_nothing(*_, **__):
        pass


class GpuMaxMemoryAllocated(NoInputMetric):
    """Max GPU memory allocated in MB"""
    highest: float

    def reset(self):
        self.highest = 0
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def update(self, _: None):
        self.highest = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))

    def compute(self):
        return self.highest // 2 ** 20

    def attach(self, engine, name='gpu_mb'):
        # Reset highest at the beginning of every epoch
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # Update highest after every iteration
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # Copy metric to engine.state.metrics after every iteration
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class BatchMetric(Metric, ABC):
    """A metric computed independently for every batch."""

    def attach(self, engine, name):
        # Reset at the beginning of every iteration
        if not engine.has_event_handler(self.started, Events.ITERATION_STARTED):
            engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        # Update at the after every iteration
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # Copy metric to engine.state.metrics after every iteration
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


def mean_average_precision(annotations, scores):
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

    with np.errstate(invalid='ignore'):
        average_precisions = sklearn.metrics.average_precision_score(
            y_true=annotations,
            y_score=scores,
            average=None
        )

    return np.nanmean(average_precisions)


class MeanAveragePrecisionBatch(BatchMetric):
    avg_precision: float

    def reset(self):
        self.avg_precision = float('NaN')

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
        return mean_average_precision(
            torch.cat(self.y_true),
            torch.cat(self.y_score),
        )


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
    annotations_of_top_max_s = torch.gather(annotations, index=sorted_labels[:, :max(sizes)], dim=1)

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
    annotations_of_top_max_s = torch.gather(annotations, index=sorted_labels[:, :max(sizes)], dim=1)

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

    def __init__(self, sizes: Tuple[int] = (10, 30, 50), output_transform=lambda x: x, device=None):
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
            engine.state.metrics[f'{name}_{k}'] = v


class RecallAtEpoch(Metric):
    """Recall@x by accumulating outputs over epochs"""
    _y_true: List[torch.Tensor]
    _y_score: List[torch.Tensor]

    def __init__(self, sizes: Tuple[int] = (10, 30, 50), output_transform=lambda x: x, device=None):
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
            engine.state.metrics[f'{name}_{k}'] = v


class PredictImages(object):
    def __init__(
            self,
            grid: Tuple[int, int],
            img_dir: Union[str, Path],
            tag: str,
            logger: SummaryWriter,
            global_step_fn: Callable[[], int]
    ):
        self.grid = grid
        self.img_dir = Path(img_dir).expanduser().resolve()
        self.tag = tag
        self.logger = logger
        self.global_step_fn = global_step_fn

    def __call__(self, engine: Engine):
        samples: List[HicoDetSample] = engine.state.batch[1]
        preds = engine.state.output['output'].sigmoid()
        targets = engine.state.output['target']

        fig, axes = plt.subplots(*self.grid, figsize=(16, 12), dpi=50)
        axes_iter: Iterator[plt.Axes] = axes.flat

        for target, pred, sample, ax in zip(targets, preds, samples, axes_iter):
            image = sample.load_image(img_dir=self.img_dir)

            ax.imshow(image)
            ax.set_title(f'{sample.filename} {sample.img_size}')

            target_str = HicoDet.predicate_id_to_name(sample.gt_visual_relations.predicate_classes.unique(sorted=True))
            ax.text(
                0.05, 0.95,
                '\n'.join(target_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
            )

            top_5 = torch.argsort(pred, descending=True)[:5]
            prediction_str = [f'{pred[i]:.1%} {HicoDet.predicates[i]}' for i in top_5]
            ax.text(
                0.65, 0.95,
                '\n'.join(prediction_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
            )

            ax.tick_params(which='both', **{k: False for k in ('bottom', 'top', 'left', 'right',
                                                               'labelbottom', 'labeltop', 'labelleft', 'labelright')})
            ax.set_xlim(0, sample.img_size.width)
            ax.set_ylim(sample.img_size.height, 0)

        fig.tight_layout()

        self.logger.add_figure(f'{self.tag}', fig, self.global_step_fn(), close=True)
