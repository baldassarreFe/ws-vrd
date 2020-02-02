from abc import ABC, abstractmethod
from numbers import Number
from typing import List, Union, Tuple

import torch
import sklearn.metrics
from ignite.engine import Events
from ignite.metrics import Metric


class NoInputMetric(Metric, ABC):
    """A metric that does not require an input to be computed"""

    def __init__(self):
        super(NoInputMetric, self).__init__(output_transform=lambda x: None)

    @abstractmethod
    def update(self, _: None):
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


class OutputMetricBatch(BatchMetric):
    """At every iteration, take a simple value from the output and save it as a metric in the engine state."""
    value: Union[Number, torch.Tensor, None]

    def reset(self):
        self.value = None

    def update(self, output):
        self.value = output

    def compute(self):
        return self.value


class AveragePrecisionBatch(BatchMetric):
    avg_precision: float

    def reset(self):
        self.avg_precision = float('NaN')

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_true, y_score = output
        self.avg_precision = sklearn.metrics.average_precision_score(
            y_true=y_true,
            y_score=y_score.sigmoid(),
            average='micro'
        )

    def compute(self):
        return self.avg_precision


class AveragePrecisionEpoch(Metric):
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
        return sklearn.metrics.average_precision_score(
            y_true=torch.cat(self.y_true),
            y_score=torch.cat(self.y_score).sigmoid(),
            average='micro'
        )
