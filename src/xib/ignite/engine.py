from __future__ import annotations

from typing import Callable, Any

import torch
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch

from omegaconf import OmegaConf
from ignite.engine import Engine, Events


class CustomEngine(Engine):
    model: torch.nn.Module
    conf: OmegaConf

    def __init__(self, process_function, conf):
        super(CustomEngine, self).__init__(process_function)
        self.conf = conf

        if conf.session.device.startswith("cuda") and torch.cuda.is_available():
            self.add_event_handler(Events.EPOCH_STARTED, CustomEngine._reset_gpu_stats)
            self.add_event_handler(
                Events.EPOCH_COMPLETED, CustomEngine._compute_gpu_stats
            )

    def run(self, data, max_epochs=None, epoch_length=None, seed=None):
        old_level = self.logger.level
        # Disable messages "INFO: Engine run starting with max_epochs=1"
        if max_epochs is None or max_epochs == 1:
            self.logger.setLevel("WARNING")

        super(CustomEngine, self).run(data, max_epochs, epoch_length, seed)

        self.logger.setLevel(old_level)

    @staticmethod
    def _reset_gpu_stats(*_):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    @staticmethod
    def _compute_gpu_stats(engine: CustomEngine):
        bytes = max(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        )
        engine.state.metrics["gpu_mb"] = bytes // 2 ** 20

    @staticmethod
    def setup_training(engine: CustomEngine):
        """Ensure the model is in training mode and that grad computation is enabled"""
        engine.model.train()
        torch.set_grad_enabled(True)
        for p in engine.model.parameters():
            p.requires_grad_(True)

    @staticmethod
    def setup_validation(engine: CustomEngine):
        """Ensure the model is in validation mode and that grad computation is disabled"""
        engine.model.eval()
        torch.set_grad_enabled(False)
        for p in engine.model.parameters():
            p.requires_grad_(False)


class Trainer(CustomEngine):
    optimizer: Optimizer
    criterion: Callable[[Any, Any], torch.Tensor]

    # Make sure state.samples is serialized when `Engine.state_dict()` is called
    _state_dict_all_req_keys = CustomEngine._state_dict_all_req_keys + ("samples",)

    def __init__(self, process_function, conf):
        super(Trainer, self).__init__(process_function, conf)
        self.add_event_handler(Events.STARTED, Trainer._patch_state)
        self.add_event_handler(Events.EPOCH_STARTED, Trainer.setup_training)

    def global_step(self, *_, **__):
        """Return the global step based on how many samples have been processed.

        Raises exception if the state is not present (e.g. engine.run() not called)
        """
        return self.state.samples

    @staticmethod
    def _patch_state(trainer: Trainer):
        """Customize the ignite.engine.State instance that is created after calling ignite.Engine.run
        """
        trainer.state.samples = getattr(trainer.state, "samples", 0)

    def load_state_dict(self, state_dict):
        # Make sure state.samples is deserialized when `Engine.load_state_dict()` is called
        super(Trainer, self).load_state_dict(state_dict)
        setattr(self.state, "samples", state_dict["samples"])


class Validator(CustomEngine):
    def __init__(self, process_function, conf):
        super(Validator, self).__init__(process_function, conf)
        self.add_event_handler(Events.EPOCH_STARTED, Validator.setup_validation)
