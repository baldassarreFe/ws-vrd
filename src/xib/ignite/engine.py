from __future__ import annotations


import torch
from ignite.engine import Engine, Events
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer

from .metrics import GpuMaxMemoryAllocated
from ..logging import loguru


class CustomEngine(Engine):
    model: torch.nn.Module
    conf: OmegaConf
    lg_logger: loguru.Logger
    name = 'engine'

    _state_dict_all_req_keys = Engine._state_dict_all_req_keys + ('samples',)

    def __init__(self, process_function, conf):
        super(CustomEngine, self).__init__(process_function)
        self.conf = conf
        self.lg_logger = loguru.logger
        self.add_event_handler(Events.STARTED, CustomEngine._patch_state)
        self.add_event_handler(Events.ITERATION_COMPLETED, CustomEngine._increment_samples)

        GpuMaxMemoryAllocated().attach(self)

    @staticmethod
    def _patch_state(engine: Engine):
        """Customize the ignite.engine.State instance that is created after calling ignite.Engine.run
        """
        # This one will be serialized when `Engine.state_dict()` is called
        engine.state.samples = getattr(engine.state, 'samples', 0)

        # These others will not, they are just used to store metrics
        setattr(engine.state, 'text', {})
        setattr(engine.state, 'images', {})
        setattr(engine.state, 'figures', {})

    @staticmethod
    def _increment_samples(engine: Engine):
        engine.state.samples += engine.state.batch.num_graphs


class Trainer(CustomEngine):
    optimizer: Optimizer
    name = 'train'

    def __init__(self, process_function, conf):
        super(Trainer, self).__init__(process_function, conf)
        self.add_event_handler(Events.EPOCH_STARTED, Trainer._setup_training)

    def global_step(self, *_, **__):
        """Return the global step based on how many samples have been processed.

        Raises exception if the state is not present (e.g. engine.run() not called)
        """
        return self.state.samples

    @staticmethod
    def _setup_training(trainer: Trainer):
        """Ensure the model is in training mode and that grad computation is enabled"""
        trainer.model.train()
        torch.set_grad_enabled(True)


class Validator(CustomEngine):
    name = 'val'

    def __init__(self, process_function, conf):
        super(Validator, self).__init__(process_function, conf)
        self.add_event_handler(Events.EPOCH_STARTED, Validator._setup_validation)

    @staticmethod
    def _setup_validation(trainer: Trainer):
        """Ensure the model is in validation mode and that grad computation is disabled"""
        trainer.model.eval()
        torch.set_grad_enabled(False)

    def run(self, data, max_epochs=None, epoch_length=None, seed=None):
        # Disable messages "INFO: Engine run starting with max_epochs=1"
        old_level = self.logger.level
        self.logger.setLevel('WARNING')
        super(Validator, self).run(data, max_epochs, epoch_length, seed)
        self.logger.setLevel(old_level)
