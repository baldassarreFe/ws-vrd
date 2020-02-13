from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch

from omegaconf import OmegaConf
from ignite.engine import Engine, Events


class CustomEngine(Engine):
    model: torch.nn.Module
    conf: OmegaConf
    name: str

    def __init__(self, process_function, conf):
        super(CustomEngine, self).__init__(process_function)
        self.conf = conf

        if torch.cuda.is_available():
            self.gpu_stats = GpuMaxMemoryAllocated()
            self.add_event_handler(Events.EPOCH_STARTED, CustomEngine._reset_gpu_stats)
            self.add_event_handler(Events.EPOCH_COMPLETED, CustomEngine._compute_gpu_stats)

    @staticmethod
    def _reset_gpu_stats(engine: CustomEngine):
        engine.gpu_stats.reset()

    @staticmethod
    def _compute_gpu_stats(engine: CustomEngine):
        engine.state.metrics['gpu_mb'] = engine.gpu_stats.compute()


class Trainer(CustomEngine):
    optimizer: Optimizer
    name = 'train'

    # Make sure state.samples is serialized when `Engine.state_dict()` is called
    _state_dict_all_req_keys = Engine._state_dict_all_req_keys + ('samples',)

    def __init__(self, process_function, conf):
        super(Trainer, self).__init__(process_function, conf)
        self.add_event_handler(Events.STARTED, Trainer._patch_state)
        self.add_event_handler(Events.EPOCH_STARTED, Trainer._setup_training)
        self.add_event_handler(Events.ITERATION_COMPLETED, Trainer._increment_samples)

    def global_step(self, *_, **__):
        """Return the global step based on how many samples have been processed.

        Raises exception if the state is not present (e.g. engine.run() not called)
        """
        return self.state.samples

    @staticmethod
    def _patch_state(trainer: Trainer):
        """Customize the ignite.engine.State instance that is created after calling ignite.Engine.run
        """
        trainer.state.samples = getattr(trainer.state, 'samples', 0)

    @staticmethod
    def _setup_training(trainer: Trainer):
        """Ensure the model is in training mode and that grad computation is enabled"""
        trainer.model.train()
        torch.set_grad_enabled(True)
        for p in trainer.model.parameters():
            p.requires_grad_(True)

    @staticmethod
    def _increment_samples(trainer: Trainer):
        graphs: Batch = trainer.state.batch
        trainer.state.samples += graphs.num_graphs


class Validator(CustomEngine):
    name = 'val'

    def __init__(self, process_function, conf):
        super(Validator, self).__init__(process_function, conf)
        self.add_event_handler(Events.EPOCH_STARTED, Validator._setup_validation)

    def run(self, data, max_epochs=None, epoch_length=None, seed=None):
        # Disable messages "INFO: Engine run starting with max_epochs=1"
        old_level = self.logger.level
        self.logger.setLevel('WARNING')
        super(Validator, self).run(data, max_epochs, epoch_length, seed)
        self.logger.setLevel(old_level)

    @staticmethod
    def _setup_validation(validator: Validator):
        """Ensure the model is in validation mode and that grad computation is disabled"""
        validator.model.eval()
        torch.set_grad_enabled(True)
        for p in validator.model.parameters():
            p.requires_grad_(False)


class GpuMaxMemoryAllocated(object):
    """Max GPU memory allocated in MB"""

    def reset(self):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def compute(self):
        bytes = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        return bytes // 2 ** 20
