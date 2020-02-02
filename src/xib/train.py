import sys
import textwrap

from typing import Callable, Tuple, Any
from pathlib import Path

import pyaml

import torch
import torch.utils.data
import torch_geometric

from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.metrics import Average

from omegaconf import OmegaConf
from torch_geometric.data import Batch

from .utils import import_, random_name
from .logging import logger, add_logfile, TensorboardLogHandler
from .ignite.engine import Trainer, Validator
from .ignite.metrics import OutputMetricBatch, AveragePrecisionEpoch, AveragePrecisionBatch


def parse_args():
    OmegaConf.register_resolver('randomname', random_name)

    conf = OmegaConf.create()
    for s in sys.argv[1:]:
        if s.endswith('.yaml'):
            conf.merge_with(OmegaConf.load(s))
        else:
            conf.merge_with_dotlist([s])

    # Make sure everything is resolved
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    return conf


def setup_logging(conf: OmegaConf) -> TensorboardLogger:
    folder = Path(conf.checkpoint.folder).expanduser().joinpath(conf.fullname).resolve()
    add_logfile(folder / 'logs')
    tb_logger = TensorboardLogger(folder)
    return tb_logger


def training_step(trainer: Trainer, graphs: Batch):
    graphs = graphs.to(trainer.conf.session.device)
    output = trainer.model(graphs)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, graphs.target)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        'output': output.detach().cpu(),
        'target': graphs.target.detach().cpu(),
        'loss': loss.item(),
    }


def validation_step(validator: Validator, graphs: Batch):
    graphs = graphs.to(validator.conf.session.device)
    output = validator.model(graphs)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, graphs.target)

    return {
        'output': output.cpu(),
        'target': graphs.target.cpu(),
        'loss': loss.item(),
    }


def build_optimizer_and_scheduler(conf: OmegaConf, model: torch.nn.Module) -> Tuple[torch.optim.Optimizer, Any]:
    conf = OmegaConf.to_container(conf, resolve=True)

    optimizer_fn = import_(conf.pop('name'))
    optimizer = optimizer_fn(model.parameters(), **conf)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    return optimizer, scheduler


def build_dataloaders(conf):
    dataset_class = import_(conf.dataset.name)

    if 'trainval' in conf.dataset and 'split' in conf.dataset:
        dataset = dataset_class(conf.dataset.trainval.folder)
        if conf.dataset.eager:
            dataset.load_eager()
        train_split = int(conf.dataset.split * len(dataset))
        val_split = len(dataset) - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])
    elif 'train' in conf.dataset and 'val' in conf.dataset:
        train_dataset = dataset_class(conf.dataset.train.folder)
        val_dataset = dataset_class(conf.dataset.val.folder)
        if conf.dataset.eager:
            train_dataset.load_eager()
            val_dataset.load_eager()
    else:
        raise ValueError(f'Invalid data specification:\n{conf.dataset.pretty()}')

    logger.info(f'Train/Val split: {len(train_dataset)}/{len(val_dataset)} '
                f'{len(train_dataset) / (len(train_dataset) + len(val_dataset)):.1%}/'
                f'{len(val_dataset) / (len(train_dataset) + len(val_dataset)):.1%}')

    train_dataloader = torch_geometric.data.DataLoader(
        train_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True
    )

    val_dataloader = torch_geometric.data.DataLoader(
        val_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True
    )

    return train_dataloader, val_dataloader


def build_val_dataloader(conf):
    dataset_class = import_(conf.dataset.name)
    dataset = dataset_class(**conf.dataset.train)

    dataloader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory=False,
        drop_last=True
    )

    return dataloader


def build_model(conf: OmegaConf) -> torch.nn.Module:
    model_fn: Callable[[OmegaConf], torch.nn.Module] = import_(conf.name)
    model = model_fn(conf)
    return model


@logger.catch
def main():
    # region Setup
    conf = parse_args()
    tb_logger = setup_logging(conf)

    trainer = Trainer(training_step, conf)
    validator = Validator(validation_step, conf)

    trainer.model = validator.model = build_model(conf.model).to(conf.session.device)
    trainer.optimizer, trainer.scheduler = build_optimizer_and_scheduler(conf.optimizer, trainer.model)

    if 'resume' in conf:
        checkpoint = Path(conf.resume.checkpoint).expanduser().resolve()
        logger.info(f'Resuming checkpoint from {checkpoint}')
        Checkpoint.load_objects({
            'model': trainer.model,
            'optimizer': trainer.optimizer,
            'scheduler': trainer.scheduler,
            'trainer': trainer,
        }, checkpoint=torch.load(checkpoint, map_location=conf.session.device))

    train_dataloader, val_dataloader = build_dataloaders(conf)
    # endregion

    # region Training callbacks
    OutputMetricBatch(output_transform=lambda o: o['loss']).attach(trainer, 'loss')
    AveragePrecisionBatch(output_transform=lambda o: (o['target'], o['output'])).attach(trainer, 'avg_prec')

    # Attach loggers after all metrics
    tb_logger.attach(trainer, TensorboardLogHandler(trainer.global_step), Events.ITERATION_COMPLETED)
    ProgressBar(persist=True, desc='Train').attach(trainer, metric_names='all')

    trainer.add_event_handler(  # Or validator.add_event_handler?
        Events.EPOCH_COMPLETED(every=conf.checkpoint.every),
        Checkpoint(
            {
                'model': trainer.model,
                'optimizer': trainer.optimizer,
                'scheduler': trainer.scheduler,
                'trainer': trainer,
            },
            DiskSaver(Path(conf.checkpoint.folder) / conf.fullname),
            n_saved=None,
            global_step_transform=trainer.global_step
        )
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(val_dataloader))
    # endregion

    # region Validation callbacks
    Average(output_transform=lambda o: o['loss']).attach(validator, 'loss')
    AveragePrecisionEpoch(output_transform=lambda o: (o['target'], o['output'])).attach(validator, 'avg_prec')
    validator.add_event_handler(Events.COMPLETED, EarlyStopping(
        patience=10,
        score_function=lambda val_engine: - val_engine.state.metrics['loss'],
        trainer=trainer
    ))

    # Attach loggers after all metrics
    tb_logger.attach(validator, TensorboardLogHandler(trainer.global_step), Events.EPOCH_COMPLETED)
    ProgressBar(persist=True, desc='Val').attach(validator, metric_names='all')
    # endregion

    # Log configuration before starting
    yaml = pyaml.dump(OmegaConf.to_container(trainer.conf), safe=True, sort_dicts=False, force_embed=True)
    logger.info('\n' + yaml)
    tb_logger.writer.add_text('Configuration', textwrap.indent(yaml, '    '), trainer.global_step() or 0)
    with open(Path(trainer.conf.checkpoint.folder).expanduser() / trainer.conf.fullname / 'conf', mode='w') as f:
        f.write(yaml)

    # Finally run
    trainer.run(train_dataloader, max_epochs=conf.session.max_epochs)


if __name__ == '__main__':
    main()
