import textwrap
from operator import itemgetter

from typing import Callable, Tuple, Any, Dict
from pathlib import Path

import pyaml
import torch
import torch.utils.data
from torch.optim.optimizer import Optimizer

from torch_geometric.data import Batch, DataLoader

from omegaconf import OmegaConf

from ignite.engine import Events
from ignite.metrics import Average
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler

from .utils import import_
from .config import parse_args
from .ignite import Trainer, Validator
from .ignite import RecallAtBatch, RecallAtEpoch
from .ignite import MeanAveragePrecisionEpoch, MeanAveragePrecisionBatch
from .ignite import MetricsHandler, OptimizerParamsHandler, EpochHandler
from .logging import logger, add_logfile


def setup_logging(conf: OmegaConf) -> TensorboardLogger:
    folder = Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname
    add_logfile(folder / 'logs')
    tb_logger = TensorboardLogger(folder)
    return tb_logger


def training_step(trainer: Trainer, graphs: Batch):
    graphs = graphs.to(trainer.conf.session.device)
    output = trainer.model(graphs)

    loss, loss_dict = criterion(trainer.conf.losses, output, graphs)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        'output': output.predicate_scores.detach().cpu(),
        'target': graphs.target_bce.detach().cpu(),
        'losses': loss_dict,
    }


def validation_step(validator: Validator, graphs: Batch):
    graphs = graphs.to(validator.conf.session.device)
    output = validator.model(graphs)

    _, loss_dict = criterion(validator.conf.losses, output, graphs)

    return {
        'output': output.predicate_scores.cpu(),
        'target': graphs.target_bce.cpu(),
        'losses': loss_dict,
    }


def criterion(conf: OmegaConf, results: Batch, targets: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_dict = {}
    loss_total = torch.tensor(0., device=results.predicate_scores.device)

    if conf['bce']['weight'] > 0:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            results.predicate_scores,
            targets.target_bce,
            reduction='mean'
        )
        loss_dict['bce'] = bce
        loss_total += conf['bce']['weight'] * bce

    if conf['rank']['weight'] > 0:
        rank = torch.nn.functional.multilabel_margin_loss(
            results.predicate_scores,
            targets.target_rank,
            reduction='mean'
        )
        loss_dict['rank'] = rank
        loss_total += conf['rank']['weight'] * rank

    loss_dict['total'] = loss_total
    loss_dict = {
        f'loss/{k}': v.detach().cpu().item()
        for k, v in loss_dict.items()
    }

    return loss_total, loss_dict


def build_optimizer_and_scheduler(conf: OmegaConf, model: torch.nn.Module) -> Tuple[Optimizer, Any]:
    conf = OmegaConf.to_container(conf, resolve=True)

    optimizer_fn = import_(conf.pop('name'))
    optimizer = optimizer_fn(model.parameters(), **conf)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    return optimizer, scheduler


def build_dataloaders(conf) -> Tuple[DataLoader, DataLoader, Callable[[], None]]:
    dataset_class = import_(conf.dataset.name)

    if 'trainval' in conf.dataset and 'split' in conf.dataset:
        dataset = dataset_class(**conf.dataset.trainval)
        train_split = int(conf.dataset.split * len(dataset))
        val_split = len(dataset) - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])

        eager_op = dataset.load_eager
    elif 'train' in conf.dataset and 'val' in conf.dataset:
        train_dataset = dataset_class(**conf.dataset.train)
        val_dataset = dataset_class(**conf.dataset.val)

        def eager_op():
            train_dataset.load_eager()
            val_dataset.load_eager()
    else:
        raise ValueError(f'Invalid data specification:\n{conf.dataset.pretty()}')

    logger.info(f'Train/Val split: {len(train_dataset)}/{len(val_dataset)} '
                f'{len(train_dataset) / (len(train_dataset) + len(val_dataset)):.1%}/'
                f'{len(val_dataset) / (len(train_dataset) + len(val_dataset)):.1%}')

    if not conf.dataset.eager:
        def eager_op():
            pass

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True
    )

    return train_dataloader, val_dataloader, eager_op


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
    trainer.optimizer, scheduler = build_optimizer_and_scheduler(conf.optimizer, trainer.model)

    if 'resume' in conf:
        checkpoint = Path(conf.resume.checkpoint).expanduser().resolve()
        logger.debug(f'Resuming checkpoint from {checkpoint}')
        Checkpoint.load_objects({
            'model': trainer.model,
            'optimizer': trainer.optimizer,
            'scheduler': scheduler,
            'trainer': trainer,
        }, checkpoint=torch.load(checkpoint, map_location=conf.session.device))
        logger.info(f'Resumed from {checkpoint}, epoch {trainer.state.epoch}, samples {trainer.global_step()}')

    train_dataloader, val_dataloader, load_datasets_eager = build_dataloaders(conf)
    # endregion

    # region Training callbacks
    tb_logger.attach(
        trainer,
        OptimizerParamsHandler(trainer.optimizer, param_name='lr', tag='z', global_step_transform=trainer.global_step),
        Events.EPOCH_STARTED
    )

    MeanAveragePrecisionBatch(output_transform=itemgetter('target', 'output')).attach(trainer, 'mAP')
    RecallAtBatch(output_transform=itemgetter('target', 'output')).attach(trainer, 'recall_at')

    ProgressBar(persist=True, desc='Train').attach(trainer, metric_names='all', output_transform=itemgetter('losses'))

    tb_logger.attach(
        trainer,
        OutputHandler('train', output_transform=itemgetter('losses'), global_step_transform=trainer.global_step),
        Events.ITERATION_COMPLETED
    )
    tb_logger.attach(
        trainer,
        MetricsHandler('train', global_step_transform=trainer.global_step),
        Events.ITERATION_COMPLETED
    )

    tb_logger.attach(
        trainer,
        EpochHandler(trainer, tag='z', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=conf.checkpoint.every),
        Checkpoint(
            {
                'model': trainer.model,
                'optimizer': trainer.optimizer,
                'scheduler': scheduler,
                'trainer': trainer,
            },
            DiskSaver(Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname),
            n_saved=None,
            global_step_transform=trainer.global_step
        )
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(val_dataloader))
    # endregion

    # region Validation callbacks
    if conf.losses['bce']['weight'] > 0:
        Average(output_transform=lambda o: o['losses']['loss/bce']).attach(validator, 'loss/bce')
    if conf.losses['rank']['weight'] > 0:
        Average(output_transform=lambda o: o['losses']['loss/rank']).attach(validator, 'loss/rank')
    Average(output_transform=lambda o: o['losses']['loss/total']).attach(validator, 'loss/total')
    MeanAveragePrecisionEpoch(output_transform=itemgetter('target', 'output')).attach(validator, 'mAP')
    RecallAtEpoch(output_transform=itemgetter('target', 'output')).attach(validator, 'recall_at')

    ProgressBar(persist=True, desc='Val').attach(validator, metric_names='all', output_transform=itemgetter('losses'))

    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda val_engine: scheduler.step(val_engine.state.metrics['loss/total'])
    )
    validator.add_event_handler(Events.COMPLETED, EarlyStopping(
        patience=conf.session.early_stopping.patience,
        score_function=lambda val_engine: - val_engine.state.metrics['loss/total'],
        trainer=trainer
    ))
    tb_logger.attach(
        validator,
        MetricsHandler('val', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )
    # endregion

    # Log configuration before starting
    yaml = pyaml.dump(OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True)
    logger.info('\n' + yaml)
    global_step = trainer.global_step() if 'resume' in conf else 0
    tb_logger.writer.add_text('Configuration', textwrap.indent(yaml, '    '), global_step)
    tb_logger.writer.flush()  # Prevent tensorboard complaining "Unable to get first event timestamp"
    p = Path(conf.checkpoint.folder).expanduser() / conf.fullname / f'conf.{global_step}.yaml'
    with p.open(mode='w') as f:
        f.write(yaml)

    # Finally run
    load_datasets_eager()
    trainer.run(train_dataloader, max_epochs=conf.session.max_epochs)
    tb_logger.close()


if __name__ == '__main__':
    main()
