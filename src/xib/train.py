import os
import random
import socket
import textwrap
from operator import itemgetter

from typing import Callable, Tuple, Any, Dict, Mapping
from pathlib import Path

import pyaml
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer

from torch_geometric.data import Batch

from omegaconf import OmegaConf

from ignite.engine import Engine, Events
from ignite.metrics import Average
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler

from .utils import import_, noop
from .config import parse_args
from .ignite import PredictPredicatesImg, PredictRelationsImg
from .ignite import Trainer, Validator
from .ignite import RecallAtBatch, RecallAtEpoch
from .ignite import MeanAveragePrecisionEpoch, MeanAveragePrecisionBatch
from .ignite import MetricsHandler, OptimizerParamsHandler, EpochHandler
from .logging import logger, add_logfile, add_custom_scalars
from .datasets import DatasetCatalog
from .models.visual_relations_explainer import VisualRelationExplainer
from .logging.hyperparameters import add_hparam_summary, add_session_start, add_session_end


def setup_seeds(seed):
    # Ignite will set up its own seeds, these are for operations that happen
    # before engine.run(), e.g. building the model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(conf: OmegaConf) -> [TensorboardLogger, TensorboardLogger]:
    folder = Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname
    folder.mkdir(parents=True, exist_ok=True)

    add_logfile(folder / 'logs')

    logger.info(f'PID: {os.getpid()}')
    logger.info(f'Host: {socket.gethostname()}')
    logger.info(f'SLURM_JOB_ID: {os.getenv("SLURM_JOB_ID")}')

    # Prepare two loggers, the second one specifically for images, so the first one stays slim
    tb_logger = TensorboardLogger(logdir=folder)
    tb_img_logger = TensorboardLogger(logdir=folder, filename_suffix='.images')

    add_custom_scalars(tb_logger.writer)
    add_hparam_summary(tb_logger.writer, conf.hparams)

    return tb_logger, tb_img_logger


def training_step(trainer: Trainer, batch: Batch):
    inputs, targets, _ = batch
    inputs = inputs.to(trainer.conf.session.device)
    targets = targets.to(trainer.conf.session.device)

    outputs = trainer.model(inputs)
    loss, loss_dict = trainer.criterion(outputs, targets)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        'output': outputs.predicate_scores.detach().cpu(),
        'target': targets.predicate_bce.detach().cpu(),
        'losses': loss_dict,
    }


def predicate_prediction_validation_step(validator: Validator, batch: Batch):
    inputs, targets, filenames = batch
    inputs = inputs.to(validator.conf.session.device)
    targets = targets.to(validator.conf.session.device)

    outputs = validator.model(inputs)

    _, loss_dict = validator.criterion(outputs, targets)

    return {
        'output': outputs.predicate_scores.detach().cpu(),
        'target': targets.predicate_bce.detach().cpu(),
        'losses': loss_dict,
    }


def relationship_prediction_validation_step(validator: Validator, batch: Batch):
    inputs, targets, filenames = batch
    inputs = inputs.to(validator.conf.session.device)

    relations = validator.model(inputs)

    return {'relations': relations.to('cpu')}


class PredicatePredictionCriterion(object):
    def __init__(self, conf: OmegaConf):
        self.bce_weight = conf.bce.weight
        self.rank_weight = conf.rank.weight

    def __call__(self, results: Batch, targets: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        loss_total = torch.tensor(0., device=results.predicate_scores.device)

        if self.bce_weight > 0:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                results.predicate_scores,
                targets.predicate_bce,
                reduction='mean'
            )
            loss_dict['bce'] = bce
            loss_total += self.bce_weight * bce

        if self.rank_weight > 0:
            rank = torch.nn.functional.multilabel_margin_loss(
                results.predicate_scores,
                targets.predicate_rank,
                reduction='mean'
            )
            loss_dict['rank'] = rank
            loss_total += self.rank_weight * rank

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


def build_datasets(conf) -> Tuple[Dataset, Dataset, Callable[[], None], Mapping[str, Any]]:
    ds = DatasetCatalog.get(conf.dataset.name)

    if 'trainval' in conf.dataset and 'split' in conf.dataset:
        rg = np.random.default_rng(conf.session.seed)
        dataset = ds['class'](**conf.dataset.trainval)
        indexes = rg.permutation(len(dataset))
        train_split = indexes[:int(conf.dataset.split * len(dataset))]
        val_split = indexes[int(conf.dataset.split * len(dataset)):]
        train_dataset = torch.utils.data.Subset(dataset, train_split)
        val_dataset = torch.utils.data.Subset(dataset, val_split)

        def eager_op():
            dataset.load_eager()
    elif 'train' in conf.dataset and 'val' in conf.dataset:
        train_dataset = ds['class'](**conf.dataset.train)
        val_dataset = ds['class'](**conf.dataset.val)

        def eager_op():
            train_dataset.load_eager()
            val_dataset.load_eager()
    else:
        raise ValueError(f'Invalid data specification:\n{conf.dataset.pretty()}')

    logger.info(f'Data split: {len(train_dataset)} train, {len(val_dataset)} val ('
                f'{100 * len(train_dataset) / (len(train_dataset) + len(val_dataset)):.1f}/'
                f'{100 * len(val_dataset) / (len(train_dataset) + len(val_dataset)):.1f}%)')

    if not conf.dataset.eager:
        eager_op = noop

    return train_dataset, val_dataset, eager_op, ds['metadata']


def build_dataloaders(conf, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    def collate_fn(batch):
        inputs, targets, filenames = zip(*batch)

        inputs = Batch.from_data_list(inputs)
        targets = Batch.from_data_list(targets)

        return inputs, targets, filenames

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.dataloader.batch_size,
        shuffle=False,
        num_workers=conf.dataloader.num_workers,
        pin_memory='cuda' in conf.session.device,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def build_model(conf: OmegaConf) -> torch.nn.Module:
    model_fn: Callable[[OmegaConf], torch.nn.Module] = import_(conf.name)
    model = model_fn(conf)
    return model


def log_metrics(engine: Engine, tag: str, global_step_fn: Callable[[], int]):
    global_step = global_step_fn()
    yaml = pyaml.dump({f'{tag}/{k}': v for k, v in engine.state.metrics.items()},
                      safe=True, sort_dicts=True, force_embed=True)
    logger.info(f'{tag} {global_step}:\n{yaml}')


def log_effective_config(conf, trainer, tb_logger):
    global_step = trainer.global_step() if 'resume' in conf else 0
    yaml = pyaml.dump(OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True)
    tb_logger.writer.add_text('Configuration', textwrap.indent(yaml, '    '), global_step)
    add_session_start(tb_logger.writer, conf.hparams)
    tb_logger.writer.flush()
    p = Path(conf.checkpoint.folder).expanduser() / conf.fullname / f'conf.{global_step}.yaml'
    with p.open(mode='w') as f:
        f.write(f'# Effective configuration at global step {global_step}\n')
        f.write(yaml)


@logger.catch(reraise=True)
def main():
    # region Setup
    conf = parse_args()
    setup_seeds(conf.session.seed)
    tb_logger, tb_img_logger = setup_logging(conf)
    logger.info('Parsed configuration:\n' +
                pyaml.dump(OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True))

    trainer = Trainer(training_step, conf)
    predicate_predication_validator = Validator(predicate_prediction_validation_step, conf)

    model = build_model(conf.model).to(conf.session.device)
    trainer.model = predicate_predication_validator.model = model
    trainer.criterion = predicate_predication_validator.criterion = PredicatePredictionCriterion(conf.losses)
    trainer.optimizer, scheduler = build_optimizer_and_scheduler(conf.optimizer, trainer.model)

    relationship_prediction_validator = Validator(relationship_prediction_validation_step, conf)
    relationship_prediction_validator.model = VisualRelationExplainer(model, **conf.visual_relations)

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

    train_dataset, val_dataset, maybe_load_datasets, dataset_metadata = build_datasets(conf)
    train_dataloader, val_dataloader = build_dataloaders(conf, train_dataset, val_dataset)
    # endregion

    # region Training callbacks
    tb_logger.attach(
        trainer,
        OptimizerParamsHandler(trainer.optimizer, param_name='lr', tag='z', global_step_transform=trainer.global_step),
        Events.EPOCH_STARTED
    )

    MeanAveragePrecisionBatch(output_transform=itemgetter('target', 'output')).attach(trainer, 'mAP')
    RecallAtBatch(output_transform=itemgetter('target', 'output'), sizes=(5, 10)).attach(trainer, 'recall_at')

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

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        'train',
        trainer.global_step
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredictPredicatesImg(grid=(2, 3), img_dir=conf.dataset.image_dir, tag='train',
                             logger=tb_img_logger.writer, predicate_vocabulary=dataset_metadata['predicates'],
                             global_step_fn=trainer.global_step)
    )
    tb_logger.attach(
        trainer,
        EpochHandler(trainer, tag='z', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda _: predicate_predication_validator.run(val_dataloader)
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=3),
        lambda _: relationship_prediction_validator.run(val_dataloader)
    )
    # endregion

    # region Predicate prediction validation callbacks
    if conf.losses['bce']['weight'] > 0:
        Average(
            output_transform=lambda o: o['losses']['loss/bce'],
            device=conf.session.device
        ).attach(predicate_predication_validator, 'loss/bce')
    if conf.losses['rank']['weight'] > 0:
        Average(
            output_transform=lambda o: o['losses']['loss/rank'],
            device=conf.session.device
        ).attach(predicate_predication_validator, 'loss/rank')
    Average(
        output_transform=lambda o: o['losses']['loss/total'],
        device=conf.session.device
    ).attach(predicate_predication_validator, 'loss/total')

    MeanAveragePrecisionEpoch(itemgetter('target', 'output')).attach(predicate_predication_validator, 'mAP')
    RecallAtEpoch((5, 10), itemgetter('target', 'output')).attach(predicate_predication_validator, 'recall_at')

    ProgressBar(persist=True, desc='Pred val').attach(
        predicate_predication_validator, metric_names='all', output_transform=itemgetter('losses'))

    predicate_predication_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda val_engine: scheduler.step(val_engine.state.metrics['loss/total'])
    )
    tb_logger.attach(
        predicate_predication_validator,
        MetricsHandler('val', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )
    predicate_predication_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        'val',
        trainer.global_step
    )
    predicate_predication_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredictPredicatesImg(grid=(2, 3), img_dir=conf.dataset.image_dir, tag='val',
                             logger=tb_img_logger.writer, predicate_vocabulary=dataset_metadata['predicates'],
                             global_step_fn=trainer.global_step)
    )
    predicate_predication_validator.add_event_handler(Events.COMPLETED, EarlyStopping(
        patience=conf.session.early_stopping.patience,
        score_function=lambda val_engine: - val_engine.state.metrics['loss/total'],
        trainer=trainer
    ))
    predicate_predication_validator.add_event_handler(
        Events.COMPLETED,
        Checkpoint(
            {'model': trainer.model, 'optimizer': trainer.optimizer, 'scheduler': scheduler, 'trainer': trainer},
            DiskSaver(Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname),
            score_function=lambda val_engine: val_engine.state.metrics['recall_at_5'],
            score_name='recall_at_5',
            n_saved=conf.checkpoint.keep,
            global_step_transform=trainer.global_step
        )
    )
    # endregion

    # region Relationship detection validation callbacks
    # MeanAveragePrecisionEpoch(itemgetter('target', 'output')).attach(relationship_prediction_validator, 'mAP')
    # RecallAtEpoch((5, 10), itemgetter('target', 'output')).attach(relationship_prediction_validator, 'recall_at')

    ProgressBar(persist=True, desc='Relations val').attach(relationship_prediction_validator, metric_names='all')

    tb_logger.attach(
        relationship_prediction_validator,
        MetricsHandler('val_vr', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )
    relationship_prediction_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        'val_vr',
        trainer.global_step
    )
    relationship_prediction_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredictRelationsImg(grid=(2, 3), img_dir=conf.dataset.image_dir, tag='val_vr', logger=tb_img_logger.writer,
                            top_x_relations=conf.visual_relations.top_x_relations,
                            object_vocabulary=dataset_metadata['objects'],
                            predicate_vocabulary=dataset_metadata['predicates'],
                            global_step_fn=trainer.global_step)
    )
    # endregion

    # region Run
    maybe_load_datasets()
    log_effective_config(conf, trainer, tb_logger)
    trainer.run(train_dataloader, max_epochs=conf.session.max_epochs, seed=conf.session.seed)

    add_session_end(tb_logger.writer, 'SUCCESS')
    tb_logger.close()
    tb_img_logger.close()
    # endregion


if __name__ == '__main__':
    main()
