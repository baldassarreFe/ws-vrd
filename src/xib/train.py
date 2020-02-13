import os
import socket
import textwrap
from operator import itemgetter

from typing import Callable, Tuple, Any, Dict, Sequence, Mapping
from pathlib import Path

import pyaml
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer

import torch_scatter
from torch_geometric.data import Batch

from omegaconf import OmegaConf

from ignite.engine import Engine, Events
from ignite.metrics import Average
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler

from .utils import import_, noop
from .config import parse_args
from .ignite import PredictImages
from .ignite import Trainer, Validator
from .ignite import RecallAtBatch, RecallAtEpoch
from .ignite import MeanAveragePrecisionEpoch, MeanAveragePrecisionBatch
from .ignite import MetricsHandler, OptimizerParamsHandler, EpochHandler
from .logging import logger, add_logfile, add_custom_scalars
from .datasets import DatasetCatalog
from .structures import VisualRelations
from .logging.hyperparameters import add_hparam_summary, add_session_start, add_session_end


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


def training_step(trainer: Trainer, inputs: Batch):
    inputs = inputs.to(trainer.conf.session.device)
    outputs = trainer.model(inputs)

    loss, loss_dict = trainer.criterion(outputs, inputs)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        'output': outputs.predicate_scores.detach().cpu(),
        'target': inputs.target_predicate_bce.detach().cpu(),
        'losses': loss_dict,
    }


def validation_step(validator: Validator, inputs: Batch):
    inputs = inputs.to(validator.conf.session.device)
    outputs = validator.model(inputs)

    # # For each graph, try to explain only the TOP_K_PREDICATES
    # TOP_K_PREDICATES = 5
    #
    # # For each graph and predicate to explain, keep only the TOP_X_PAIRS (subject, object) as explanations
    # TOP_X_PAIRS = 10
    #
    # def zero_grad_(tensor:torch.Tensor):
    #     if tensor.grad is not None:
    #         tensor.grad.detach_()
    #         tensor.grad.zero_()
    #     return tensor
    #
    # # Prepare input graphs
    # inputs = inputs.to(validator.conf.session.device)
    # num_edges_per_graph = tuple(torch_scatter.scatter_add(
    #     src=torch.ones(inputs.num_edges, device=validator.conf.session.device, dtype=torch.long),
    #     index=inputs.batch[inputs.edge_index[0]],
    #     dim_size=inputs.num_graphs
    # ).tolist())
    # edge_offset_per_graph = [0] + torch_scatter.scatter_add(
    #     src=torch.ones(inputs.num_nodes, device=validator.conf.session.device, dtype=torch.long),
    #     index=inputs.batch,
    #     dim_size=inputs.num_graphs
    # ).tolist()[:-1]
    # inputs.apply(torch.Tensor.requires_grad_, 'node_linear_features', 'node_conv_features', 'edge_attr')
    #
    # # Forward pass to get predicate predictions
    # output = validator.model(inputs)
    #
    # visual_relations = [{
    #     'relation_indexes': torch.empty((2, 0), dtype=torch.long, device=validator.conf.session.device),
    #     'predicate_classes': torch.empty((0,), dtype=torch.long, device=validator.conf.session.device),
    #     'predicate_scores': torch.empty((0,), dtype=torch.float, device=validator.conf.session.device),
    #     'relation_scores': torch.empty((0,), dtype=torch.float, device=validator.conf.session.device),
    # } for _ in range(inputs.num_graphs)]
    #
    # # Sort predicate predictions per graph and iterate through each one of the TOP_K_PREDICATES predictions
    # predicate_scores_sorted, predicate_labels_sorted = torch.sort(
    #     output.predicate_scores.sigmoid(),
    #     dim=1,
    #     descending=True
    # )
    # predicate_scores_sorted = predicate_scores_sorted[:, :TOP_K_PREDICATES]
    # predicate_labels_sorted = predicate_labels_sorted[:, :TOP_K_PREDICATES]
    # for predicate_score, predicate_label in zip(
    #         predicate_scores_sorted.unbind(dim=1), predicate_labels_sorted.unbind(dim=1)):
    #     inputs.apply(zero_grad_, 'node_linear_features', 'node_conv_features', 'edge_attr')
    #
    #     # Propagate gradient of prediction to outputs, use L1 norm of the gradient as relevance
    #     predicate_score.backward(torch.ones_like(predicate_score), retain_graph=True)
    #     relevance_nodes = (
    #             inputs.node_linear_features.grad.abs().sum(dim=1) +
    #             inputs.node_conv_features.grad.abs().flatten(start_dim=1).sum(dim=1)
    #     )
    #     relevance_edges = inputs.edge_attr.grad.abs().sum(dim=1)
    #
    #     # Each (subject, object) pair receives a relevance score that is proportional
    #     # to the relevance of the subject, the object, and the edge that connects them
    #     subject_object_scores = (
    #         relevance_nodes[inputs.edge_index[0]] *
    #         relevance_edges *
    #         relevance_nodes[inputs.edge_index[1]]
    #     )
    #
    #     # Now we would like to retain the TOP_X_PAIRS edges, i.e. (subject, object) pairs, whose relevance
    #     # score w.r.t. to the current predicate_label is highest. Ideally we would do this in a batched fashion,
    #     # but there is no scatter_sort, so we have to iterate over the edges of each graph and their scores.
    #     subject_object_scores = torch.split_with_sizes(subject_object_scores, split_sizes=num_edges_per_graph)
    #     edge_indexes = torch.split_with_sizes(inputs.edge_index, split_sizes=num_edges_per_graph, dim=1)
    #
    #     for vr, pred_score, pred_label, edge_scores, edge_index, edge_offset in zip(
    #             visual_relations, predicate_score.detach(), predicate_label,
    #             subject_object_scores, edge_indexes, edge_offset_per_graph
    #     ):
    #         edge_scores_sorted_index = torch.argsort(edge_scores, descending=True)[:TOP_X_PAIRS]
    #         so_scores_sorted = edge_scores[edge_scores_sorted_index]
    #
    #         vr['relation_scores'] = torch.cat((
    #             vr['relation_scores'],
    #             pred_score * so_scores_sorted
    #         ), dim=0)
    #         vr['predicate_scores'] = torch.cat((
    #             vr['predicate_scores'],
    #             pred_score.repeat(len(so_scores_sorted))
    #         ), dim=0)
    #         vr['predicate_classes'] = torch.cat((
    #             vr['predicate_classes'],
    #             pred_label.repeat(len(so_scores_sorted))
    #         ), dim=0)
    #         vr['relation_indexes'] = torch.cat((
    #             vr['relation_indexes'],
    #             edge_index[:, edge_scores_sorted_index] - edge_offset
    #         ), dim=1)
    #
    # # visual_relations = [VisualRelations(**vr) for vr in visual_relations]
    # visual_relations = [VisualRelations(
    #     **vr,
    #     object_vocabulary=DatasetCatalog.get('hico_det')['metadata']['objects'],
    #     predicate_vocabulary=DatasetCatalog.get('hico_det')['metadata']['predicates']
    # ) for vr in visual_relations]
    #
    # for vr in visual_relations:
    #     print(*vr.relation_str(), sep='\n', end='\n\n')

    _, loss_dict = validator.criterion(outputs, inputs)

    return {
        'output': outputs.predicate_scores.detach().cpu(),
        'target': inputs.target_predicate_bce.detach().cpu(),
        'losses': loss_dict,
    }


class Criterion(object):
    def __init__(self, conf: OmegaConf):
        self.bce_weight = conf.bce.weight
        self.rank_weight = conf.rank.weight

    def __call__(self, results: Batch, targets: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        loss_total = torch.tensor(0., device=results.predicate_scores.device)

        if self.bce_weight > 0:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                results.predicate_scores,
                targets.target_predicate_bce,
                reduction='mean'
            )
            loss_dict['bce'] = bce
            loss_total += self.bce_weight * bce

        if self.rank_weight > 0:
            rank = torch.nn.functional.multilabel_margin_loss(
                results.predicate_scores,
                targets.target_predicate_rank,
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
        dataset = ds['class'](**conf.dataset.trainval)
        train_split = int(conf.dataset.split * len(dataset))
        val_split = len(dataset) - train_split
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])

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
        # Hacky way to get around torch_geometric not supporting string attributes
        graphs = Batch.from_data_list([graph for graph, _ in batch], follow_batch=['target_object_boxes'])
        # TODO check this doesn't cause problems in multiprocessing, maybe do a deepcopy?
        graphs.filenames = np.array([filename for _, filename in batch])
        return graphs

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
        # TODO restore
        # batch_size=conf.dataloader.batch_size,
        batch_size=4,
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


def log_metrics(engine: Engine, tag: str):
    yaml = pyaml.dump(engine.state.metrics, safe=True, sort_dicts=True, force_embed=True)
    logger.info(f'{tag}\n{yaml}')


@logger.catch(reraise=True)
def main():
    # region Setup
    conf = parse_args()
    tb_logger, tb_img_logger = setup_logging(conf)
    logger.info('Parsed configuration:\n' +
                pyaml.dump(OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True))

    trainer = Trainer(training_step, conf)
    validator = Validator(validation_step, conf)

    trainer.model = validator.model = build_model(conf.model).to(conf.session.device)
    trainer.criterion = validator.criterion = Criterion(conf.losses)
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

    train_dataset, val_dataset, load_datasets_eager, dataset_metadata = build_datasets(conf)
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
        'train'
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredictImages(grid=(2, 3), img_dir=conf.dataset.image_dir, tag='train',
                      logger=tb_img_logger.writer, predicate_vocabulary=dataset_metadata['predicates'],
                      global_step_fn=trainer.global_step)
    )
    tb_logger.attach(
        trainer,
        EpochHandler(trainer, tag='z', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(val_dataloader))
    # endregion

    # region Validation callbacks
    # PredictImages(output_transform=itemgetter('target', 'output')).attach(validator)
    if conf.losses['bce']['weight'] > 0:
        Average(output_transform=lambda o: o['losses']['loss/bce']).attach(validator, 'loss/bce')
    if conf.losses['rank']['weight'] > 0:
        Average(output_transform=lambda o: o['losses']['loss/rank']).attach(validator, 'loss/rank')
    Average(output_transform=lambda o: o['losses']['loss/total']).attach(validator, 'loss/total')
    MeanAveragePrecisionEpoch(output_transform=itemgetter('target', 'output')).attach(validator, 'mAP')
    RecallAtEpoch(output_transform=itemgetter('target', 'output'), sizes=(5, 10)).attach(validator, 'recall_at')

    ProgressBar(persist=True, desc='Val').attach(validator, metric_names='all', output_transform=itemgetter('losses'))

    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda val_engine: scheduler.step(val_engine.state.metrics['loss/total'])
    )
    tb_logger.attach(
        validator,
        MetricsHandler('val', global_step_transform=trainer.global_step),
        Events.EPOCH_COMPLETED
    )
    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        'val'
    )
    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredictImages(grid=(2, 3), img_dir=conf.dataset.image_dir, tag='val',
                      logger=tb_img_logger.writer, predicate_vocabulary=dataset_metadata['predicates'],
                      global_step_fn=trainer.global_step)
    )
    validator.add_event_handler(Events.COMPLETED, EarlyStopping(
        patience=conf.session.early_stopping.patience,
        score_function=lambda val_engine: - val_engine.state.metrics['loss/total'],
        trainer=trainer
    ))
    validator.add_event_handler(
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

    # Log effective configuration before starting
    global_step = trainer.global_step() if 'resume' in conf else 0
    yaml = pyaml.dump(OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True)
    tb_logger.writer.add_text('Configuration', textwrap.indent(yaml, '    '), global_step)
    add_session_start(tb_logger.writer, conf.hparams)
    tb_logger.writer.flush()
    p = Path(conf.checkpoint.folder).expanduser() / conf.fullname / f'conf.{global_step}.yaml'
    with p.open(mode='w') as f:
        f.write(f'# Effective configuration at global step {global_step}\n')
        f.write(yaml)

    # If requested, load datasets eagerly
    load_datasets_eager()

    # Finally run
    trainer.run(train_dataloader, max_epochs=conf.session.max_epochs, seed=conf.session.seed)

    add_session_end(tb_logger.writer, 'SUCCESS')
    tb_logger.close()
    tb_img_logger.close()


if __name__ == '__main__':
    main()
