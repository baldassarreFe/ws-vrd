import json
import os
import random
import socket
import textwrap
import time
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Callable, Tuple, Any, Dict, Mapping, Optional

import numpy as np
import pyaml
import torch
import torch.utils.data
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.metrics import Average
from loguru import logger
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from .config import parse_args
from .datasets import DatasetFolder, VrDataset, register_datasets
from .ignite import HOImAP
from .ignite import MeanAveragePrecisionEpoch, MeanAveragePrecisionBatch
from .ignite import MetricsHandler, OptimizerParamsHandler, EpochHandler
from .ignite import PredicatePredictionLogger
from .ignite import RecallAtBatch, RecallAtEpoch
from .ignite import Trainer, Validator
from .ignite import VisualRelationPredictionLogger, VisualRelationRecallAt
from .logging import setup_logging, add_logfile, add_custom_scalars
from .logging.hyperparameters import (
    add_hparam_summary,
    add_session_start,
    add_session_end,
)
from .models.visual_relations_explainer import VisualRelationExplainer
from .utils import import_


def setup_seeds(seed):
    # Ignite will set up its own seeds, these are for operations that happen
    # before engine.run(), e.g. building the model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_all_loggers(conf: OmegaConf) -> [TensorboardLogger, TensorboardLogger, Path]:
    folder = Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname
    folder.mkdir(parents=True, exist_ok=True)

    # Loguru logger: stderr and logs.txt
    add_logfile(folder / "logs.txt")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Host: {socket.gethostname()}")
    logger.info(f'SLURM_JOB_ID: {os.getenv("SLURM_JOB_ID")}')

    # Tensorboard: two loggers, the second one specifically for images, so the first one stays slim
    tb_logger = TensorboardLogger(logdir=folder)
    tb_img_logger = TensorboardLogger(logdir=folder, filename_suffix=".images")
    add_custom_scalars(tb_logger.writer)
    add_hparam_summary(tb_logger.writer, conf.hparams)

    # Json: only validation metrics
    json_logger = folder / "metrics.json"

    return tb_logger, tb_img_logger, json_logger


def pred_class_training_step(trainer: Trainer, batch: Batch):
    """Predicate classification training step"""
    inputs, targets, _ = batch
    inputs = inputs.to(trainer.conf.session.device)
    targets = targets.to(trainer.conf.session.device)

    outputs = trainer.model(inputs)
    loss, loss_dict = trainer.criterion(outputs, targets)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        "output": outputs.predicate_scores.detach().cpu(),
        "target": targets.predicate_bce.detach().cpu(),
        "losses": loss_dict,
    }


def pred_class_validation_step(validator: Validator, batch: Batch):
    """Predicate classification validation step"""
    inputs, targets, filenames = batch
    inputs = inputs.to(validator.conf.session.device)
    targets = targets.to(validator.conf.session.device)

    outputs = validator.model(inputs)

    _, loss_dict = validator.criterion(outputs, targets)

    return {
        "output": outputs.predicate_scores.detach().cpu(),
        "target": targets.predicate_bce.detach().cpu(),
        "losses": loss_dict,
    }


def vr_validation_step(validator: Validator, batch: Batch):
    """Visual relationship detection validation step

    Shared by:
    - predicate detection (if run with ground-truth boxes as input)
    - phrase detection (if run with detectron2 boxes as input)
    - relationship detection (if run with detectron2 boxes as input)

    """
    inputs, targets, filenames = batch
    inputs = inputs.to(validator.conf.session.device)

    relations = validator.model(inputs)

    return {
        "relations": {k: r.to("cpu") for k, r in relations.items()},
        "targets": targets,
    }


class PredicateClassificationCriterion(object):
    """Binary cross entropy and ranking loss for predicate classification"""

    def __init__(self, conf: OmegaConf):
        self.bce_weight = conf.bce.weight
        self.rank_weight = conf.rank.weight

    def __call__(
        self, results: Batch, targets: Batch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        loss_total = torch.tensor(0.0, device=results.predicate_scores.device)

        if self.bce_weight > 0:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                results.predicate_scores, targets.predicate_bce, reduction="mean"
            )
            loss_dict["bce"] = bce
            loss_total += self.bce_weight * bce

        if self.rank_weight > 0:
            rank = torch.nn.functional.multilabel_margin_loss(
                results.predicate_scores, targets.predicate_rank, reduction="mean"
            )
            loss_dict["rank"] = rank
            loss_total += self.rank_weight * rank

        loss_dict["total"] = loss_total
        loss_dict = {f"loss/{k}": v.detach().cpu().item() for k, v in loss_dict.items()}

        return loss_total, loss_dict


def build_optimizer_and_scheduler(
    conf: OmegaConf, model: torch.nn.Module
) -> Tuple[Optimizer, Any]:
    conf = OmegaConf.to_container(conf, resolve=True)

    optimizer_fn = import_(conf.pop("name"))
    optimizer = optimizer_fn(model.parameters(), **conf)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2
    )

    return optimizer, scheduler


def build_datasets(conf, seed: int) -> Tuple[Mapping[str, VrDataset], Metadata]:
    register_datasets(conf.folder)

    if "trainval" in conf:
        metadata = MetadataCatalog.get(conf.trainval.name)
        trainval_path = Path(conf.folder) / metadata.graph_root
        trainval_folder = DatasetFolder.from_folder(trainval_path, suffix=".graph.pth")
        train_folder, val_folder = trainval_folder.split(
            conf.trainval.split, random_state=seed
        )
    elif "train" in conf and "val" in conf:
        metadata = MetadataCatalog.get(conf.train.name)
        train_path = Path(conf.folder) / metadata.graph_root
        train_folder = DatasetFolder.from_folder(train_path, suffix=".graph.pth")

        metadata = MetadataCatalog.get(conf.val.name)
        val_path = Path(conf.folder) / metadata.graph_root
        val_folder = DatasetFolder.from_folder(val_path, suffix=".graph.pth")
    else:
        raise ValueError(f"Invalid data specification:\n{conf.pretty()}")

    logger.info(
        f"Data split: {len(train_folder)} train, {len(val_folder)} val ("
        f"{100 * len(train_folder) / (len(train_folder) + len(val_folder)):.1f}/"
        f"{100 * len(val_folder) / (len(train_folder) + len(val_folder)):.1f}%)"
    )

    def make_bce_and_rank_targets(
        input_graph: Batch, target_graph: Batch, *, num_classes
    ):
        """Binary and rank encoding of unique predicates"""
        unique_preds = torch.unique(target_graph.predicate_classes, sorted=False)
        target_graph.predicate_bce = (
            torch.zeros(num_classes, dtype=torch.float)
            .scatter_(dim=0, index=unique_preds, value=1.0)
            .view(1, -1)
        )
        target_graph.predicate_rank = torch.constant_pad_nd(
            unique_preds, pad=(0, num_classes - len(unique_preds)), value=-1
        ).view(1, -1)
        return input_graph, target_graph

    targets = partial(
        make_bce_and_rank_targets, num_classes=len(metadata.predicate_classes)
    )
    datasets = {
        "train_gt": VrDataset(train_folder, input_mode="GT", transforms=targets),
        "val_gt": VrDataset(val_folder, input_mode="GT", transforms=targets),
        "val_d2": VrDataset(val_folder, input_mode="D2", transforms=targets),
    }
    return datasets, metadata


def build_dataloaders(conf, datasets: Mapping[str, Dataset]) -> Dict[str, DataLoader]:
    def collate_fn(batch):
        inputs, targets, filenames = zip(*batch)
        inputs = Batch.from_data_list(inputs)
        targets = Batch.from_data_list(targets)
        return inputs, targets, filenames

    kwargs = dict(
        batch_size=conf.dataloader.batch_size,
        num_workers=conf.dataloader.num_workers,
        pin_memory="cuda" in conf.session.device,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return {
        "train_gt": DataLoader(datasets["train_gt"], shuffle=True, **kwargs),
        "val_gt": DataLoader(datasets["val_gt"], shuffle=False, **kwargs),
        "val_d2": DataLoader(datasets["val_d2"], shuffle=False, **kwargs),
    }


def build_model(conf: OmegaConf, dataset_metadata) -> torch.nn.Module:
    model_fn: Callable[[OmegaConf, Metadata], torch.nn.Module] = import_(conf.name)
    model = model_fn(conf, dataset_metadata)
    return model


def log_metrics(
    engine: Engine,
    name,
    tag: str,
    json_logger: Optional[Path],
    global_step_fn: Callable[[], int],
):
    global_step = global_step_fn()
    metrics = {f"{tag}/{k}": v for k, v in engine.state.metrics.items()}

    yaml = pyaml.dump(metrics, safe=True, sort_dicts=True, force_embed=True)
    logger.info(f"{name} {global_step}:\n{yaml}")

    if json_logger is not None:
        json_metrics = json.dumps(
            {"walltime": time.time(), "global_step": global_step, **metrics}
        )
        with json_logger.open(mode="a") as f:
            f.write(json_metrics + "\n")


def log_effective_config(conf, trainer, tb_logger):
    global_step = trainer.global_step() if "resume" in conf else 0
    yaml = pyaml.dump(
        OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True
    )
    tb_logger.writer.add_text(
        "Configuration", textwrap.indent(yaml, "    "), global_step
    )
    add_session_start(tb_logger.writer, conf.hparams)
    tb_logger.writer.flush()
    p = (
        Path(conf.checkpoint.folder).expanduser()
        / conf.fullname
        / f"conf.{global_step}.yaml"
    )
    with p.open(mode="w") as f:
        f.write(f"# Effective configuration at global step {global_step}\n")
        f.write(yaml)


def epoch_filter(every: int):
    """Ignite event filter that runs every X events or when the engine is terminating"""

    def f(engine: Engine, event: int):
        return (
            engine.state.epoch % every == 0
            or engine.state.epoch == engine.state.max_epochs
            or engine.should_terminate
        )

    return f


@logger.catch(reraise=True)
def main():
    # region Setup
    conf = parse_args()
    setup_seeds(conf.session.seed)
    tb_logger, tb_img_logger, json_logger = setup_all_loggers(conf)
    logger.info(
        "Parsed configuration:\n"
        + pyaml.dump(
            OmegaConf.to_container(conf), safe=True, sort_dicts=False, force_embed=True
        )
    )

    # region Predicate classification engines
    datasets, dataset_metadata = build_datasets(conf.dataset, seed=conf.session.seed)
    dataloaders = build_dataloaders(conf, datasets)

    model = build_model(conf.model, dataset_metadata).to(conf.session.device)
    criterion = PredicateClassificationCriterion(conf.losses)

    pred_class_trainer = Trainer(pred_class_training_step, conf)
    pred_class_trainer.model = model
    pred_class_trainer.criterion = criterion
    pred_class_trainer.optimizer, scheduler = build_optimizer_and_scheduler(
        conf.optimizer, pred_class_trainer.model
    )

    pred_class_validator = Validator(pred_class_validation_step, conf)
    pred_class_validator.model = model
    pred_class_validator.criterion = criterion
    # endregion

    # region Visual Relations engines
    vr_model = VisualRelationExplainer(model, **conf.visual_relations)

    vr_predicate_validator = Validator(vr_validation_step, conf)
    vr_predicate_validator.model = vr_model

    vr_phrase_and_relation_validator = Validator(vr_validation_step, conf)
    vr_phrase_and_relation_validator.model = vr_model
    # endregion

    if "resume" in conf:
        checkpoint = Path(conf.resume.checkpoint).expanduser().resolve()
        logger.debug(f"Resuming checkpoint from {checkpoint}")
        Checkpoint.load_objects(
            {
                "model": pred_class_trainer.model,
                "optimizer": pred_class_trainer.optimizer,
                "scheduler": scheduler,
                "trainer": pred_class_trainer,
            },
            checkpoint=torch.load(checkpoint, map_location=conf.session.device),
        )
        logger.info(
            f"Resumed from {checkpoint}, "
            f"epoch {pred_class_trainer.state.epoch}, "
            f"samples {pred_class_trainer.global_step()}"
        )
    # endregion

    # region Predicate classification training callbacks
    tb_logger.attach(
        pred_class_trainer,
        OptimizerParamsHandler(
            pred_class_trainer.optimizer,
            param_name="lr",
            tag="z",
            global_step_transform=pred_class_trainer.global_step,
        ),
        Events.EPOCH_STARTED,
    )

    MeanAveragePrecisionBatch(output_transform=itemgetter("target", "output")).attach(
        pred_class_trainer, "mAP"
    )
    RecallAtBatch(
        output_transform=itemgetter("target", "output"), sizes=(5, 10)
    ).attach(pred_class_trainer, "recall_at")

    ProgressBar(persist=True, desc="Pred class train").attach(
        pred_class_trainer, metric_names="all", output_transform=itemgetter("losses")
    )

    tb_logger.attach(
        pred_class_trainer,
        OutputHandler(
            "train",
            output_transform=itemgetter("losses"),
            global_step_transform=pred_class_trainer.global_step,
        ),
        Events.ITERATION_COMPLETED,
    )
    tb_logger.attach(
        pred_class_trainer,
        MetricsHandler("train", global_step_transform=pred_class_trainer.global_step),
        Events.ITERATION_COMPLETED,
    )

    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate classification training",
        "train",
        json_logger=None,
        global_step_fn=pred_class_trainer.global_step,
    )
    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicatePredictionLogger(
            grid=(2, 3),
            data_root=conf.dataset.folder,
            tag="train",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata,
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    tb_logger.attach(
        pred_class_trainer,
        EpochHandler(
            pred_class_trainer,
            tag="z",
            global_step_transform=pred_class_trainer.global_step,
        ),
        Events.EPOCH_COMPLETED,
    )

    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda _: pred_class_validator.run(dataloaders["val_gt"]),
    )
    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED(epoch_filter(5)),
        lambda _: vr_predicate_validator.run(dataloaders["val_gt"]),
    )
    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED(epoch_filter(5)),
        lambda _: vr_phrase_and_relation_validator.run(dataloaders["val_d2"]),
    )
    # endregion

    # region Predicate classification validation callbacks
    if conf.losses["bce"]["weight"] > 0:
        Average(
            output_transform=lambda o: o["losses"]["loss/bce"],
            device=conf.session.device,
        ).attach(pred_class_validator, "loss/bce")
    if conf.losses["rank"]["weight"] > 0:
        Average(
            output_transform=lambda o: o["losses"]["loss/rank"],
            device=conf.session.device,
        ).attach(pred_class_validator, "loss/rank")
    Average(
        output_transform=lambda o: o["losses"]["loss/total"], device=conf.session.device
    ).attach(pred_class_validator, "loss/total")

    MeanAveragePrecisionEpoch(itemgetter("target", "output")).attach(
        pred_class_validator, "mAP"
    )
    RecallAtEpoch((5, 10), itemgetter("target", "output")).attach(
        pred_class_validator, "recall_at"
    )

    ProgressBar(persist=True, desc="Pred class val").attach(
        pred_class_validator, metric_names="all", output_transform=itemgetter("losses")
    )

    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda val_engine: scheduler.step(val_engine.state.metrics["loss/total"]),
    )
    tb_logger.attach(
        pred_class_validator,
        MetricsHandler("val", global_step_transform=pred_class_trainer.global_step),
        Events.EPOCH_COMPLETED,
    )
    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate classification validation",
        "val",
        json_logger,
        pred_class_trainer.global_step,
    )
    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicatePredictionLogger(
            grid=(2, 3),
            data_root=conf.dataset.folder,
            tag="val",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata,
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    pred_class_validator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(
            patience=conf.session.early_stopping.patience,
            score_function=lambda val_engine: -val_engine.state.metrics["loss/total"],
            trainer=pred_class_trainer,
        ),
    )
    pred_class_validator.add_event_handler(
        Events.COMPLETED,
        Checkpoint(
            {
                "model": pred_class_trainer.model,
                "optimizer": pred_class_trainer.optimizer,
                "scheduler": scheduler,
                "trainer": pred_class_trainer,
            },
            DiskSaver(
                Path(conf.checkpoint.folder).expanduser().resolve() / conf.fullname
            ),
            score_function=lambda val_engine: val_engine.state.metrics["recall_at_5"],
            score_name="recall_at_5",
            n_saved=conf.checkpoint.keep,
            global_step_transform=pred_class_trainer.global_step,
        ),
    )
    # endregion

    # region Predicate detection validation callbacks
    vr_predicate_validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        VisualRelationRecallAt(type="predicate", steps=(50, 100)),
    )
    for mode in ("with_obj_scores", "no_obj_scores"):
        for step in (50, 100):
            Average(
                output_transform=itemgetter(f"predicate/{mode}/recall_at_{step}")
            ).attach(vr_predicate_validator, f"predicate/{mode}/recall_at_{step}")

    ProgressBar(persist=True, desc="Pred det val").attach(vr_predicate_validator)

    tb_logger.attach(
        vr_predicate_validator,
        MetricsHandler("val_vr", global_step_transform=pred_class_trainer.global_step),
        Events.EPOCH_COMPLETED,
    )
    vr_predicate_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate detection validation",
        "val_vr",
        json_logger,
        pred_class_trainer.global_step,
    )
    vr_predicate_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        VisualRelationPredictionLogger(
            grid=(2, 3),
            data_root=conf.dataset.folder,
            tag="with GT boxes",
            logger=tb_img_logger.writer,
            top_x_relations=conf.visual_relations.top_x_relations,
            metadata=dataset_metadata,
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    # endregion

    # region Phrase and relationship detection validation callbacks
    vr_phrase_and_relation_validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        VisualRelationRecallAt(type="phrase", steps=(50, 100)),
    )
    vr_phrase_and_relation_validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        VisualRelationRecallAt(type="relationship", steps=(50, 100)),
    )
    for mode in ("with_obj_scores", "no_obj_scores"):
        for name in ["phrase", "relationship"]:
            for step in (50, 100):
                Average(
                    output_transform=itemgetter(f"{name}/{mode}/recall_at_{step}")
                ).attach(
                    vr_phrase_and_relation_validator, f"{name}/{mode}/recall_at_{step}"
                )
    if conf.dataset.name == "hico":
        HOImAP().attach(vr_phrase_and_relation_validator, "hoi/mAP")

    ProgressBar(persist=True, desc="Phrase and relation det val").attach(
        vr_phrase_and_relation_validator
    )

    tb_logger.attach(
        vr_phrase_and_relation_validator,
        MetricsHandler("val_vr", global_step_transform=pred_class_trainer.global_step),
        Events.EPOCH_COMPLETED,
    )
    vr_phrase_and_relation_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Phrase and relationship detection validation",
        "val_vr",
        json_logger,
        pred_class_trainer.global_step,
    )
    vr_phrase_and_relation_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        VisualRelationPredictionLogger(
            grid=(2, 3),
            data_root=conf.dataset.folder,
            tag="with D2 boxes",
            logger=tb_img_logger.writer,
            top_x_relations=conf.visual_relations.top_x_relations,
            metadata=dataset_metadata,
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    # endregion

    # region Run
    if conf.dataset.eager:
        for d in datasets.values():
            d.load_eager()
    log_effective_config(conf, pred_class_trainer, tb_logger)
    max_epochs = conf.session.max_epochs
    if "resume" in conf:
        max_epochs += pred_class_trainer.state.epoch
    pred_class_trainer.run(
        dataloaders["train_gt"],
        max_epochs=max_epochs,
        seed=conf.session.seed,
        epoch_length=len(dataloaders["train_gt"]),
    )

    add_session_end(tb_logger.writer, "SUCCESS")
    tb_logger.close()
    tb_img_logger.close()
    # endregion


if __name__ == "__main__":
    setup_logging()
    main()
