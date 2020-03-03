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

from xib.metrics.relation_det import (
    VisualRelationPredictionExporter,
    VrdZeroShotVisualRelationRecallAt,
)
from .config import parse_args
from .datasets import DatasetFolder, VrDataset, register_datasets
from .ignite import Trainer, Validator, OptimizerParamsHandler, EpochHandler
from .logging import setup_logging, add_logfile, add_custom_scalars
from .logging.hyperparameters import (
    add_hparam_summary,
    add_session_start,
    add_session_end,
)
from .metrics.pred_class import (
    RecallAtBatch,
    RecallAtEpoch,
    PredicateClassificationLogger,
    PredicateClassificationMeanAveragePrecisionBatch,
    PredicateClassificationMeanAveragePrecisionEpoch,
)
from .metrics.relation_det import (
    HoiDetectionMeanAvgPrecision,
    VisualRelationPredictionLogger,
    HoiClassificationMeanAvgPrecision,
    UnRelDetectionMeanAvgPrecision,
    VisualRelationRecallAt,
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
        "relations": relations.to("cpu"),
        "targets": targets,
        "filenames": filenames,
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


def build_datasets(
    conf, seed: int
) -> Tuple[Mapping[str, VrDataset], Mapping[str, Metadata]]:
    register_datasets(conf.folder)

    metadata = {}
    transforms = []

    if "trainval" in conf:
        metadata["train_gt"] = metadata["val_gt"] = metadata[
            "val_d2"
        ] = MetadataCatalog.get(conf.trainval.name)
        trainval_path = Path(conf.folder) / metadata["train_gt"].graph_root
        trainval_folder = DatasetFolder.from_folder(trainval_path, suffix=".graph.pth")
        train_folder, val_folder = trainval_folder.split(
            conf.trainval.split, random_state=seed
        )
    elif "train" in conf and "val" in conf:
        metadata["train_gt"] = MetadataCatalog.get(conf.train.name)
        train_path = Path(conf.folder) / metadata["train_gt"].graph_root
        train_folder = DatasetFolder.from_folder(train_path, suffix=".graph.pth")

        metadata["val_gt"] = metadata["val_d2"] = MetadataCatalog.get(conf.val.name)
        val_path = Path(conf.folder) / metadata["val_gt"].graph_root
        val_folder = DatasetFolder.from_folder(val_path, suffix=".graph.pth")
    else:
        raise ValueError(f"Invalid data specification:\n{conf.pretty()}")

    logger.info(
        f"Data split: {len(train_folder)} train, {len(val_folder)} val ("
        f"{100 * len(train_folder) / (len(train_folder) + len(val_folder)):.1f}/"
        f"{100 * len(val_folder) / (len(train_folder) + len(val_folder)):.1f}%)"
    )

    def make_bce_and_rank_targets(
        input_graph: Batch, target_graph: Batch, filename: str, *, num_classes
    ):
        """Binary and rank encoding of unique predicates"""
        unique_predicates = torch.unique(target_graph.predicate_classes, sorted=False)
        target_graph.predicate_bce = (
            torch.zeros(num_classes, dtype=torch.float)
            .scatter_(dim=0, index=unique_predicates, value=1.0)
            .view(1, -1)
        )
        target_graph.predicate_rank = torch.constant_pad_nd(
            unique_predicates, pad=(0, num_classes - len(unique_predicates)), value=-1
        ).view(1, -1)
        return input_graph, target_graph

    def keep_human_object_interactions(
        input_graph: Batch, target_graph: Batch, filename: str, *, human_class: int
    ):
        subjs = input_graph.object_classes[input_graph.relation_indexes[0]]
        keep = subjs == human_class

        input_graph.n_edges = keep.sum().item()
        input_graph.relation_indexes = input_graph.relation_indexes[:, keep]
        input_graph.relation_linear_features = input_graph.relation_linear_features[
            keep
        ]

        return input_graph, target_graph

    transforms.append(
        partial(
            make_bce_and_rank_targets,
            num_classes=len(metadata["train_gt"].predicate_classes),
        )
    )

    if conf.graph_type == "human-object":
        transforms.append(
            partial(
                keep_human_object_interactions,
                human_class=metadata["train_gt"].thing_classes.index("person"),
            )
        )

    datasets = {
        "train_gt": VrDataset(
            train_folder,
            input_mode="GT",
            transforms=transforms,
            metadata=metadata["train_gt"],
        ),
        "val_gt": VrDataset(
            val_folder,
            input_mode="GT",
            transforms=transforms,
            metadata=metadata["val_gt"],
        ),
        "val_d2": VrDataset(
            val_folder,
            input_mode="D2",
            transforms=transforms,
            metadata=metadata["val_d2"],
        ),
    }

    if "test" in conf:
        metadata["test_gt"] = metadata["test_d2"] = MetadataCatalog.get(conf.test.name)
        test_path = Path(conf.folder) / metadata["test_gt"].graph_root
        test_folder = DatasetFolder.from_folder(test_path, suffix=".graph.pth")
        logger.info(f"Test split: {len(test_folder)} test")
        datasets["test_gt"] = VrDataset(
            test_folder,
            input_mode="GT",
            transforms=transforms,
            metadata=metadata["test_gt"],
        )
        datasets["test_d2"] = VrDataset(
            test_folder,
            input_mode="D2",
            transforms=transforms,
            metadata=metadata["test_d2"],
        )

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

    dataloaders = {
        "train_gt": DataLoader(datasets["train_gt"], shuffle=True, **kwargs),
        "val_gt": DataLoader(datasets["val_gt"], shuffle=False, **kwargs),
        "val_d2": DataLoader(datasets["val_d2"], shuffle=False, **kwargs),
    }

    if "test" in conf.dataset:
        kwargs["drop_last"] = False
        dataloaders["test_gt"] = DataLoader(
            datasets["test_gt"], shuffle=False, **kwargs
        )
        dataloaders["test_d2"] = DataLoader(
            datasets["test_d2"], shuffle=False, **kwargs
        )

    return dataloaders


def build_model(conf: OmegaConf, dataset_metadata) -> torch.nn.Module:
    model_fn: Callable[[OmegaConf, Metadata], torch.nn.Module] = import_(conf.name)
    model = model_fn(conf, dataset_metadata)
    return model


def log_metrics(
    engine: Engine,
    name,
    tag: str,
    json_logger: Optional[Path],
    tb_logger: Optional[TensorboardLogger],
    global_step_fn: Callable[[], int],
):
    global_step = global_step_fn()

    metrics = {}
    for k, v in engine.state.metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        metrics[f"{tag}/{k}"] = v

    yaml = pyaml.dump(metrics, safe=True, sort_dicts=True, force_embed=True)
    logger.info(f"{name} {global_step}:\n{yaml}")

    if tb_logger is not None:
        for k, v in metrics.items():
            tb_logger.writer.add_scalar(k, v, global_step)

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

    model = build_model(conf.model, dataset_metadata["train_gt"]).to(
        conf.session.device
    )
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

    pred_class_tester = Validator(pred_class_validation_step, conf)
    pred_class_tester.model = model
    pred_class_tester.criterion = criterion
    # endregion

    # region Visual Relations engines
    vr_model = VisualRelationExplainer(
        model, dataset_metadata["train_gt"], **conf.visual_relations
    )

    vr_predicate_validator = Validator(vr_validation_step, conf)
    vr_predicate_validator.model = vr_model

    vr_predicate_tester = Validator(vr_validation_step, conf)
    vr_predicate_tester.model = vr_model

    vr_phrase_and_relation_validator = Validator(vr_validation_step, conf)
    vr_phrase_and_relation_validator.model = vr_model

    vr_phrase_and_relation_tester = Validator(vr_validation_step, conf)
    vr_phrase_and_relation_tester.model = vr_model
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
    def increment_samples(trainer: Trainer):
        graphs: Batch = trainer.state.batch[0]
        trainer.state.samples += graphs.num_graphs

    pred_class_trainer.add_event_handler(Events.ITERATION_COMPLETED, increment_samples)

    ProgressBar(persist=True, desc="[train] Predicate classification").attach(
        pred_class_trainer, output_transform=itemgetter("losses")
    )

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

    pred_class_trainer.add_event_handler(
        Events.ITERATION_COMPLETED, PredicateClassificationMeanAveragePrecisionBatch()
    )
    pred_class_trainer.add_event_handler(
        Events.ITERATION_COMPLETED, RecallAtBatch(sizes=(5, 10))
    )

    tb_logger.attach(
        pred_class_trainer,
        OutputHandler(
            "train_gt",
            output_transform=lambda o: {
                **o["losses"],
                "pc/mAP": o["pc/mAP"].mean().item(),
                **{k: r.mean().item() for k, r in o["recalls"].items()},
            },
            global_step_transform=pred_class_trainer.global_step,
        ),
        Events.ITERATION_COMPLETED,
    )

    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate classification training",
        "train_gt",
        json_logger=None,
        tb_logger=tb_logger,
        global_step_fn=pred_class_trainer.global_step,
    )
    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicateClassificationLogger(
            grid=(2, 3),
            tag="train_gt",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata["train_gt"],
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
        Events.EPOCH_COMPLETED(epoch_filter(2)),
        lambda _: vr_phrase_and_relation_validator.run(dataloaders["val_d2"]),
    )
    # endregion

    # region Predicate classification validation callbacks
    ProgressBar(persist=True, desc="[val] Predicate classification").attach(
        pred_class_validator
    )

    if conf.losses["bce"]["weight"] > 0:
        Average(output_transform=lambda o: o["losses"]["loss/bce"]).attach(
            pred_class_validator, "loss/bce"
        )
    if conf.losses["rank"]["weight"] > 0:
        Average(output_transform=lambda o: o["losses"]["loss/rank"]).attach(
            pred_class_validator, "loss/rank"
        )
    Average(output_transform=lambda o: o["losses"]["loss/total"]).attach(
        pred_class_validator, "loss/total"
    )

    PredicateClassificationMeanAveragePrecisionEpoch(
        itemgetter("target", "output")
    ).attach(pred_class_validator, "pc/mAP")
    RecallAtEpoch((5, 10), itemgetter("target", "output")).attach(
        pred_class_validator, "pc/recall_at"
    )

    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda val_engine: scheduler.step(val_engine.state.metrics["loss/total"]),
    )
    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate classification validation",
        "val_gt",
        json_logger,
        tb_logger,
        pred_class_trainer.global_step,
    )
    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicateClassificationLogger(
            grid=(2, 3),
            tag="val_gt",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata["val_gt"],
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
            score_function=lambda val_engine: val_engine.state.metrics[
                "pc/recall_at_5"
            ],
            score_name="pc_recall_at_5",
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
    for step in (50, 100):
        Average(output_transform=itemgetter(f"vr/predicate/recall_at_{step}")).attach(
            vr_predicate_validator, f"vr/predicate/recall_at_{step}"
        )

    if conf.dataset.name in {"unrel", "unrel_vrd"}:
        UnRelDetectionMeanAvgPrecision("GT").attach(
            vr_predicate_validator, "vr/unrel/mAP"
        )

    ProgressBar(persist=True, desc="[val] Predicate detection").attach(
        vr_predicate_validator
    )

    vr_predicate_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Predicate detection validation",
        "val_gt",
        json_logger,
        tb_logger,
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
            metadata=dataset_metadata["val_gt"],
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    # endregion

    # region Phrase and relationship detection validation callbacks
    ProgressBar(persist=True, desc="[val] Phrase and relation detection").attach(
        vr_phrase_and_relation_validator
    )
    vr_phrase_and_relation_validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        VisualRelationRecallAt(type="phrase", steps=(50, 100)),
    )
    vr_phrase_and_relation_validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        VisualRelationRecallAt(type="relationship", steps=(50, 100)),
    )
    for name in ["phrase", "relationship"]:
        for step in (50, 100):
            Average(output_transform=itemgetter(f"vr/{name}/recall_at_{step}")).attach(
                vr_phrase_and_relation_validator, f"vr/{name}/recall_at_{step}"
            )
    if conf.dataset.name == "hico":
        HoiClassificationMeanAvgPrecision().attach(
            vr_phrase_and_relation_validator, "pc/hoi/mAP"
        )
        HoiDetectionMeanAvgPrecision().attach(
            vr_phrase_and_relation_validator, "vr/hoi/mAP"
        )
    elif conf.dataset.name in {"unrel", "unrel_vrd"}:
        UnRelDetectionMeanAvgPrecision("D2").attach(
            vr_phrase_and_relation_validator, "vr/unrel/mAP"
        )

    vr_phrase_and_relation_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        "Phrase and relationship detection validation",
        "val_d2",
        json_logger,
        tb_logger,
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
            metadata=dataset_metadata["val_d2"],
            global_step_fn=pred_class_trainer.global_step,
        ),
    )
    # endregion

    if "test" in conf.dataset:
        # region Predicate classification testing callbacks
        if conf.losses["bce"]["weight"] > 0:
            Average(
                output_transform=lambda o: o["losses"]["loss/bce"],
                device=conf.session.device,
            ).attach(pred_class_tester, "loss/bce")
        if conf.losses["rank"]["weight"] > 0:
            Average(
                output_transform=lambda o: o["losses"]["loss/rank"],
                device=conf.session.device,
            ).attach(pred_class_tester, "loss/rank")
        Average(
            output_transform=lambda o: o["losses"]["loss/total"],
            device=conf.session.device,
        ).attach(pred_class_tester, "loss/total")

        PredicateClassificationMeanAveragePrecisionEpoch(
            itemgetter("target", "output")
        ).attach(pred_class_tester, "pc/mAP")
        RecallAtEpoch((5, 10), itemgetter("target", "output")).attach(
            pred_class_tester, "pc/recall_at"
        )

        ProgressBar(persist=True, desc="[test] Predicate classification").attach(
            pred_class_tester
        )

        pred_class_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_metrics,
            "Predicate classification test",
            "test_gt",
            json_logger,
            tb_logger,
            pred_class_trainer.global_step,
        )
        pred_class_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            PredicateClassificationLogger(
                grid=(2, 3),
                tag="test_gt",
                logger=tb_img_logger.writer,
                metadata=dataset_metadata["test_gt"],
                global_step_fn=pred_class_trainer.global_step,
            ),
        )
        # endregion

        # region Predicate detection testing callbacks
        vr_predicate_tester.add_event_handler(
            Events.ITERATION_COMPLETED,
            VisualRelationRecallAt(type="predicate", steps=(50, 100)),
        )
        for step in (50, 100):
            Average(
                output_transform=itemgetter(f"vr/predicate/recall_at_{step}")
            ).attach(vr_predicate_tester, f"vr/predicate/recall_at_{step}")

        if (
            "test" in conf.dataset
            and conf.dataset.test.name == "vrd_relationship_detection_zero_shot"
        ):
            vr_predicate_tester.add_event_handler(
                Events.ITERATION_COMPLETED,
                VrdZeroShotVisualRelationRecallAt(type="predicate", steps=(50, 100)),
            )
            for step in (50, 100):
                Average(
                    output_transform=itemgetter(
                        f"vr/predicate/zero_shot/recall_at_{step}"
                    )
                ).attach(
                    vr_predicate_tester, f"vr/predicate/zero_shot/recall_at_{step}"
                )

        if conf.dataset.name in {"unrel", "unrel_vrd"}:
            UnRelDetectionMeanAvgPrecision("GT").attach(
                vr_predicate_tester, "vr/unrel/mAP"
            )

        ProgressBar(persist=True, desc="[test] Predicate detection").attach(
            vr_predicate_tester
        )

        vr_predicate_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_metrics,
            "Predicate detection test",
            "test_gt",
            json_logger,
            tb_logger,
            pred_class_trainer.global_step,
        )
        vr_predicate_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            VisualRelationPredictionLogger(
                grid=(2, 3),
                data_root=conf.dataset.folder,
                tag="with GT boxes",
                logger=tb_img_logger.writer,
                top_x_relations=conf.visual_relations.top_x_relations,
                metadata=dataset_metadata["test_gt"],
                global_step_fn=pred_class_trainer.global_step,
            ),
        )
        vr_predicate_tester.add_event_handler(
            Events.ITERATION_COMPLETED,
            VisualRelationPredictionExporter(
                mode="GT", dest=Path(conf.checkpoint.folder) / conf.fullname
            ),
        )
        # endregion

        # region Phrase and relationship detection testing callbacks
        ProgressBar(persist=True, desc="[test] Phrase and relation detection").attach(
            vr_phrase_and_relation_tester
        )
        vr_phrase_and_relation_tester.add_event_handler(
            Events.ITERATION_COMPLETED,
            VisualRelationRecallAt(type="phrase", steps=(50, 100)),
        )
        vr_phrase_and_relation_tester.add_event_handler(
            Events.ITERATION_COMPLETED,
            VisualRelationRecallAt(type="relationship", steps=(50, 100)),
        )
        for name in ["phrase", "relationship"]:
            for step in (50, 100):
                Average(
                    output_transform=itemgetter(f"vr/{name}/recall_at_{step}")
                ).attach(vr_phrase_and_relation_tester, f"vr/{name}/recall_at_{step}")

        if conf.dataset.name == "hico":
            HoiClassificationMeanAvgPrecision().attach(
                vr_phrase_and_relation_tester, "pc/hoi/mAP"
            )
            HoiDetectionMeanAvgPrecision().attach(
                vr_phrase_and_relation_tester, "vr/hoi/mAP"
            )
        elif conf.dataset.name in {"unrel", "unrel_vrd"}:
            UnRelDetectionMeanAvgPrecision("D2").attach(
                vr_phrase_and_relation_tester, "vr/unrel/mAP"
            )
        if (
            "test" in conf.dataset
            and conf.dataset.test.name == "vrd_relationship_detection_zero_shot"
        ):
            vr_phrase_and_relation_tester.add_event_handler(
                Events.ITERATION_COMPLETED,
                VrdZeroShotVisualRelationRecallAt(type="phrase", steps=(50, 100)),
            )
            vr_phrase_and_relation_tester.add_event_handler(
                Events.ITERATION_COMPLETED,
                VrdZeroShotVisualRelationRecallAt(type="relationship", steps=(50, 100)),
            )
            for name in ["phrase", "relationship"]:
                for step in (50, 100):
                    Average(
                        output_transform=itemgetter(
                            f"vr/{name}/zero_shot/recall_at_{step}"
                        )
                    ).attach(
                        vr_phrase_and_relation_tester,
                        f"vr/{name}/zero_shot/recall_at_{step}",
                    )

        vr_phrase_and_relation_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_metrics,
            "Phrase and relationship detection test",
            "test_d2",
            json_logger,
            tb_logger,
            pred_class_trainer.global_step,
        )
        vr_phrase_and_relation_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            VisualRelationPredictionLogger(
                grid=(2, 3),
                data_root=conf.dataset.folder,
                tag="with D2 boxes",
                logger=tb_img_logger.writer,
                top_x_relations=conf.visual_relations.top_x_relations,
                metadata=dataset_metadata["test_d2"],
                global_step_fn=pred_class_trainer.global_step,
            ),
        )
        vr_phrase_and_relation_tester.add_event_handler(
            Events.ITERATION_COMPLETED,
            VisualRelationPredictionExporter(
                mode="D2", dest=Path(conf.checkpoint.folder) / conf.fullname
            ),
        )
        # endregion

    # region Run
    log_effective_config(conf, pred_class_trainer, tb_logger)
    if not ("resume" in conf and conf.resume.test_only):
        if conf.dataset.eager:
            for n, d in datasets.items():
                if n.startswith(("train", "val")):
                    d.load_eager()
        max_epochs = conf.session.max_epochs
        if "resume" in conf:
            max_epochs += pred_class_trainer.state.epoch
        pred_class_trainer.run(
            dataloaders["train_gt"],
            max_epochs=max_epochs,
            seed=conf.session.seed,
            epoch_length=len(dataloaders["train_gt"]),
        )

    if "test" in conf.dataset:
        if conf.dataset.eager:
            datasets["test_gt"].load_eager()
            datasets["test_d2"].load_eager()
        pred_class_tester.run(dataloaders["test_gt"])
        vr_predicate_tester.run(dataloaders["test_gt"])
        vr_phrase_and_relation_tester.run(dataloaders["test_d2"])

    add_session_end(tb_logger.writer, "SUCCESS")
    tb_logger.close()
    tb_img_logger.close()
    # endregion


if __name__ == "__main__":
    setup_logging()
    main()
