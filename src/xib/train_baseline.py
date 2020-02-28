import json
import os
import random
import socket
import textwrap
import time
from operator import itemgetter
from pathlib import Path
from typing import Callable, Tuple, Any, Dict, Mapping, Optional

import numpy as np
import pyaml
import torch
import torch.utils.data
from detectron2.data import MetadataCatalog, DatasetCatalog
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
from torchvision.transforms import RandomResizedCrop, Resize, CenterCrop

from xib.datasets import PcDataset
from .config import parse_args
from .datasets import register_datasets
from .ignite import MeanAveragePrecisionEpoch, MeanAveragePrecisionBatch
from .ignite import OptimizerParamsHandler, EpochHandler
from .ignite import PredicatePredictionLogger
from .ignite import RecallAtBatch, RecallAtEpoch
from .ignite import Trainer, Validator
from .logging import setup_logging, add_logfile, add_custom_scalars
from .logging.hyperparameters import (
    add_hparam_summary,
    add_session_start,
    add_session_end,
)
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


def pred_class_training_step(trainer: Trainer, batch):
    """Predicate classification training step"""
    inputs, targets, _ = batch
    inputs = inputs.to(trainer.conf.session.device)
    targets = {k: t.to(trainer.conf.session.device) for k, t in targets.items()}

    outputs = trainer.model(inputs)
    loss, loss_dict = trainer.criterion(outputs, targets)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    return {
        "output": outputs.detach().cpu(),
        "target": targets["bce"].detach().cpu(),
        "losses": loss_dict,
    }


def pred_class_validation_step(validator: Validator, batch):
    """Predicate classification validation step"""
    inputs, targets, filenames = batch
    inputs = inputs.to(validator.conf.session.device)
    targets = {k: t.to(validator.conf.session.device) for k, t in targets.items()}

    outputs = validator.model(inputs)

    _, loss_dict = validator.criterion(outputs, targets)

    return {
        "output": outputs.detach().cpu(),
        "target": targets["bce"].detach().cpu(),
        "losses": loss_dict,
    }


class PredicateClassificationCriterion(object):
    """Binary cross entropy and ranking loss for predicate classification"""

    def __init__(self, conf: OmegaConf):
        self.bce_weight = conf.bce.weight
        self.rank_weight = conf.rank.weight

    def __call__(
        self, results: torch.Tensor, targets: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        loss_total = torch.tensor(0.0, device=results.device)

        if self.bce_weight > 0:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                results, targets["bce"], reduction="mean"
            )
            loss_dict["bce"] = bce
            loss_total += self.bce_weight * bce

        if self.rank_weight > 0:
            rank = torch.nn.functional.multilabel_margin_loss(
                results, targets["rank"], reduction="mean"
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
    lr = conf.pop("lr") if "lr" in conf else 1e-3
    optimizer = optimizer_fn(
        [
            {"params": model.backbone.parameters(), "lr": lr * 0.1},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        **conf,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2
    )

    return optimizer, scheduler


def build_datasets(conf) -> Tuple[Mapping[str, PcDataset], Mapping[str, Metadata]]:
    register_datasets(conf.folder)

    datasets = {}
    metadata = {}

    if "trainval" in conf:
        metadata["train"] = metadata["val"] = MetadataCatalog.get(conf.trainval.name)
        data_dicts = DatasetCatalog.get(conf.trainval.name)
        data_dicts = data_dicts
        datasets["train"], datasets["val"] = torch.utils.data.random_split(
            PcDataset(data_dicts, metadata["train"], [RandomResizedCrop(224)]),
            [
                int(conf.trainval.split * len(data_dicts)),
                len(data_dicts) - int(conf.trainval.split * len(data_dicts)),
            ],
        )
    elif "train" in conf and "val" in conf:
        metadata["train"] = MetadataCatalog.get(conf.train.name)
        train_data_dicts = DatasetCatalog.get(conf.train.name)
        datasets["train"] = PcDataset(
            train_data_dicts, metadata["train"], [RandomResizedCrop(224)]
        )

        metadata["val"] = MetadataCatalog.get(conf.val.name)
        val_data_dicts = DatasetCatalog.get(conf.val.name)
        datasets["val"] = PcDataset(
            val_data_dicts, metadata["val"], [Resize(224), CenterCrop(224)]
        )
    else:
        raise ValueError(f"Invalid data specification:\n{conf.pretty()}")

    logger.info(
        f"Data split: {len(datasets['train'])} train, {len(datasets['val'])} val ("
        f"{100 * len(datasets['train']) / (len(datasets['train']) + len(datasets['val'])):.1f}/"
        f"{100 * len(datasets['val']) / (len(datasets['train']) + len(datasets['val'])):.1f}%)"
    )

    if "test" in conf:
        metadata["test"] = MetadataCatalog.get(conf.test.name)
        test_data_dicts = DatasetCatalog.get(conf.test.name)
        datasets["test"] = PcDataset(
            test_data_dicts, metadata["test"], [Resize(224), CenterCrop(224)]
        )
        logger.info(f"Test split: {len(datasets['test'])} test")

    return datasets, metadata


def build_dataloaders(conf, datasets: Mapping[str, Dataset]) -> Dict[str, DataLoader]:
    kwargs = dict(
        batch_size=conf.dataloader.batch_size,
        num_workers=conf.dataloader.num_workers,
        pin_memory="cuda" in conf.session.device,
        drop_last=True,
    )

    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **kwargs),
    }

    if "test" in conf.dataset:
        kwargs["drop_last"] = False
        dataloaders["test"] = DataLoader(datasets["test"], shuffle=False, **kwargs)

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
    datasets, dataset_metadata = build_datasets(conf.dataset)
    dataloaders = build_dataloaders(conf, datasets)

    model = build_model(conf.model, dataset_metadata["train"]).to(conf.session.device)
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
        images = trainer.state.batch[0]
        trainer.state.samples += len(images)

    pred_class_trainer.add_event_handler(Events.ITERATION_COMPLETED, increment_samples)

    ProgressBar(persist=True, desc="Pred class train").attach(
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
        Events.ITERATION_COMPLETED, MeanAveragePrecisionBatch()
    )
    pred_class_trainer.add_event_handler(
        Events.ITERATION_COMPLETED, RecallAtBatch(sizes=(5, 10))
    )

    tb_logger.attach(
        pred_class_trainer,
        OutputHandler(
            "train",
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
        "train",
        json_logger=None,
        tb_logger=tb_logger,
        global_step_fn=pred_class_trainer.global_step,
    )
    pred_class_trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicatePredictionLogger(
            grid=(2, 3),
            tag="train",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata["train"],
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
        Events.EPOCH_COMPLETED, lambda _: pred_class_validator.run(dataloaders["val"])
    )
    # endregion

    # region Predicate classification validation callbacks
    ProgressBar(persist=True, desc="Pred class val").attach(pred_class_validator)

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

    MeanAveragePrecisionEpoch(itemgetter("target", "output")).attach(
        pred_class_validator, "pc/mAP"
    )
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
        "val",
        json_logger,
        tb_logger,
        pred_class_trainer.global_step,
    )
    pred_class_validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        PredicatePredictionLogger(
            grid=(2, 3),
            tag="val",
            logger=tb_img_logger.writer,
            metadata=dataset_metadata["val"],
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

        MeanAveragePrecisionEpoch(itemgetter("target", "output")).attach(
            pred_class_tester, "pc/mAP"
        )
        RecallAtEpoch((5, 10), itemgetter("target", "output")).attach(
            pred_class_tester, "pc/recall_at"
        )

        ProgressBar(persist=True, desc="Pred class test").attach(pred_class_tester)

        pred_class_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            log_metrics,
            "Predicate classification test",
            "test",
            json_logger,
            tb_logger,
            pred_class_trainer.global_step,
        )
        pred_class_tester.add_event_handler(
            Events.EPOCH_COMPLETED,
            PredicatePredictionLogger(
                grid=(2, 3),
                tag="test",
                logger=tb_img_logger.writer,
                metadata=dataset_metadata["test"],
                global_step_fn=pred_class_trainer.global_step,
            ),
        )
        # endregion

    # region Run
    log_effective_config(conf, pred_class_trainer, tb_logger)
    if not ("resume" in conf and conf.resume.test_only):
        max_epochs = conf.session.max_epochs
        if "resume" in conf:
            max_epochs += pred_class_trainer.state.epoch
        pred_class_trainer.run(
            dataloaders["train"],
            max_epochs=max_epochs,
            seed=conf.session.seed,
            epoch_length=len(dataloaders["train"]),
        )

    if "test" in conf.dataset:
        pred_class_tester.run(dataloaders["test"])

    add_session_end(tb_logger.writer, "SUCCESS")
    tb_logger.close()
    tb_img_logger.close()
    # endregion


if __name__ == "__main__":
    setup_logging()
    main()
