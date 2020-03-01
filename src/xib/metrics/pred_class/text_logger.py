from pathlib import Path
from typing import Tuple, Callable, Optional, Union, Iterator

import numpy as np
import torch
from PIL import Image
from detectron2.data.catalog import Metadata
from ignite.engine import Engine
from tensorboardX import SummaryWriter

from .mean_avg_prec import mean_average_precision
from .recall_at import recall_at
from xib.structures import Vocabulary, ImageSize


class PredicateClassificationLogger(object):
    def __init__(
        self,
        grid: Tuple[int, int],
        tag: str,
        logger: SummaryWriter,
        global_step_fn: Callable[[], int],
        metadata: Metadata,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """

        Args:
            grid:
            img_dir: directory where the images will be opened from
            tag:
            logger: tensorboard logger for the images
            global_step_fn:
            save_dir: optional destination for .jpg images
        """
        self.tag = tag
        self.grid = grid
        self.logger = logger
        self.global_step_fn = global_step_fn
        self.predicate_vocabulary = Vocabulary(metadata.predicate_classes)

        self.img_dir = metadata.image_root
        if not self.img_dir.is_dir():
            raise ValueError(f"Image dir must exist: {self.img_dir}")

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = Path(self.logger.logdir).expanduser().resolve()
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, engine: Engine):
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")

        global_step = self.global_step_fn()

        predicate_probs = engine.state.output["output"].sigmoid()
        targets_bce = engine.state.output["target"]
        filenames = engine.state.batch[2]

        fig, axes = plt.subplots(*self.grid, figsize=(16, 12), dpi=50)
        axes_iter: Iterator[plt.Axes] = axes.flat

        for target, pred, filename, ax in zip(
            targets_bce, predicate_probs, filenames, axes_iter
        ):
            # Some images are black and white, make sure they are read as RBG
            image = Image.open(self.img_dir.joinpath(filename)).convert("RGB")
            img_size = ImageSize(image.size[1], image.size[0])
            image = np.asarray(image)

            recall_at_5 = recall_at(target[None, :], pred[None, :], (5,))[5]
            mAP = mean_average_precision(target[None, :], pred[None, :])

            ax.imshow(image)
            ax.set_title(
                f"{Path(filename).name[:-4]} mAP {mAP:.1%} R@5 {recall_at_5:.1%}"
            )

            target_str = self.predicate_vocabulary.get_str(
                target.nonzero().flatten()
            ).tolist()
            ax.text(
                0.05,
                0.95,
                "\n".join(target_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="square", facecolor="white", alpha=0.8),
            )

            top_5 = torch.argsort(pred, descending=True)[:5]
            prediction_str = [
                f"{score:.1%} {str}"
                for score, str in zip(
                    pred[top_5], self.predicate_vocabulary.get_str(top_5)
                )
            ]
            ax.text(
                0.65,
                0.95,
                "\n".join(prediction_str),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="square", facecolor="white", alpha=0.8),
            )

            ax.tick_params(
                which="both",
                **{
                    k: False
                    for k in (
                        "bottom",
                        "top",
                        "left",
                        "right",
                        "labelbottom",
                        "labeltop",
                        "labelleft",
                        "labelright",
                    )
                },
            )
            ax.set_xlim(0, img_size.width)
            ax.set_ylim(img_size.height, 0)

        fig.tight_layout()

        if self.save_dir is not None:
            import io

            with io.BytesIO() as buff:
                fig.savefig(
                    buff, format="png", facecolor="white", bbox_inches="tight", dpi=50
                )
                pil_img = Image.open(buff).convert("RGB")
                plt.close(fig)
            save_path = self.save_dir.joinpath(f"{global_step}.{self.tag}.jpg")
            pil_img.save(save_path, "JPEG")
            self.logger.add_image(
                f"{self.tag}",
                np.moveaxis(np.asarray(pil_img), 2, 0),
                global_step=global_step,
            )
        else:
            self.logger.add_figure(
                f"{self.tag}", fig, global_step=global_step, close=True
            )
