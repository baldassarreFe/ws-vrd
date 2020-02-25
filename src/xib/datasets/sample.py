import dataclasses
from pathlib import Path
from typing import Union, Optional, Dict, Any, Mapping

import torch
from detectron2.structures import Instances

from xib.structures import ImageSize
from xib.structures import VisualRelations
from xib.structures import Vocabulary
from xib.structures.instances import (
    to_data_dict as instance_to_data_dict,
    from_data_dict as instance_from_data_dict,
)


@dataclasses.dataclass
class VrSample(object):
    # Metadata
    filename: Union[Path, str]
    img_size: ImageSize

    # Ground-truth detections and visual relations
    gt_instances: Optional[Instances] = None
    gt_visual_relations: Optional[VisualRelations] = None
    gt_visual_relations_full: Optional[VisualRelations] = None

    # Input features: image features, detection boxes, box features
    d2_feature_pyramid: Optional[Dict[str, torch.Tensor]] = None
    d2_instances: Optional[Instances] = None
    d2_visual_relations_full: Optional[VisualRelations] = None

    def __post_init__(self):
        if not isinstance(self.img_size, ImageSize):
            self.img_size = ImageSize(*self.img_size)

    def to_data_dict(self):
        res = {"filename": self.filename, "img_size": tuple(self.img_size)}

        if self.gt_instances is not None:
            res["gt_instances"] = instance_to_data_dict(self.gt_instances)
        if self.gt_visual_relations is not None:
            res["gt_visual_relations"] = self.gt_visual_relations.to_data_dict()
        if self.gt_visual_relations_full is not None:
            res[
                "gt_visual_relations_full"
            ] = self.gt_visual_relations_full.to_data_dict()

        if self.d2_instances is not None:
            res["d2_instances"] = instance_to_data_dict(self.d2_instances)
        if self.d2_feature_pyramid is not None:
            res["d2_feature_pyramid"] = self.d2_feature_pyramid
        if self.d2_visual_relations_full is not None:
            res[
                "d2_visual_relations_full"
            ] = self.d2_visual_relations_full.to_data_dict()

    @classmethod
    def from_data_dict(cls, dict: Mapping[str, Any]):
        res = cls(dict["filename"], dict["img_size"])
        if "gt_instances" in dict:
            res.gt_instances = instance_from_data_dict(dict["gt_instances"])
        if "gt_visual_relations" in dict:
            res.gt_visual_relations = VisualRelations.from_data_dict(
                dict["gt_visual_relations"]
            )
        if "gt_visual_relations_full" in dict:
            res.gt_visual_relations_full = VisualRelations.from_data_dict(
                dict["gt_visual_relations_full"]
            )

        if "d2_instances" in dict:
            res.d2_instances = instance_from_data_dict(dict["d2_instances"])
        if "d2_feature_pyramid" in dict is not None:
            res.d2_feature_pyramid = dict["d2_feature_pyramid"]
        if "d2_visual_relations_full" in dict:
            res.d2_visual_relations_full = VisualRelations.from_data_dict(
                dict["d2_visual_relations_full"]
            )

    @staticmethod
    def load_image(filename, img_dir):
        import cv2

        image = cv2.imread(img_dir.expanduser().joinpath(filename).resolve().as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def show_instances(self, img_dir: Path, objects: Vocabulary):
        import matplotlib.pyplot as plt

        old_backend = plt.get_backend()
        plt.switch_backend("TkAgg")

        image = self.load_image(self.filename, img_dir)

        instances = {}
        if self.gt_instances is not None:
            instances["GT"] = self.gt_instances
        if self.d2_instances is not None:
            instances["D2"] = self.d2_instances

        for name, inst in instances.items():
            for i in range(len(inst)):
                x0, y0, x1, y1 = inst.boxes.tensor[i].cpu().numpy()
                try:
                    score = inst.scores[i]
                except AttributeError:
                    score = 1
                fig, ax = plt.subplots(
                    1,
                    1,
                    figsize=(
                        self.img_size.width / 640 * 12,
                        self.img_size.height / 640 * 12,
                    ),
                )
                ax.imshow(image)
                ax.scatter(
                    x0,
                    y0,
                    marker="o",
                    c="r",
                    zorder=1000,
                    label=f"TL ({x0:.0f}, {y0:.0f})",
                )
                ax.scatter(
                    x1,
                    y1,
                    marker="D",
                    c="r",
                    zorder=1000,
                    label=f"BR ({x1:.0f}, {y1:.0f})",
                )
                ax.set_title(
                    f"{self.filename} {self.img_size} - "
                    f"{objects.get_str(inst.classes[i])} ({name}: {score:.1%})"
                )
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        width=x1 - x0,
                        height=y1 - y0,
                        fill=False,
                        linewidth=3,
                        color="blue",
                    )
                )
                ax.legend()
                ax.set_xlim([0, self.img_size.width])
                ax.set_ylim([self.img_size.height, 0])
                fig.tight_layout()
                plt.show()
                plt.close(fig)

        plt.switch_backend(old_backend)
