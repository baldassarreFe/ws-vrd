import dataclasses

from pathlib import Path
from typing import Union, Optional, Dict

import torch

from detectron2.structures import Instances


from ...structures import ImageSize, VisualRelations


@dataclasses.dataclass
class HicoDetSample(object):
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

    @staticmethod
    def load_image(filename, img_dir):
        import cv2
        image = cv2.imread(img_dir.expanduser().joinpath(filename).resolve().as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def show_instances(self, img_dir: Path):
        import matplotlib.pyplot as plt
        old_backend = plt.get_backend()
        plt.switch_backend('TkAgg')

        from .metadata import OBJECTS

        image = self.load_image(self.filename, img_dir)

        instances = {}
        if self.gt_instances is not None:
            instances['GT'] = self.gt_instances
        if self.d2_instances is not None:
            instances['D2'] = self.d2_instances

        for name, inst in instances.items():
            for i in range(len(inst)):
                x0, y0, x1, y1 = inst.boxes.tensor[i].cpu().numpy()
                try:
                    score = inst.scores[i]
                except AttributeError:
                    score = 1
                fig, ax = plt.subplots(1, 1, figsize=(self.img_size.width / 640 * 12, self.img_size.height / 640 * 12))
                ax.imshow(image)
                ax.scatter(x0, y0, marker='o', c='r', zorder=1000, label=f'TL ({x0:.0f}, {y0:.0f})')
                ax.scatter(x1, y1, marker='D', c='r', zorder=1000, label=f'BR ({x1:.0f}, {y1:.0f})')
                ax.set_title(f'{self.filename} {self.img_size} - '
                             f'{OBJECTS.get_str(inst.classes[i])} ({name}: {score:.1%})')
                ax.add_patch(plt.Rectangle(
                    (x0, y0),
                    width=x1 - x0,
                    height=y1 - y0,
                    fill=False,
                    linewidth=3,
                    color='blue'
                ))
                ax.legend()
                ax.set_xlim([0, self.img_size.width])
                ax.set_ylim([self.img_size.height, 0])
                fig.tight_layout()
                plt.show()
                plt.close(fig)

        plt.switch_backend(old_backend)
