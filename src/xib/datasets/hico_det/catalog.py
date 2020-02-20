from pathlib import Path
from typing import Any, Dict, List

from PIL import UnidentifiedImageError
from detectron2.structures import BoxMode
from loguru import logger

from xib.datasets.common import img_size_with_exif
from .matlab_reader import HicoDetMatlabLoader


def get_object_detection_dicts(root: Path, split: str) -> List[Dict[str, Any]]:
    ml = HicoDetMatlabLoader(root / "anno_bbox.mat")
    split = HicoDetMatlabLoader.Split[split.upper()]

    samples = []
    for hds in ml.iter_vr_samples(split, nms_threshold=0.7):
        img_path = root / split.image_dir / hds.filename

        try:
            img_size, exif_orientation = img_size_with_exif(img_path)
        except (FileNotFoundError, UnidentifiedImageError):
            logger.warning(
                f"{split.name.capitalize()} image not found or invalid: {img_path}"
            )
            continue

        if exif_orientation is not None:
            logger.warning(
                f"Skipping {split.name} image with "
                f"EXIF orientation {exif_orientation}, "
                f"check the corresponding boxes: {img_path}"
            )
            continue

        if hds.img_size != img_size:
            logger.warning(
                f"{split.name.capitalize()} image size mismatch, "
                f"{hds.img_size} from the matlab file, "
                f"{img_size} from the image: "
                f"{img_path}"
            )
            continue

        sample = {
            "file_name": img_path.as_posix(),
            "image_id": len(samples),
            "width": hds.img_size.width,
            "height": hds.img_size.height,
            "annotations": [
                {
                    "category_id": obj_class.item(),
                    "bbox": obj_box.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                }
                for obj_box, obj_class in zip(
                    hds.gt_instances.boxes, hds.gt_instances.classes
                )
            ],
        }

        samples.append(sample)

    return samples


def get_relationship_detection_dicts(root: Path, split: str) -> List[Dict[str, Any]]:
    raise NotImplementedError()


def register_hico(root):
    from .metadata import OBJECTS, PREDICATES
    from functools import partial
    from detectron2.data import DatasetCatalog, MetadataCatalog

    root = Path(root).expanduser().resolve()

    for split in ["train", "test"]:
        DatasetCatalog.register(
            f"hico_object_detection_{split}",
            partial(get_object_detection_dicts, root, split),
        )
        DatasetCatalog.register(
            f"hico_relationship_detection_{split}",
            partial(get_relationship_detection_dicts, root, split),
        )
        MetadataCatalog.get(f"hico_object_detection_{split}").set(
            thing_classes=OBJECTS.words,
            image_root=root.as_posix(),
            evaluator_type="coco",
        )
        MetadataCatalog.get(f"hico_relationship_detection_{split}").set(
            thing_classes=OBJECTS.words,
            predicate_classes=PREDICATES.words,
            image_root=root.as_posix(),
        )