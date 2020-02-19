import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, NewType, Set

from PIL import Image, UnidentifiedImageError
from detectron2.structures import BoxMode
from loguru import logger

from ..common import img_size_with_exif, get_exif_orientation

Box = NewType("Box", Tuple[float, float, float, float])


def y1y2x1x2_to_x1y1x2y2(y1y2x1x2: Box) -> Box:
    y1, y2, x1, x2 = y1y2x1x2
    return x1, y1, x2, y2


def get_object_detection_dicts(root: Path, split: str) -> List[Dict[str, Any]]:
    with open(root / f"annotations_{split}.json") as f:
        annotations = json.load(f)

    samples = []
    for filename, relations in annotations.items():
        img_path = root / f"sg_{split}_images" / filename
        try:
            img_size, exif_orientation = img_size_with_exif(img_path)
        except (FileNotFoundError, UnidentifiedImageError):
            logger.warning(
                f"{split.capitalize()} image not found or invalid: {img_path}"
            )
            continue

        if exif_orientation is not None:
            logger.warning(
                f"Skipping {split} image with "
                f"EXIF orientation {exif_orientation}, "
                f"check the corresponding boxes: {img_path}"
            )
            continue

        sample = {
            "file_name": img_path.as_posix(),
            "image_id": len(samples),
            "width": img_size.width,
            "height": img_size.height,
        }

        # Use a set to filter duplicated boxes
        instances: Set[Tuple[int, Box]] = set()
        for r in relations:
            for so in ("subject", "object"):
                instances.add((r[so]["category"], y1y2x1x2_to_x1y1x2y2(r[so]["bbox"])))

        sample["annotations"] = [
            {
                "category_id": instance[0],
                "bbox": instance[1],
                "bbox_mode": BoxMode.XYXY_ABS,
            }
            for instance in instances
        ]

        samples.append(sample)

    return samples


def get_relationship_detection_dicts(root: Path, split: str) -> List[Dict[str, Any]]:
    with open(root / f"annotations_{split}.json") as f:
        annotations = json.load(f)

    samples = []
    for filename, relations in annotations.items():
        img_path = root / f"sg_{split}_images" / filename
        try:
            with Image.open(img_path.as_posix()) as img:
                width, height = img.size
                exif_orientation = get_exif_orientation(img)
        except (FileNotFoundError, UnidentifiedImageError):
            logger.warning(f"{split.capitalize()} image not found/invalid {img_path}")
            continue

        if exif_orientation is not None:
            logger.warning(
                f"{split.capitalize()} image {img_path}"
                f"has an EXIF orientation tag, "
                f"check the corresponding boxes!"
            )
            continue

        sample = {
            "file_name": img_path.as_posix(),
            "image_id": len(samples),
            "width": width,
            "height": height,
        }

        # Use a dict to filter duplicated boxes
        annos: Dict[Tuple[int, Box], Dict] = {}
        rels: List[Dict[str, int]] = []
        for r in relations:
            subject_class = r["subject"]["category"]
            subject_box = y1y2x1x2_to_x1y1x2y2(r["subject"]["bbox"])
            subject_dict = annos.setdefault(
                (subject_class, subject_box),
                {
                    "category_id": subject_class,
                    "bbox": subject_box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "box_idx": len(annotations),
                },
            )

            object_class = r["object"]["category"]
            object_box = y1y2x1x2_to_x1y1x2y2(r["object"]["bbox"])
            object_dict = annos.setdefault(
                (object_class, object_box),
                {
                    "category_id": object_class,
                    "bbox": object_box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "box_idx": len(annotations),
                },
            )

            rels.append(
                {
                    "category_id": r["predicate"],
                    "subject_idx": subject_dict["box_idx"],
                    "object_idx": object_dict["box_idx"],
                }
            )

        sample["annotations"] = list(annos.values())
        sample["relations"] = rels
        samples.append(sample)

    return samples


def register_vrd(root):
    from xib.datasets.vrd.metadata import OBJECTS, PREDICATES
    from functools import partial
    from detectron2.data import DatasetCatalog, MetadataCatalog

    root = Path(root).expanduser().resolve()

    for split in ["train", "test"]:
        DatasetCatalog.register(
            f"vrd_object_detection_{split}",
            partial(get_object_detection_dicts, root, split),
        )
        DatasetCatalog.register(
            f"vrd_relationship_detection_{split}",
            partial(get_relationship_detection_dicts, root, split),
        )
        MetadataCatalog.get(f"vrd_object_detection_{split}").set(
            thing_classes=OBJECTS.words,
            image_root=root.as_posix(),
            evaluator_type="coco",
        )
        MetadataCatalog.get(f"vrd_relationship_detection_{split}").set(
            thing_classes=OBJECTS.words,
            predicate_classes=PREDICATES.words,
            image_root=root.as_posix(),
        )
