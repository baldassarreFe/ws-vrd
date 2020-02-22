import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, NewType, Set, Mapping, Union

import torch
from PIL import Image, UnidentifiedImageError
from detectron2.structures import BoxMode, Boxes, Instances
from loguru import logger

from xib.structures import ImageSize, VisualRelations
from ..common import img_size_with_exif, get_exif_orientation
from ..sample import VrSample

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
                    "box_idx": len(annos),
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
                    "box_idx": len(annos),
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


def data_dict_to_vr_sample(data_dict: Mapping) -> VrSample:
    sample = VrSample(
        filename=Path(data_dict["file_name"]).name,
        img_size=ImageSize(data_dict["height"], data_dict["width"]),
    )

    # There are images for which the json file contains an empty list of relations.
    # Since boxes are implicitly given from that field, this results in an
    # image with 0 boxes and 0 relations. We still parse it here, but it should
    # be discarded later.
    boxes = []
    classes = []
    if len(data_dict['annotations']) > 0:
        boxes, classes = zip(*(
            (a["bbox"], a["category_id"]) for a in data_dict["annotations"]
        ))
    boxes = Boxes(torch.tensor(boxes, dtype=torch.float))
    classes = torch.tensor(classes, dtype=torch.long)
    sample.gt_instances = Instances(sample.img_size, boxes=boxes, classes=classes)

    relation_indexes = []
    predicate_classes = []
    if len(data_dict["relations"]) > 0:
        relation_indexes, predicate_classes = zip(*(
            ((r["subject_idx"], r["object_idx"]), r["category_id"])
            for r in data_dict["relations"]
        ))

    relation_indexes = (
        torch.tensor(relation_indexes, dtype=torch.long).view(-1, 2).transpose(0, 1)
    )
    predicate_classes = torch.tensor(predicate_classes, dtype=torch.long)

    sample.gt_visual_relations = VisualRelations(
        relation_indexes=relation_indexes,
        predicate_classes=predicate_classes,
        instances=sample.gt_instances,
    )
    return sample


def register_vrd(data_root: Union[str, Path]):
    from xib.datasets.vrd.metadata import OBJECTS, PREDICATES
    from functools import partial
    from detectron2.data import DatasetCatalog, MetadataCatalog

    data_root = Path(data_root).expanduser().resolve()
    raw = Path(data_root).expanduser().resolve() / 'vrd' / 'raw'

    for split in ["train", "test"]:
        DatasetCatalog.register(
            f"vrd_object_detection_{split}",
            partial(get_object_detection_dicts, raw, split),
        )
        DatasetCatalog.register(
            f"vrd_relationship_detection_{split}",
            partial(get_relationship_detection_dicts, raw, split),
        )
        MetadataCatalog.get(f"vrd_object_detection_{split}").set(
            thing_classes=OBJECTS.words,
            image_root=f'vrd/raw/sg_{split}_images',
            evaluator_type="coco",
        )
        MetadataCatalog.get(f"vrd_relationship_detection_{split}").set(
            thing_classes=OBJECTS.words,
            predicate_classes=PREDICATES.words,
            object_linear_features=3 + len(OBJECTS.words),
            edge_linear_features=10,
            graph_root=f'vrd/processed/{split}',
            image_root=f'vrd/raw/sg_{split}_images',
        )
