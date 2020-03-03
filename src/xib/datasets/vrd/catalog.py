import json
from collections import defaultdict
from functools import partial, lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, NewType, Set, Union, Iterator

from PIL import Image, UnidentifiedImageError
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from loguru import logger

from .metadata import OBJECTS, PREDICATES
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

        # There are images for which the json file contains an empty list of relations.
        # Since boxes are implicitly given from that field, this results in an
        # image with 0 boxes and 0 relations. We still parse it here, but it should
        # be discarded later.
        if len(relations) == 0:
            logger.warning(
                f"{split.capitalize()} image {filename} has 0 annotated relations!"
            )

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


@lru_cache(maxsize=1)
def get_zero_shot_triplets() -> Set[Tuple[int, int, int]]:
    """

    Returns: a set of (subject_class, predicate_class, object_class) triplets

    """
    # This actually counts the occurrences of all triplets,
    # it's useless now but it's super fast so it doesn't hurt
    train_data_dicts = DatasetCatalog.get("vrd_relationship_detection_train")
    train_spo = defaultdict(lambda: 0)
    for d in train_data_dicts:
        for r in d["relations"]:
            spo = (
                d["annotations"][r["subject_idx"]]["category_id"],
                r["category_id"],
                d["annotations"][r["object_idx"]]["category_id"],
            )
            train_spo[spo] += 1

    test_data_dicts = DatasetCatalog.get("vrd_relationship_detection_test")
    test_spo = defaultdict(lambda: 0)
    for d in test_data_dicts:
        for r in d["relations"]:
            spo = (
                d["annotations"][r["subject_idx"]]["category_id"],
                r["category_id"],
                d["annotations"][r["object_idx"]]["category_id"],
            )
            test_spo[spo] += 1

    zero_shot = set.difference(set(test_spo.keys()), set(train_spo.keys()))
    assert len(zero_shot) == 1025

    return zero_shot


def get_zero_shot_data_dicts() -> Iterator[Dict]:
    def rel_to_spo(r, d):
        return (
            d["annotations"][r["subject_idx"]]["category_id"],
            r["category_id"],
            d["annotations"][r["object_idx"]]["category_id"],
        )

    # List all triplets in the training set
    train_data_dicts = DatasetCatalog.get("vrd_relationship_detection_train")
    train_triplets = set()
    for d in train_data_dicts:
        for r in d["relations"]:
            spo = rel_to_spo(r, d)
            train_triplets.add(spo)

    # List all triplets in the test set
    test_data_dicts = DatasetCatalog.get("vrd_relationship_detection_test")
    test_triplets = set()
    for d in test_data_dicts:
        for r in d["relations"]:
            spo = rel_to_spo(r, d)
            test_triplets.add(spo)

    # Find zero shot triplets: present in test set but not in train set
    zero_shot_triplets = set.difference(test_triplets, train_triplets)

    # Filter out test images that do not contain zero-shot triplets
    for d in test_data_dicts:
        if any(rel_to_spo(r, d) in zero_shot_triplets for r in d["relations"]):
            yield d


def register_vrd(data_root: Union[str, Path]):
    data_root = Path(data_root).expanduser().resolve()
    raw = data_root / "vrd" / "raw"
    processed = data_root / "vrd" / "processed"

    metadata_common = dict(
        thing_classes=OBJECTS.words,
        predicate_classes=PREDICATES.words,
        object_linear_features=3 + len(OBJECTS.words),
        edge_linear_features=10,
        raw=raw,
        processed=processed,
        prob_s_o_given_p=processed / "prob_s_o_given_p.pkl.xz",
        prob_s_o_given_p_test=processed / "prob_s_o_given_p_test.pkl.xz",
        prob_s_p_o=processed / "prob_s_p_o.pkl.xz",
    )

    MetadataCatalog.get(f"vrd_relationship_detection").set(
        splits=("train", "test"), **metadata_common
    )

    for split in MetadataCatalog.get(f"vrd_relationship_detection").splits:
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
            image_root=raw / f"sg_{split}_images",
            evaluator_type="coco",
        )
        MetadataCatalog.get(f"vrd_relationship_detection_{split}").set(
            image_root=raw / f"sg_{split}_images",
            graph_root=processed / split,
            **metadata_common,
        )


def register_vrd_zero_shot(data_root: Union[str, Path]):
    data_root = Path(data_root).expanduser().resolve()
    raw = data_root / "vrd_zero_shot" / "raw"
    processed = data_root / "vrd_zero_shot" / "processed"

    metadata_common = dict(
        thing_classes=OBJECTS.words,
        predicate_classes=PREDICATES.words,
        object_linear_features=3 + len(OBJECTS.words),
        edge_linear_features=10,
        raw=raw,
        processed=processed,
        prob_s_o_given_p=processed / "prob_s_o_given_p.pkl.xz",
        prob_s_o_given_p_test=processed / "prob_s_o_given_p_test.pkl.xz",
        prob_s_p_o=processed / "prob_s_p_o.pkl.xz",
    )

    DatasetCatalog.register(
        "vrd_relationship_detection_zero_shot", get_zero_shot_data_dicts
    )
    MetadataCatalog.get("vrd_relationship_detection_zero_shot").set(
        splits=("test",),
        image_root=raw / "sg_test_images",
        graph_root=processed / "test",
        **metadata_common,
    )
