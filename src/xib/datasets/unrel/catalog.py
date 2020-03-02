from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Union

import scipy.io
from PIL import Image, UnidentifiedImageError
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from loguru import logger

from .metadata import OBJECTS, PREDICATES
from ..common import get_exif_orientation


def get_relationship_detection_dicts(root: Path) -> List[Dict[str, Any]]:
    samples = []

    for d in scipy.io.loadmat(root.joinpath("annotations.mat").as_posix())[
        "annotations"
    ]:
        d = d.squeeze(0).item()
        filename = d["filename"].item().item()
        image_id = d["im_id"].item().item()

        img_path = root / "images" / filename
        try:
            with Image.open(img_path.as_posix()) as img:
                width, height = img.size
                exif_orientation = get_exif_orientation(img)
        except (FileNotFoundError, UnidentifiedImageError):
            logger.warning(f"Image not found/invalid {img_path}")
            continue

        if exif_orientation is not None:
            logger.warning(
                f"Image {img_path}"
                f"has an EXIF orientation tag, "
                f"check the corresponding boxes!"
            )
            continue

        objects = {}
        for o in d["objects"].item():
            o = o.squeeze(0).item()
            class_str = o["category"].item().item()
            class_id = OBJECTS.get_id(class_str.replace(" ", "_"))
            # xmin,ymin,xmax,ymax
            box = tuple(o["box"].squeeze(0).item().squeeze(0))

            objects[(class_str, box)] = {
                "category_id": class_id,
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,
                "box_idx": len(objects),
            }

        relations = []
        for r in d["relationships"].item():
            r = r.squeeze(0).item()
            subj_class_str = r["sub"].item().item()
            subj_box = tuple(r["sub_box"].squeeze(0).item().squeeze(0))
            subj_idx = objects[(subj_class_str, subj_box)]["box_idx"]

            obj_class_str = r["obj"].item().item()
            obj_box = tuple(r["obj_box"].squeeze(0).item().squeeze(0))
            obj_idx = objects[(obj_class_str, obj_box)]["box_idx"]

            for c in r["rels"].item():
                c = c.item().item().replace(" ", "_")
                relations.append(
                    {
                        "category_id": PREDICATES.get_id(c),
                        "subject_idx": subj_idx,
                        "object_idx": obj_idx,
                    }
                )

        if len(relations) == 0:
            logger.warning(f"Image {filename} has 0 annotated relations!")

        samples.append(
            {
                "file_name": img_path.as_posix(),
                "image_id": image_id,
                "width": width,
                "height": height,
                "annotations": sorted(objects.values(), key=itemgetter("box_idx")),
                "relations": relations,
            }
        )

    return samples


def register_unrel(data_root: Union[str, Path]):
    data_root = Path(data_root).expanduser().resolve()
    raw = data_root / "unrel" / "raw"
    processed = data_root / "unrel" / "processed"

    metadata_common = dict(
        thing_classes=OBJECTS.words,
        predicate_classes=PREDICATES.words,
        object_linear_features=3 + len(OBJECTS.words),
        edge_linear_features=10,
        raw=raw,
        processed=processed,
        prob_s_o_given_p=processed / "prob_s_o_given_p.pkl.xz",
        prob_s_p_o=processed / "prob_s_p_o.pkl.xz",
    )

    MetadataCatalog.get("unrel_relationship_detection").set(
        splits=("test",), **metadata_common
    )

    DatasetCatalog.register(
        "unrel_object_detection_test", partial(get_relationship_detection_dicts, raw)
    )
    DatasetCatalog.register(
        "unrel_relationship_detection_test",
        partial(get_relationship_detection_dicts, raw),
    )
    MetadataCatalog.get("unrel_object_detection_test").set(
        thing_classes=OBJECTS.words, image_root=raw / "images", evaluator_type="coco"
    )
    MetadataCatalog.get("unrel_relationship_detection_test").set(
        image_root=raw / "images", graph_root=processed / "test", **metadata_common
    )


def register_unrel_vrd(data_root: Union[str, Path]):
    data_root = Path(data_root).expanduser().resolve()
    raw = data_root / "unrel_vrd" / "raw"
    processed = data_root / "unrel_vrd" / "processed"

    metadata_common = dict(
        thing_classes=OBJECTS.words,
        predicate_classes=PREDICATES.words,
        object_linear_features=3 + len(OBJECTS.words),
        edge_linear_features=10,
        raw=raw,
        processed=processed,
        prob_s_o_given_p=processed / "prob_s_o_given_p.pkl.xz",
        prob_s_p_o=processed / "prob_s_p_o.pkl.xz",
    )

    MetadataCatalog.get("unrel_vrd_relationship_detection").set(
        splits=("test",), **metadata_common
    )

    MetadataCatalog.get("unrel_vrd_relationship_detection_test").set(
        image_root=raw / "images", graph_root=processed / "test", **metadata_common
    )
