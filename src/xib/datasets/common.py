from pathlib import Path
from typing import Union, Mapping

import torch
from PIL import Image
from detectron2.structures import Boxes, Instances

from xib.datasets.sample import VrSample
from xib.structures import ImageSize, VisualRelations


def get_exif_orientation(img: Image) -> Union[int, None]:
    orientation = img.getexif().get(274)
    return {
        2: "FLIP_LEFT_RIGHT",
        3: "ROTATE_180",
        4: "FLIP_TOP_BOTTOM",
        5: "TRANSPOSE",
        6: "ROTATE_270",
        7: "TRANSVERSE",
        8: "ROTATE_90",
    }.get(orientation)


def img_size_with_exif(img_path: Path):
    with Image.open(img_path.as_posix()) as img:
        width, height = img.size
        exif_orientation = get_exif_orientation(img)
    return ImageSize(height, width), exif_orientation


def data_dict_to_vr_sample(data_dict: Mapping) -> VrSample:
    sample = VrSample(
        filename=Path(data_dict["file_name"]).name,
        img_size=ImageSize(data_dict["height"], data_dict["width"]),
    )

    # There are images for which the annotations file contains an empty list of relations.
    # Since boxes are implicitly given from that field, this results in an
    # image with 0 boxes and 0 relations. We still parse it here, but it should
    # be discarded later.
    boxes = []
    classes = []
    if len(data_dict["annotations"]) > 0:
        boxes, classes = zip(
            *((a["bbox"], a["category_id"]) for a in data_dict["annotations"])
        )
    boxes = Boxes(torch.tensor(boxes, dtype=torch.float))
    classes = torch.tensor(classes, dtype=torch.long)
    sample.gt_instances = Instances(sample.img_size, boxes=boxes, classes=classes)

    relation_indexes = []
    predicate_classes = []
    if len(data_dict["relations"]) > 0:
        relation_indexes, predicate_classes = zip(
            *(
                ((r["subject_idx"], r["object_idx"]), r["category_id"])
                for r in data_dict["relations"]
            )
        )

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
