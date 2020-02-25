from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Union, Iterator, Tuple, Mapping

import scipy.io
import torch
from detectron2.structures import Boxes, Instances, pairwise_iou
from loguru import logger

from .metadata import OBJECTS, PREDICATES
from ..sample import VrSample
from xib.structures import ImageSize, VisualRelations


class HicoDetMatlabLoader(object):
    """Helper class to load HICO-Det annotations from the provided matlab file."""

    _path: Path
    _matlab_dict: Optional[dict] = None
    _interaction_triplets: List[Dict[str, str]] = None

    class Split(Enum):
        TRAIN = "bbox_train", "images/train2015"
        TEST = "bbox_test", "images/test2015"

        def __init__(self, matlab_name, image_dir):
            self._matlab_name = matlab_name
            self._image_dir = image_dir

        @property
        def matlab_name(self):
            return self._matlab_name

        def __str__(self):
            return self.name.lower()

        @property
        def image_dir(self):
            return self._image_dir

    def __init__(self, matlab_path: Union[str, Path]):
        self._path = Path(matlab_path).expanduser().resolve()

    @property
    def matlab_dict(self):
        if self._matlab_dict is None:
            self._matlab_dict = scipy.io.loadmat(self._path.as_posix())
        return self._matlab_dict

    @property
    def interaction_triplets(self):
        """Interaction triplets (subject, predicate, object)

        The matlab file defines 600 unique (subject, predicate, object) triplets where subject is always "person".
        For every image, the matlab file contains a list of interaction ids that can be looked up in this list.
        """
        if self._interaction_triplets is None:
            self._interaction_triplets = []
            for interaction in self.matlab_dict["list_action"].squeeze():
                self.interaction_triplets.append(
                    {
                        "subject": "person",
                        "predicate": interaction["vname"].item().replace(" ", "_"),
                        "object": interaction["nname"].item().replace(" ", "_"),
                    }
                )
        return self._interaction_triplets

    def iter_hico_samples(self, split: Split) -> Iterator[dict]:
        """Iterate over samples from the split passed as parameter"""
        for md in self.matlab_dict[split.matlab_name].squeeze(0):
            yield HicoDetMatlabLoader._parse_hico(md)

    def iter_vr_samples(self, split: Split, nms_threshold) -> Iterator[VrSample]:
        """Iterate over samples from the split passed as parameter"""
        for hico_dict in self.iter_hico_samples(split):
            yield self._parse_vr(hico_dict, nms_threshold)

    @staticmethod
    def _parse_hico(matlab_dict) -> Dict:
        """Parses one HICO-DET sample from the corresponding matlab dict using the default HICO structure."""

        filename = matlab_dict["filename"].item()
        size = ImageSize(
            # img['size']['depth'].item().item(),
            matlab_dict["size"]["height"].item().item(),
            matlab_dict["size"]["width"].item().item(),
        )
        interactions = []

        # All interaction types present in this image
        for interaction in matlab_dict["hoi"].squeeze(0):
            interaction_id = interaction["id"].item() - 1

            bb_subjects: List[Tuple[int, int, int, int]] = []
            bb_objects: List[Tuple[int, int, int, int]] = []
            connections: List[Tuple[int, int]] = []

            # Invisible interaction, no humans or objects visible
            visible = interaction["invis"].item() == 0

            if visible:
                # All subject boxes for this interaction
                bb_subjects = [
                    (
                        human["x1"].item(),
                        human["y1"].item(),
                        human["x2"].item(),
                        human["y2"].item(),
                    )
                    for human in interaction["bboxhuman"].squeeze(0)
                ]

                # All object boxes for this interaction
                bb_objects = [
                    (
                        object["x1"].item(),
                        object["y1"].item(),
                        object["x2"].item(),
                        object["y2"].item(),
                    )
                    for object in interaction["bboxobject"].squeeze(0)
                ]

                # All instances of this interaction type
                connections: List[Tuple[int, int]] = []
                for subject_box_id, object_box_id in interaction["connection"]:
                    connections.append((subject_box_id - 1, object_box_id - 1))

            interactions.append(
                {
                    "interaction_id": interaction_id,
                    "visible": visible,
                    "bb_subjects": bb_subjects,
                    "bb_objects": bb_objects,
                    "connections": connections,
                }
            )

        hico_dict = {"filename": filename, "size": size, "interactions": interactions}

        return hico_dict

    def _parse_vr(self, hico_dict: Mapping, nms_threshold: float = 0.7) -> VrSample:
        """Parse one VrSample from the corresponding hico_dict using the visual relationship structure

        Also merge duplicate instances, i.e. boxes with the same object category
        that overlap with IoU > nms_threshold
        """

        # NOTE: interesting images to debug:
        # hico_dict['filename'] in {
        #   'HICO_train2015_00000001.jpg',
        #   'HICO_train2015_00000061.jpg', # 2 people, 1 motorbike, many actions
        #   'HICO_train2015_00000014.jpg',
        #   'HICO_train2015_00000009.jpg'
        # }

        subject_boxes = []
        subject_classes = []

        object_boxes = []
        object_classes = []

        subject_indexes = []
        object_indexes = []

        predicate_classes = []

        for interaction in hico_dict["interactions"]:
            interaction_triplet = self.interaction_triplets[
                interaction["interaction_id"]
            ]

            # Invisible interaction, no humans or objects visible
            if not interaction["visible"]:
                logger.debug(
                    f"Skipping invisible interaction ("
                    f'{interaction_triplet["subject"]}, '
                    f'{interaction_triplet["predicate"]}, '
                    f'{interaction_triplet["object"]}) '
                    f'in {hico_dict["filename"]}'
                )
                continue

            subj_offset = len(subject_boxes)
            obj_offset = len(object_boxes)
            for subj_idx, obj_idx in interaction["connections"]:
                subject_indexes.append(subj_idx + subj_offset)
                object_indexes.append(obj_idx + obj_offset)

            subject_boxes.extend(interaction["bb_subjects"])
            object_boxes.extend(interaction["bb_objects"])

            subject_classes.extend(
                [interaction_triplet["subject"]] * len(interaction["bb_subjects"])
            )
            predicate_classes.extend(
                [interaction_triplet["predicate"]] * len(interaction["connections"])
            )
            object_classes.extend(
                [interaction_triplet["object"]] * len(interaction["bb_objects"])
            )

        # Concatenate subject and object instances into a single list of objects
        boxes = Boxes(torch.tensor(subject_boxes + object_boxes))
        classes = torch.tensor(
            OBJECTS.get_id(subject_classes + object_classes).values, dtype=torch.long
        )

        # Stack relationship indexes into a 2xM tensor (possibly 2x0),
        # also offset all object indexes since now they appear after all subjects
        relation_indexes = torch.tensor(
            [subject_indexes, object_indexes], dtype=torch.long
        )
        relation_indexes[1, :] += len(subject_boxes)

        # Merge boxes that overlap and have the same label
        boxes, classes, relation_indexes = HicoDetMatlabLoader._merge_overlapping(
            boxes=boxes,
            classes=classes,
            relation_indexes=relation_indexes,
            nms_threshold=nms_threshold,
        )

        gt_instances = Instances(hico_dict["size"], classes=classes, boxes=boxes)

        gt_visual_relations = VisualRelations(
            instances=gt_instances,
            predicate_classes=torch.tensor(
                PREDICATES.get_id(predicate_classes).values, dtype=torch.long
            ),
            relation_indexes=relation_indexes,
        )

        return VrSample(
            filename=hico_dict["filename"],
            img_size=hico_dict["size"],
            gt_instances=gt_instances,
            gt_visual_relations=gt_visual_relations,
        )

    @staticmethod
    def _merge_overlapping(
        boxes: Boxes,
        classes: torch.LongTensor,
        relation_indexes: torch.LongTensor,
        nms_threshold: float,
    ):
        # Boxes are candidate for merging if their IoU is above a threshold
        iou_above_thres = pairwise_iou(boxes, boxes) > nms_threshold

        # Also, they have to belong to the same class to be candidates.
        # Here we treat "person subj" and "person obj" as two
        # separate classes, to avoid merging cases of "person hugs person"
        # where the two people have high overlap but must remain separate
        obj_idx = relation_indexes[1]
        obj_is_person = classes[obj_idx] == 0
        classes_tmp = classes.clone()
        classes_tmp[obj_idx[obj_is_person]] = -1
        same_class = classes_tmp[:, None] == classes_tmp[None, :]

        candidates = iou_above_thres & same_class

        keep = []
        visited = torch.full((len(boxes),), False, dtype=torch.bool)
        relation_indexes = relation_indexes.clone()

        for old_box_idx, skip in enumerate(visited):
            if skip:
                continue
            new_box_idx = len(keep)
            keep.append(old_box_idx)

            matches = torch.nonzero(
                candidates[old_box_idx, :] & ~visited, as_tuple=True
            )[0]
            visited[matches] = True

            rel_idx_to_fix = torch.any(
                relation_indexes[:, :, None] == matches[None, None, :], dim=2
            )
            relation_indexes[rel_idx_to_fix] = new_box_idx

        return boxes[keep], classes[keep], relation_indexes
