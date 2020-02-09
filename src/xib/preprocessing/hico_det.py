import time
import argparse
from enum import Enum

from typing import Union, Iterator, Optional, List, Dict, Tuple, Mapping
from pathlib import Path

import torch
import scipy.io
from detectron2.structures import Boxes, Instances

from loguru import logger

from ..utils import SigIntCatcher
from ..structures import VisualRelations, ImageSize
from ..detectron2 import DetectronWrapper, boxes_to_edge_features
from ..datasets.hico_det import HicoDet, HicoDetSample


def parse_args():
    def resolve_path(path: str):
        return Path(path).expanduser().resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument('--hico-dir', required=True, type=resolve_path)
    parser.add_argument('--output-dir', required=True, type=resolve_path)
    parser.add_argument('--skip-existing', action='store_true')

    parser.add_argument('--nms-threshold', required=True, type=float, default=.7)
    parser.add_argument('--confidence-threshold', required=True, type=float, default=.3)
    parser.add_argument('--detectron-home', required=True, type=resolve_path, default='~/detectron2')

    return parser.parse_args()


class HicoDetMatlabLoader(object):
    """Helper class to load HICO-Det annotations from the provided matlab file."""
    _path: Path
    _matlab_dict: Optional[dict] = None
    _interaction_triplets: List[Dict[str, str]] = None

    class Split(Enum):
        TRAIN = 'bbox_train', 'images/train2015'
        TEST = 'bbox_test', 'images/test2015'

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
            for interaction in self.matlab_dict['list_action'].squeeze():
                self.interaction_triplets.append({
                    'subject': 'person',
                    'predicate': interaction['vname'].item().replace(' ', '_'),
                    'object': interaction['nname'].item().replace(' ', '_')
                })
        return self._interaction_triplets

    def iter_hico_samples(self, split: Split) -> Iterator[dict]:
        """Iterate over samples from the split passed as parameter"""
        for md in self.matlab_dict[split.matlab_name].squeeze(0):
            yield HicoDetMatlabLoader._parse_hico(md)

    def iter_vr_samples(self, split: Split, nms_threshold) -> Iterator[HicoDetSample]:
        """Iterate over samples from the split passed as parameter"""
        for hico_dict in self.iter_hico_samples(split):
            yield self._parse_vr(hico_dict, nms_threshold)

    @staticmethod
    def _parse_hico(matlab_dict) -> Dict:
        """Parses one HICO-DET sample from the corresponding matlab dict using the default HICO structure."""

        filename = matlab_dict['filename'].item()
        size = ImageSize(
            # img['size']['depth'].item().item(),
            matlab_dict['size']['height'].item().item(),
            matlab_dict['size']['width'].item().item(),
        )
        interactions = []

        # All interaction types present in this image
        for interaction in matlab_dict['hoi'].squeeze(0):
            interaction_id = interaction['id'].item() - 1

            bb_subjects: List[Tuple[int, int, int, int]] = []
            bb_objects: List[Tuple[int, int, int, int]] = []
            connections: List[Tuple[int, int]] = []

            # Invisible interaction, no humans or objects visible
            visible = interaction['invis'].item() == 0

            if visible:
                # All subject boxes for this interaction
                bb_subjects = [
                    (human['x1'].item(), human['y1'].item(), human['x2'].item(), human['y2'].item())
                    for human in interaction['bboxhuman'].squeeze(0)
                ]

                # All object boxes for this interaction
                bb_objects = [
                    (object['x1'].item(), object['y1'].item(), object['x2'].item(), object['y2'].item())
                    for object in interaction['bboxobject'].squeeze(0)
                ]

                # All instances of this interaction type
                connections: List[Tuple[int, int]] = []
                for subject_box_id, object_box_id in interaction['connection']:
                    connections.append((subject_box_id - 1, object_box_id - 1))

            interactions.append({
                'interaction_id': interaction_id,
                'visible': visible,
                'bb_subjects': bb_subjects,
                'bb_objects': bb_objects,
                'connections': connections,
            })

        hico_dict = {
            'filename': filename,
            'size': size,
            'interactions': interactions
        }

        return hico_dict

    def _parse_vr(self, hico_dict: Mapping, nms_threshold: float = .7) -> HicoDetSample:
        """Parse one HicoDetSample from the corresponding hico_dict using the visual relationship structure"""

        # NOTE: interesting images to debug:
        # hico_dict['filename'] in {
        #   'HICO_train2015_00000061.jpg',
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

        for interaction in hico_dict['interactions']:
            interaction_triplet = self.interaction_triplets[interaction['interaction_id']]

            # Invisible interaction, no humans or objects visible
            if not interaction['visible']:
                logger.debug(f'Skipping invisible interaction ('
                             f'{interaction_triplet["subject"]}, '
                             f'{interaction_triplet["predicate"]}, '
                             f'{interaction_triplet["object"]}) '
                             f'in {hico_dict["filename"]}')
                continue

            subj_offset = len(subject_boxes)
            obj_offset = len(object_boxes)
            for subj_idx, obj_idx in interaction['connections']:
                subject_indexes.append(subj_idx + subj_offset)
                object_indexes.append(obj_idx + obj_offset)

            subject_boxes.extend(interaction['bb_subjects'])
            object_boxes.extend(interaction['bb_objects'])

            subject_classes.extend([interaction_triplet['subject']] * len(interaction['bb_subjects']))
            predicate_classes.extend([interaction_triplet['predicate']] * len(interaction['connections']))
            object_classes.extend([interaction_triplet['object']] * len(interaction['bb_objects']))

        # Stack relationship indexes into a 2xM tensor (possibly 2x0)
        relation_indexes = torch.tensor([
            subject_indexes,
            object_indexes
        ], dtype=torch.long)

        # TODO merge boxes
        subject_instances = Instances(
            hico_dict['size'],
            classes=torch.tensor(HicoDet.object_name_to_id(subject_classes)),
            boxes=Boxes(torch.tensor(subject_boxes))
        )
        object_instances = Instances(
            hico_dict['size'],
            classes=torch.tensor(HicoDet.object_name_to_id(object_classes)),
            boxes=Boxes(torch.tensor(object_boxes))
        )
        # Concatenate subject and object instances into a single list of objects
        gt_instances = Instances.cat([subject_instances, object_instances])

        # Offset all object indexes since now they appear after all subjects
        relation_indexes[1, :] += len(subject_instances)

        gt_visual_relations = VisualRelations(
            instances=gt_instances,
            predicate_classes=torch.tensor(HicoDet.predicate_name_to_id(predicate_classes)),
            relation_indexes=relation_indexes,
        )

        return HicoDetSample(
            filename=hico_dict['filename'],
            img_size=hico_dict['size'],
            gt_instances=gt_instances,
            gt_visual_relations=gt_visual_relations,
        )


@logger.catch
def main():
    args = parse_args()

    # Check files and folders
    if not args.detectron_home.is_dir():
        raise ValueError(f'Not a directory: {args.detectron_home}')
    if not args.hico_dir.is_dir():
        raise ValueError(f'Not a directory: {args.hico_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build detectron model
    torch.set_grad_enabled(False)
    detectron = DetectronWrapper(detectron_home=args.detectron_home, threshold=args.confidence_threshold)

    # Load ground truth bounding boxes and human-object relations from the matlab file,
    # then process the image with detectron to extract visual features of the boxes.
    should_stop = SigIntCatcher()
    loader = HicoDetMatlabLoader(args.hico_dir / 'anno_bbox.mat')
    for split in HicoDetMatlabLoader.Split:
        if should_stop:
            break
        img_count = 0
        gt_vr_count = 0
        gt_det_count = 0
        d2_det_count = 0
        skipped_images = 0

        output_dir = args.output_dir.joinpath(split.name.lower())
        output_dir.mkdir(exist_ok=True)

        for s in loader.iter_vr_samples(split, args.nms_threshold):
            if should_stop:
                break

            # If a .pth file already exist we might skip the image
            output_path = output_dir.joinpath(s.filename).with_suffix('.pth')
            if args.skip_existing and output_path.is_file():
                logger.debug(f'Skipping {split} image with existing .pth file: {s.filename}')
                continue

            # If no ground-truth visual relation is present, we skip the image
            img_count += 1
            gt_vr_count += len(s.gt_visual_relations)
            gt_det_count += len(s.gt_instances)
            if len(s.gt_visual_relations) == 0:
                logger.warning(f'Skipping {split} image without any visible ground-truth relation: {s.filename}')
                skipped_images += 1
                continue

            # Run detectron on the image, extracting features from both the detected and ground-truth objects
            image_path = args.hico_dir / split.image_dir / s.filename
            s.d2_feature_pyramid, s.d2_instances, s.gt_instances = detectron(image_path, s.gt_instances)

            # Move everything to cpu
            s.d2_instances = s.d2_instances.to('cpu')
            s.d2_feature_pyramid = {l: v.cpu() for l, v in s.d2_feature_pyramid.items()}

            # Counts
            d2_det_count += len(s.d2_instances)
            if len(s.d2_instances) == 0:
                logger.warning(f'Detectron could not find any object in {split} image: {s.filename}')

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(
                s.gt_visual_relations.instances.boxes, s.gt_visual_relations.instances.image_size)
            s.gt_visual_relations_full = VisualRelations(
                relation_indexes=indexes,
                linear_features=features,
            )

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(s.d2_instances.boxes, s.d2_instances.image_size)
            s.d2_visual_relations_full = VisualRelations(
                relation_indexes=indexes,
                linear_features=features,
            )

            # TODO s.feature_pyramid is quite big (50-70 MB) and not always needed
            del s.d2_feature_pyramid

            torch.save(s, output_path)

        message = '\n'.join((
            f'Split "{split}"',
            f'- Total images {img_count:,}',
            f'- Valid images {img_count - skipped_images:,}',
            f'- Skipped images {skipped_images:,}',
            f'- Ground-truth visual relations {gt_vr_count:,}',
            f'- Ground-truth instances {gt_det_count:,}',
            f'- Detectron instances {d2_det_count:,}',
        ))
        logger.info(f'\n{message}')
        with output_dir.joinpath(f'preprocessing_{int(time.time())}.log').open(mode='w') as f:
            f.write(message)


if __name__ == '__main__':
    main()
