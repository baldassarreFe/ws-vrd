import time
import argparse

from typing import Union, Iterator, Optional, List, Dict
from pathlib import Path

import torch
import scipy.io

from loguru import logger

from ..utils import SigIntCatcher
from ..structures import VisualRelations, Boxes
from ..detectron2 import FeatureExtractor
from ..datasets.hico_det import HicoDet, HicoDetSample
from ..structures.boxes import boxes_to_node_features, boxes_to_edge_features


def parse_args():
    def resolve_path(path: str):
        return Path(path).expanduser().resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument('--hico-dir', required=True, type=resolve_path)
    parser.add_argument('--output-dir', required=True, type=resolve_path)
    parser.add_argument('--skip-existing', action='store_true')

    parser.add_argument('--threshold', required=True, type=float, default=.3)
    parser.add_argument('--detectron-home', required=True, type=resolve_path, default='~/detectron2')

    return parser.parse_args()


class HicoDetMatlabLoader(object):
    """Helper class to load HICO-Det annotations from the provided matlab file."""
    _path: Path
    _matlab_dict: Optional[dict] = None
    _interaction_triplets: List[Dict[str, str]] = None

    splits = ('train', 'train')

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

    def sample_iterator(self, split) -> Iterator[HicoDetSample]:
        """Iterate over samples from the split passed as parameter"""
        for img in self.matlab_dict[f'bbox_{split}'].squeeze(0):
            yield self._parse_one(img)

    def _parse_one(self, img) -> HicoDetSample:
        """Parse one HicoDetSample from the corresponding matlab dict"""
        filename = img['filename'].item()
        size = (
            img['size']['depth'].item().item(),
            img['size']['height'].item().item(),
            img['size']['width'].item().item(),
        )
        visual_relations = {
            'subject_classes': [],
            'predicate_classes': [],
            'object_classes': [],
            'subject_boxes': [],
            'object_boxes': [],
        }

        # All interaction types present in this image
        for interaction in img['hoi'].squeeze(0):
            interaction_id = interaction['id'].item() - 1
            num_interactions = interaction['connection'].shape[0]
            interaction_triplet = self.interaction_triplets[interaction_id]

            # Invisible interaction, no humans or objects visible
            if interaction['invis'].item() == 1:
                logger.debug(f'Skipping invisible interaction ('
                             f'{interaction_triplet["subject"]}, '
                             f'{interaction_triplet["predicate"]}, '
                             f'{interaction_triplet["object"]}) '   
                             f'in {filename}')
                continue

            visual_relations['subject_classes'].extend([interaction_triplet['subject']] * num_interactions)
            visual_relations['predicate_classes'].extend([interaction_triplet['predicate']] * num_interactions)
            visual_relations['object_classes'].extend([interaction_triplet['object']] * num_interactions)

            # All subjects for this interaction
            bb_subjects = [
                (human['x1'].item(), human['y1'].item(), human['x2'].item(), human['y2'].item())
                for human in interaction['bboxhuman'].squeeze(0)
            ]

            # All objects for this interaction
            bb_objects = [
                (object['x1'].item(), object['y1'].item(), object['x2'].item(), object['y2'].item())
                for object in interaction['bboxobject'].squeeze(0)
            ]

            # All instances of this interaction type
            for subject_box_id, object_box_id in interaction['connection']:
                visual_relations['subject_boxes'].append(bb_subjects[subject_box_id - 1])
                visual_relations['object_boxes'].append(bb_objects[object_box_id - 1])

        return HicoDetSample(
            filename=filename,
            img_size=size,
            gt_visual_relations=VisualRelations(
                subject_classes=torch.tensor(HicoDet.object_name_to_id(visual_relations['subject_classes'])),
                predicate_classes=torch.tensor(HicoDet.predicate_name_to_id(visual_relations['predicate_classes'])),
                object_classes=torch.tensor(HicoDet.object_name_to_id(visual_relations['object_classes'])),
                subject_boxes=Boxes(torch.tensor(visual_relations['subject_boxes'])),
                object_boxes=Boxes(torch.tensor(visual_relations['object_boxes'])),
            )
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
    feature_extractor = FeatureExtractor.build(detectron_home=args.detectron_home, threshold=args.threshold)

    # Load ground truth bounding boxes and human-object relations from the matlab file,
    # then process the image with detectron to extract visual features of the boxes.
    should_stop = SigIntCatcher()
    loader = HicoDetMatlabLoader(args.hico_dir / 'anno_bbox.mat')
    for split in HicoDetMatlabLoader.splits:
        if should_stop:
            break
        img_count = 0
        vr_count = 0
        det_count = 0
        skipped_images = 0
        args.output_dir.joinpath(split).mkdir(exist_ok=True)

        for s in loader.sample_iterator(split):
            if should_stop:
                break

            # If a .pth file already exist we might skip the image
            output_path = args.output_dir.joinpath(split).joinpath(s.filename).with_suffix('.pth')
            if args.skip_existing and output_path.is_file():
                logger.debug(f'Skipping {split} image with existing .pth file: {s.filename}')
                continue

            # If no ground-truth visual relation is present, we skip the image
            img_count += 1
            vr_count += len(s.gt_visual_relations)
            if len(s.gt_visual_relations) == 0:
                logger.warning(f'Skipping {split} image without any visible relation: {s.filename}')
                skipped_images += 1
                continue

            # If detectron can't find any objects, we skip the image
            feature_pyramid, detections = feature_extractor(args.hico_dir / 'images' / f'{split}2015' / s.filename)
            det_count += len(detections)
            if len(detections) == 0:
                logger.warning(f'Skipping {split} image because detectron could not find any object: {s.filename}')
                skipped_images += 1
                continue

            s.detections = detections
            s.node_conv_features = s.detections.features
            s.node_linear_features = boxes_to_node_features(
                s.detections.pred_boxes, s.detections.image_size)
            s.edge_linear_features, s.edge_indices = boxes_to_edge_features(
                s.detections.pred_boxes, s.detections.image_size)
            # TODO sample.feature_pyramid is quite big (50-70 MB) and not always needed
            # sample.feature_pyramid = feature_pyramid

            torch.save(s, output_path)

        message = (
            f'Split "{split}"\n'
            f'- Total images {img_count:,}\n'
            f'- Valid images {img_count - skipped_images:,}\n'
            f'- Skipped images {skipped_images:,}\n'
            f'- Visual relations {vr_count:,}\n'
            f'- Detections {det_count:,}\n'
        )
        logger.info(f'\n{message}')
        with args.output_dir.joinpath(split).joinpath(f'preprocessing_{int(time.time())}.log').open() as f:
            f.write(message)


if __name__ == '__main__':
    main()
