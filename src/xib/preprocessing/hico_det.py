import argparse

from typing import Union, Iterator
from pathlib import Path

import torch
import scipy.io
from loguru import logger

from ..utils import SigIntCatcher
from ..structures import VisualRelations, Boxes
from ..detectron2.utils import FeatureExtractor
from ..datasets.hico_det import HicoDet, HicoDetSample
from ..structures.boxes import boxes_to_node_features, boxes_to_edge_features


def parse_args():
    def resolve_path(path: str):
        return Path(path).expanduser().resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument('--matlab-path', required=True, type=resolve_path)
    parser.add_argument('--images-path', required=True, type=resolve_path)

    parser.add_argument('--output-path', required=True, type=resolve_path)
    parser.add_argument('--skip-existing', action='store_true')

    parser.add_argument('--threshold', required=True, type=float, default=.3)
    parser.add_argument('--detectron-home', required=True, type=resolve_path, default='~/detectron2')

    return parser.parse_args()


def load_visual_relations(matlab_path: Union[str, Path]) -> Iterator[HicoDetSample]:
    """Iterate over visual relations loaded from the matlab annotation file"""
    path = Path(matlab_path).expanduser().resolve()
    matlab_dict = scipy.io.loadmat(path)

    # Load interactions (600 unique subject-predicate-object
    # triplets where the subject is always a person)
    interactions = []
    for interaction in matlab_dict['list_action'].squeeze():
        interactions.append({
            'subject': 'person',
            'predicate': interaction['vname'].item().replace(' ', '_'),
            'object': interaction['nname'].item().replace(' ', '_')
        })

    # Load samples
    samples = []
    vr_count = 0
    skipped_images = 0
    for img in matlab_dict['bbox_train'].squeeze(0):
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
            # Invisible interaction, no humans or objects visible
            if interaction['invis'].item() == 1:
                continue

            interaction_id = interaction['id'].item() - 1
            num_interactions = interaction['connection'].shape[0]

            visual_relations['subject_classes'].extend([interactions[interaction_id]['subject']] * num_interactions)
            visual_relations['predicate_classes'].extend([interactions[interaction_id]['predicate']] * num_interactions)
            visual_relations['object_classes'].extend([interactions[interaction_id]['object']] * num_interactions)

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

        if len(visual_relations['subject_classes']) == 0:
            skipped_images += 1
            logger.debug(f'Skipping image without any visible relation: {filename}')
        else:
            vr_count += len(visual_relations['subject_classes'])
            yield HicoDetSample(
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

    logger.info(f'Loaded {vr_count:,} visual relations from {len(samples):,} images')
    logger.warning(f'Skipped {skipped_images} images without any visible relation')


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    should_stop = SigIntCatcher()

    # Check files and folders
    if not args.matlab_path.is_file():
        raise ValueError(f'Not a file: {args.matlab_path}')
    if not args.detectron_home.is_dir():
        raise ValueError(f'Not a directory: {args.detectron_home}')
    if not args.images_path.is_dir():
        raise ValueError(f'Not a directory: {args.images_path}')
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Build detectron model
    feature_extractor = FeatureExtractor.build(detectron_home=args.detectron_home, threshold=args.threshold)

    # Ground truth bounding boxes and human-object relations
    for sample in load_visual_relations(args.matlab_path):
        output_path = args.output_path.joinpath(sample.filename).with_suffix('.pth')
        if args.skip_existing and output_path.is_file():
            continue

        # TODO sample.feature_pyramid is quite big (50-70 MB) and not always needed
        # sample.feature_pyramid, sample.detections = feature_extractor(args.images_path / sample.filename)
        _, sample.detections = feature_extractor(args.images_path / sample.filename)

        sample.node_conv_features = sample.detections.features
        sample.node_linear_features = boxes_to_node_features(sample.detections.pred_boxes, sample.detections.image_size)
        sample.edge_linear_features, sample.edge_indices = boxes_to_edge_features(
            sample.detections.pred_boxes, sample.detections.image_size)

        torch.save(sample, output_path)

        if should_stop:
            break


if __name__ == '__main__':
    main()
