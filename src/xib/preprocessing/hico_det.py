import argparse
import time
from pathlib import Path

import torch
from loguru import logger

from ..datasets.hico_det.matlab_reader import HicoDetMatlabLoader
from ..detectron2 import DetectronWrapper, boxes_to_edge_features
from ..logging import setup_logging, add_logfile
from ..structures import VisualRelations
from ..utils import SigIntCatcher


def parse_args():
    def resolve_path(path: str):
        return Path(path).expanduser().resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument("--hico-dir", required=True, type=resolve_path)
    parser.add_argument("--output-dir", required=True, type=resolve_path)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--nms-threshold", required=True, type=float, default=0.7)
    parser.add_argument(
        "--confidence-threshold", required=True, type=float, default=0.3
    )

    return parser.parse_args()


@logger.catch
def main():
    args = parse_args()

    # Check files and folders
    if not args.hico_dir.is_dir():
        raise ValueError(f"Not a directory: {args.hico_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_logfile(args.output_dir / f"preprocessing_{int(time.time())}.log")

    # Build detectron model
    torch.set_grad_enabled(False)
    detectron = DetectronWrapper(threshold=args.confidence_threshold)

    # Load ground truth bounding boxes and human-object relations from the matlab file,
    # then process the image with detectron to extract visual features of the boxes.
    should_stop = SigIntCatcher()
    loader = HicoDetMatlabLoader(args.hico_dir / "anno_bbox.mat")
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
            output_path = output_dir.joinpath(s.filename).with_suffix(".pth")
            if args.skip_existing and output_path.is_file():
                logger.debug(
                    f"Skipping {split} image with existing .pth file: {s.filename}"
                )
                continue

            # If no ground-truth visual relation is present, we skip the image
            img_count += 1
            gt_vr_count += len(s.gt_visual_relations)
            gt_det_count += len(s.gt_instances)
            if len(s.gt_visual_relations) == 0:
                logger.warning(
                    f"Skipping {split} image without any visible ground-truth relation: {s.filename}"
                )
                skipped_images += 1
                continue

            # Run detectron on the image, extracting features from both the detected and ground-truth objects
            image_path = args.hico_dir / split.image_dir / s.filename
            s.d2_feature_pyramid, s.d2_instances, s.gt_instances = detectron(
                image_path, s.gt_instances
            )

            # Move everything to cpu
            s.d2_instances = s.d2_instances.to("cpu")
            s.d2_feature_pyramid = {l: v.cpu() for l, v in s.d2_feature_pyramid.items()}

            # Counts
            d2_det_count += len(s.d2_instances)
            if len(s.d2_instances) == 0:
                logger.warning(
                    f"Detectron could not find any object in {split} image: {s.filename}"
                )

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(
                s.gt_visual_relations.instances.boxes,
                s.gt_visual_relations.instances.image_size,
            )
            s.gt_visual_relations_full = VisualRelations(
                relation_indexes=indexes, linear_features=features
            )

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(
                s.d2_instances.boxes, s.d2_instances.image_size
            )
            s.d2_visual_relations_full = VisualRelations(
                relation_indexes=indexes, linear_features=features
            )

            # TODO s.feature_pyramid is quite big (50-70 MB) and not always needed
            del s.d2_feature_pyramid

            torch.save(s, output_path)

        message = "\n".join(
            (
                f'Split "{split}":',
                f"- Total images {img_count:,}",
                f"- Valid images {img_count - skipped_images:,}",
                f"- Skipped images {skipped_images:,}",
                f"- Ground-truth visual relations {gt_vr_count:,}",
                f"- Ground-truth instances {gt_det_count:,}",
                f"- Detectron instances {d2_det_count:,}",
            )
        )
        logger.info(f"\n{message}")


if __name__ == "__main__":
    setup_logging()
    main()
