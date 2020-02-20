import argparse
import time
from pathlib import Path

import torch
from detectron2.data import DatasetCatalog
from loguru import logger

from ..datasets.vrd import register_vrd, data_dict_to_vr_sample
from ..detectron2 import DetectronWrapper, boxes_to_edge_features
from ..logging import setup_logging, add_logfile
from ..structures import VisualRelations
from ..utils import SigIntCatcher


def parse_args():
    def resolve_path(path: str):
        return Path(path).expanduser().resolve()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d2-dir",
        required=True,
        type=resolve_path,
        help="Where the pretrained detectron model is",
    )
    parser.add_argument(
        "--vrd-dir", required=True, type=resolve_path, help="Where the VRD dataset is"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=resolve_path,
        help="Where the processed samples will be save to",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not process images that already have a corresponding .pth file in the output directory",
    )
    parser.add_argument(
        "--confidence-threshold",
        required=True,
        type=float,
        default=0.3,
        help="Confidence threshold to keep a detected instance",
    )

    return parser.parse_args()


@logger.catch(reraise=True)
def main():
    args = parse_args()

    # Check files and folders
    if not args.vrd_dir.is_dir():
        raise ValueError(f"Not a directory: {args.vrd_dir}")
    register_vrd(args.vrd_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_logfile(args.output_dir / f"preprocessing_{int(time.time())}.log")

    # Build detectron model
    #
    # If the image has an EXIF orientation tag, `cv2` automatically applies
    # the rotation when loading the image, while `PIL` loads the image "as is".
    # However, the annotations from the matlab file of HICO-DET are made w.r.t.
    # the non-rotated image. Therefore, we must ignore the EXIF tag and extract
    # features for those boxes w.r.t. the non-rotated image.
    torch.set_grad_enabled(False)
    detectron = DetectronWrapper(
        config_file=args.d2_dir.joinpath("config.yaml").as_posix(),
        weights=args.d2_dir.joinpath("model_final.pth").as_posix(),
        threshold=args.confidence_threshold,
        image_library="PIL",
    )

    # Load ground truth bounding boxes and human-object relations from the matlab file,
    # then process the image with detectron to extract visual features of the boxes.
    should_stop = SigIntCatcher()
    for split in ["train", "test"]:
        if should_stop:
            break
        img_count = 0
        gt_vr_count = 0
        gt_det_count = 0
        d2_det_count = 0
        skipped_images = 0

        output_dir = args.output_dir / split
        output_dir.mkdir(exist_ok=True)

        for data_dict in DatasetCatalog.get(f"vrd_relationship_detection_{split}"):
            if should_stop:
                break

            # If a .pth file already exist we might skip the image
            filename=Path(data_dict["file_name"]).name
            output_path = output_dir.joinpath(filename).with_suffix(".pth")
            if args.skip_existing and output_path.is_file():
                logger.debug(
                    f"Skipping {split} image with existing .pth file: {filename}"
                )
                continue

            sample = data_dict_to_vr_sample(data_dict)

            # If no ground-truth visual relation is present, we skip the image
            img_count += 1
            gt_vr_count += len(sample.gt_visual_relations)
            gt_det_count += len(sample.gt_instances)
            if len(sample.gt_visual_relations) == 0:
                logger.warning(
                    f"Skipping {split} image without any visible ground-truth relation: {sample.filename}"
                )
                skipped_images += 1
                continue

            # Run detectron on the image, extracting features from both the detected and ground-truth objects
            sample.d2_feature_pyramid, sample.d2_instances, sample.gt_instances = detectron(
                Path(data_dict["file_name"]), sample.gt_instances
            )

            # Move everything to cpu
            sample.d2_instances = sample.d2_instances.to("cpu")
            sample.d2_feature_pyramid = {
                l: v.cpu() for l, v in sample.d2_feature_pyramid.items()
            }

            # Counts
            d2_det_count += len(sample.d2_instances)
            if len(sample.d2_instances) == 0:
                logger.warning(
                    f"Detectron could not find any object in {split} image: {sample.filename}"
                )

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(
                sample.gt_visual_relations.instances.boxes,
                sample.gt_visual_relations.instances.image_size,
            )
            sample.gt_visual_relations_full = VisualRelations(
                relation_indexes=indexes, linear_features=features
            )

            # Build a fully connected graph using gt boxes as nodes
            features, indexes = boxes_to_edge_features(
                sample.d2_instances.boxes, sample.d2_instances.image_size
            )
            sample.d2_visual_relations_full = VisualRelations(
                relation_indexes=indexes, linear_features=features
            )

            # TODO data_dict.feature_pyramid is quite big (50-70 MB) and not always needed
            del sample.d2_feature_pyramid

            torch.save(sample, output_path)

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
