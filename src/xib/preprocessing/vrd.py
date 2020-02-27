import argparse
import time
from pathlib import Path

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from loguru import logger

from xib.datasets import register_unrel, register_vrd, register_datasets
from xib.datasets.common import data_dict_to_vr_sample
from xib.detectron2 import DetectronWrapper, boxes_to_edge_features
from xib.logging import setup_logging, add_logfile
from xib.structures import VisualRelations
from xib.utils import SigIntCatcher


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
        "--dataset",
        required=True,
        choices=["vrd", "unrel"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--data-dir", required=True, type=resolve_path, help="Where the dataset is"
    )
    parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Save the feature pyramid extracted by detectron",
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
    if not args.data_dir.is_dir():
        raise ValueError(f"Not a directory: {args.data_dir}")

    register_datasets(args.data_dir)
    metadata = MetadataCatalog.get(f"{args.dataset}_relationship_detection")
    add_logfile(metadata.processed / f"preprocessing_{int(time.time())}.log")

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

    # Load ground truth bounding boxes and human-object relations from
    # the json annotation file, then process the image with detectron
    # to extract visual features of the boxes.
    should_stop = SigIntCatcher()
    for split in metadata.splits:
        if should_stop:
            break

        output_dir = MetadataCatalog.get(
            f"{args.dataset}_relationship_detection_{split}"
        ).graph_root
        output_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        gt_vr_count = 0
        gt_det_count = 0
        d2_det_count = 0
        skipped_images = 0

        for data_dict in DatasetCatalog.get(
            f"{args.dataset}_relationship_detection_{split}"
        ):
            if should_stop:
                break

            # If a .pth file already exist we might skip the image
            filename = Path(data_dict["file_name"]).name
            out_path_graph = output_dir.joinpath(filename).with_suffix(".graph.pth")
            out_path_fp = output_dir.joinpath(filename).with_suffix(".fp.pth")
            if (
                args.skip_existing
                and out_path_graph.is_file()
                and out_path_fp.is_file()
            ):
                logger.debug(
                    f"Skipping {split} image with existing .pth files: {filename}"
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
            d2_feature_pyramid, sample.d2_instances, sample.gt_instances = detectron(
                Path(data_dict["file_name"]), sample.gt_instances
            )

            # Move everything to cpu
            sample.d2_instances = sample.d2_instances.to("cpu")
            d2_feature_pyramid = {l: v.cpu() for l, v in d2_feature_pyramid.items()}

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

            # d2_feature_pyramid is quite big (50-70 MB) and not always needed,
            # so we save it in a separate file
            torch.save(sample, out_path_graph)
            if args.pyramid:
                torch.save(d2_feature_pyramid, out_path_fp)

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
