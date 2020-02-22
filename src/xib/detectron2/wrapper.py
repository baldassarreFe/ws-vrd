from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Tuple, overload, Optional

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.structures import Boxes, Instances
from loguru import logger

from .node_features import boxes_to_node_features
from ..structures import ImageSize, clone_instances

# Detectron model pretrained on COCO object detection
COCO_CFG = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"

# Detectron model pretrained on LVIS instance segmentation
LVIS_CFG = "LVIS-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"


class DetectronWrapper(object):
    def __init__(
        self,
        config_file: Optional[str] = None,
        weights: Optional[str] = None,
        threshold: float = 0.5,
        image_library="PIL",
    ):
        cfg = get_cfg()

        if config_file == "model_zoo/coco-detection":
            cfg.merge_from_file(get_config_file(COCO_CFG))
            cfg.MODEL.WEIGHTS = get_checkpoint_url(COCO_CFG)
            if weights is not None:
                logger.warning(
                    "Weights should not be provided when selecting a model from the zoo."
                )
        elif config_file == "model_zoo/lvis-segmentation":
            cfg.merge_from_file(get_config_file(LVIS_CFG))
            cfg.MODEL.WEIGHTS = get_checkpoint_url(LVIS_CFG)
            if weights is not None:
                logger.warning(
                    "Weights should not be provided when selecting a model from the zoo."
                )
        else:
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHTS = weights

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        logger.info(f"Detectron2 configuration:\n{cfg}")

        self._patch_detectron()
        self.d2 = DefaultPredictor(cfg)
        self.image_library = image_library

    def open_image(self, img_path):
        """Loads an image as a numpy array with uint8 values in the range [0, 255]

        If the image has an EXIF orientation tag, `cv2` automatically applies
        the rotation when loading the image, while `PIL` loads the image "as is".

        If this wrapper is only used to detect objects, it's probably good
        to respect the EXIF tag and rotate the image before feeding it to the model.

        However, some external annotations (e.g. from the matlab file of HICO-DET
        or the json file of VRD) are made w.r.t. the non-rotated image. In this case,
        we must ignore the EXIF tag and extract features for those boxes w.r.t.
        the non-rotated image.
        """
        if self.image_library == "PIL":
            from PIL import Image

            # Some images are black and white, make sure they are read as RBG
            img = Image.open(img_path).convert("RGB")
            size = ImageSize(img.size[1], img.size[0])
            img = np.asarray(img)

            # PIL reads the image in RGB
            # The model might expect BGR inputs or RGB
            if self.d2.input_format == "BGR":
                img = img[:, :, ::-1]

        elif self.image_library == "cv2":
            import cv2

            img = cv2.imread(img_path.as_posix())
            size = ImageSize(*img.shape[:2])

            # cv2 reads the image in BGR
            # The model might expect BGR inputs or RGB
            if self.d2.input_format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Invalid image libary: {self.image_library}")

        return img, size

    @overload
    def __call__(
        self, image_path: Path, other_instances: None
    ) -> Tuple[Dict[str, torch.Tensor], Instances, None]:
        ...

    @overload
    def __call__(
        self, image_path: Path, other_instances: Instances
    ) -> Tuple[Dict[str, torch.Tensor], Instances, Instances]:
        ...

    def __call__(self, image_path, other_instances):
        """Perform detection, return feature pyramids for the whole image and box features for every detection.

        This method mimics `DefaultPredictor.__call__`, but:
        - it takes an image path as input
        - it does not enable/disable gradient computation
        - returns the full feature pyramid
        - returns the pooled box features for every detected box
        - optionally returns the pooled box features for every given box
        """
        start_time = time.perf_counter()

        # original_image.shape = [480, 640, 3]
        original_image, original_size = self.open_image(image_path)

        # Detectron has a requirement on min/max image size
        # img_tensor.shape = [3, 800, 1067]
        transform = self.d2.transform_gen.get_transform(original_image)
        img_tensor = torch.as_tensor(
            transform.apply_image(original_image).astype("float32").transpose(2, 0, 1)
        )

        # The image is moved to the right device, put into a batch, normalized and
        # padded (FPN has a requirement on size divisibility of the input tensor):
        # preprocessed_inputs.image_sizes[0] = [800, 1067]
        # preprocessed_inputs.tensor.shape = [1, 3, 800, 1088]
        batched_inputs = [
            {
                "image": img_tensor,
                "height": original_size.height,
                "width": original_size.width,
            }
        ]
        preprocessed_inputs = self.d2.model.preprocess_image(batched_inputs)
        feature_pyramid: Dict[str, torch.Tensor] = self.d2.model.backbone(
            preprocessed_inputs.tensor
        )

        # Proposals and detections are defined w.r.t. the size of the image tensor:
        # proposals.image_size == detections.image_size == img_tensor.shape[1:]
        proposals, _ = self.d2.model.proposal_generator(
            preprocessed_inputs, feature_pyramid
        )
        detections, _ = self.d2.model.roi_heads(
            preprocessed_inputs, feature_pyramid, proposals
        )

        # Keep detected_boxes for later because we need them defined w.r.t. img_tensor.shape = [800, 1067]
        # Also, detach boxes so that explanations produced through some form of backpropagation will only consider
        # FPN features extracted from the image. In other words, consider the detected boxes as if they were
        # ground-truth boxes, even if they are generated by the RPN.
        detected_boxes = Boxes(detections[0].pred_boxes.tensor.detach().clone())

        # Postprocessing of the newly detected object boxes:
        # - Resize boxes to match the original image size = [480, 640]
        # - Remove batch dimension
        # - Rename fields:
        #   - `pred_boxes` -> `boxes`
        #   - `pred_classes` -> `classes`
        detections: Instances = self.d2.model._postprocess(
            detections, batched_inputs, preprocessed_inputs.image_sizes
        )[0]["instances"]
        detections.boxes = Boxes(detections.pred_boxes.tensor)
        detections.remove("pred_boxes")
        detections.classes = detections.pred_classes
        detections.remove("pred_classes")

        # Use RoI Pooling to extract FPN features for each of the newly detected object boxes.
        # The boxes given to box_pooler must be defined w.r.t. img_tensor.shape = [800, 1067]
        features_list = [
            feature_pyramid[level] for level in self.d2.model.roi_heads.in_features
        ]
        detections.conv_features = self.d2.model.roi_heads.box_pooler(
            features_list, [detected_boxes]
        )

        # Also compute linear features for each of the newly detected object boxes
        detections.linear_features = boxes_to_node_features(
            detections.boxes, detections.image_size
        )

        # If other object instances are given, extract features for them as well.
        # Return the extracted features on the same device as the given instances.
        if other_instances is not None:
            if original_size != other_instances.image_size:
                logger.warning(
                    f"Original image has shape {original_image.shape[:2]}, but additional boxes "
                    f"are defined on an image of shape {other_instances.image_size}"
                )

            # Clone instances so the input is not modified
            other_instances = clone_instances(other_instances)

            # The boxes are defined on the original image, so we apply the transformation to resize them.
            other_boxes = Boxes(
                transform.apply_box(other_instances.boxes.tensor.detach().cpu().numpy())
            ).to(self.d2.model.device)

            # Extract conv features from the feature pyramid
            conv_features = self.d2.model.roi_heads.box_pooler(
                features_list, [other_boxes]
            )
            other_instances.conv_features = conv_features.to(
                other_instances.boxes.device
            )

            # Forward those conv features through the box predictor to get class logits.
            # It's true that we already have the ground-truth class for these
            # boxes, but the logits could hide some dark knowledge.
            pred_class_logits, _ = self.d2.model.roi_heads.box_predictor(
                self.d2.model.roi_heads.box_head(conv_features)
            )
            other_instances.probabs = pred_class_logits.softmax(dim=1).to(
                other_instances.boxes.device
            )[:, :-1]
            other_instances.scores = other_instances.probabs.gather(
                dim=1, index=other_instances.classes[:, None]
            ).squeeze(dim=1)

            # Compute linear features for each of the given object boxes
            # (it's ok to use the original boxes and original image size instead of the resized ones)
            other_instances.linear_features = boxes_to_node_features(
                other_instances.boxes, other_instances.image_size
            )

        # Squeeze out the batch dimension from the pyramid features
        feature_pyramid = {l: v.squeeze(dim=0) for l, v in feature_pyramid.items()}

        logger.debug(
            "".join(
                (
                    f"Extracted features for {len(detections):,} detected ",
                    f"and {len(other_instances):,} given "
                    if other_instances is not None
                    else "",
                    f"objects from {image_path.name} in {time.perf_counter() - start_time:.1f}s",
                )
            )
        )

        # Debug utils:
        """
        import matplotlib.pyplot as plt
        from xib.datasets.hico_det import OBJECTS
        for name, inst in {'detectron': detections, 'other': other_instances}.items():
            for i in range(len(inst)):
                x0, y0, x1, y1 = inst.boxes.tensor[i].cpu().numpy()
                try:
                    score = inst.scores[i]
                except:
                    score = 1
                fig, ax = plt.subplots(1, 1, figsize=(
                    original_size.width / 640 * 12, original_size.height / 640 * 12
                ))
                ax.imshow(original_image[:,:,::-1])
                ax.scatter(x0, y0, marker='o', c='r', zorder=1000, label=f'TL ({x0:.0f}, {y0:.0f})')
                ax.scatter(x1, y1, marker='D', c='r', zorder=1000, label=f'BR ({x1:.0f}, {y1:.0f})')
                ax.set_title(f'{image_path.name} {original_size} - '
                             f'{OBJECTS.get_str(inst.classes[i].item())} ({name}: {score:.1%})')
                ax.add_patch(plt.Rectangle(
                    (x0, y0),
                    width=x1 - x0,
                    height=y1 - y0,
                    fill=False,
                    linewidth=3,
                    color='blue'
                ))
                ax.set_xlim([0, original_size.width])
                ax.set_ylim([original_size.height, 0])
                ax.legend()
                fig.tight_layout()
                plt.show()
                plt.close(fig)
        """

        return feature_pyramid, detections, other_instances

    @staticmethod
    def _patch_detectron():
        """Patch detectron so that the class probabilities are returned.

        The method `detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference_single_image`
        would return only the highest score and the corresponding class, not the
        full vector of probabilities.
        """
        import detectron2.modeling.roi_heads.fast_rcnn
        from detectron2.layers.nms import batched_nms

        def fast_rcnn_inference_single_image_with_logits(
            boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
        ):
            """
            Single-image inference. Return bounding-box detection results by thresholding
            on scores and applying non-maximum suppression (NMS).

            Args:
                Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
                per image.

            Returns:
                Same as `fast_rcnn_inference`, but for only one image.
            """
            scores = scores[:, :-1]
            probabs = scores.clone()
            num_bbox_reg_classes = boxes.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            # Filter results based on detection scores
            filter_mask = scores > score_thresh  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero()
            if num_bbox_reg_classes == 1:
                boxes = boxes[filter_inds[:, 0], 0]
            else:
                boxes = boxes[filter_mask]

            # R'
            scores = scores[filter_mask]
            # R' x K
            probabs = probabs[filter_inds[:, 0], :]

            # Apply per-class NMS
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
            if topk_per_image >= 0:
                keep = keep[:topk_per_image]
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
            probabs = probabs[keep]

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            result.probabs = probabs
            result.pred_classes = filter_inds[:, 1]
            return result, filter_inds[:, 0]

        detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference_single_image = (
            fast_rcnn_inference_single_image_with_logits
        )
