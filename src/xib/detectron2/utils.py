import time
from pathlib import Path
from typing import Union, Dict, Tuple

import cv2
import torch
from loguru import logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from ..structures import Instances

class FeatureExtractor(DefaultPredictor):
    CFG_PATH = 'configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    WEIGHTS_URL = 'detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'

    def __call__(self, image_path: Path) -> (Dict[str, torch.Tensor], Instances):
        """Perform detection, return feature pyramids for the whole image and box features for every detection.

        This method mimics `DefaultPredictor.__call__`, but:
        - it takes an image path as input
        - it does not enable/disable gradient computation
        - returns the full feature pyramid and the pooled box features for every detected box

        All results are returned on the cpu.
        """
        start_time = time.perf_counter()

        original_image = cv2.imread(image_path.as_posix())
        resized_image = self.preprocess_image(original_image)
        feature_pyramid, detections = self.extract_features(resized_image)

        detections = detections.to('cpu')
        for level in feature_pyramid:
            feature_pyramid[level] = feature_pyramid[level].to('cpu')

        logger.debug(f'Extracted {len(detections):,} detections from '
                     f'{image_path.name} in {time.perf_counter() - start_time:.1f}s')

        # Debug utils:
        # import matplotlib.pyplot as plt
        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # plt.show()
        # print(HicoDet.object_id_to_name(detections.pred_classes))

        return feature_pyramid, detections

    def preprocess_image(self, original_image) -> torch.FloatTensor:
        # Whether the model expects BGR inputs or RGB
        if self.input_format == "RGB":
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Resize and make tensor
        resized_image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        resized_image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1))

        return resized_image

    def extract_features(self, preprocessed_image) -> Tuple[Dict[str, torch.Tensor], Instances]:
        batched_inputs = [{'image': preprocessed_image}]
        processed_images = self.model.preprocess_image(batched_inputs)

        feature_pyramid: Dict[str, torch.Tensor] = self.model.backbone(processed_images.tensor)
        proposals, _ = self.model.proposal_generator(processed_images, feature_pyramid)
        detections, _ = self.model.roi_heads(processed_images, feature_pyramid, proposals)

        # Detach boxes so that explanations produced through some form of backpropagation
        # will only consider FPN features and not RPN proposals. In other words, consider
        # the detected boxes as if they were ground-truth boxes.
        detections: Instances = detections[0]
        detections.pred_boxes.tensor.detach_()

        detections.features = self.model.roi_heads.box_pooler(
            [feature_pyramid[level] for level in self.cfg.MODEL.ROI_HEADS.IN_FEATURES],
            [detections.pred_boxes]
        )

        # Squeeze out the batch dimension from the pyramid features
        for level in feature_pyramid:
            feature_pyramid[level] = feature_pyramid[level].squeeze(dim=0)

        return feature_pyramid, detections

    @classmethod
    def build(cls, detectron_home: Union[str, Path], threshold: float = .5):
        cfg = get_cfg()
        cfg.merge_from_file(Path(detectron_home).joinpath(cls.CFG_PATH).expanduser().resolve().as_posix())
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = cls.WEIGHTS_URL
        return cls(cfg)
