from collections import OrderedDict

import torchvision
from detectron2.data.catalog import Metadata
from omegaconf import OmegaConf
from torch import nn


def build_baseline(conf: OmegaConf, dataset_metadata: Metadata) -> nn.Module:
    num_classes = len(dataset_metadata.predicate_classes)

    backbone = torchvision.models.resnext50_32x4d(pretrained=True)
    backbone.fc = nn.Identity()

    classifier = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Linear(
            in_features=512, out_features=num_classes, bias=conf.classifier.last_bias
        ),
    )

    return nn.Sequential(OrderedDict(backbone=backbone, classifier=classifier))
