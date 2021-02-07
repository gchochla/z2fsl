"""Global configuration options."""

from z2fsl.modules.feature_extractors import (
    GoogleNetAvgpool,
    ResNetAvgpool,
    Inception3Avgpool,
    VGGAvgpool,
    GoogleNet_dim,
    ResNet_dim,
    Inception_dim,
    VGG_dim,
)

EXTRACTOR_MAPPING = {
    'googlenet': GoogleNetAvgpool,
    'vgg': VGGAvgpool,
    'inception': Inception3Avgpool,
    'resnet': ResNetAvgpool,
}

DIM_MAPPING = {
    'googlenet': GoogleNet_dim,
    'vgg': VGG_dim,
    'inception': Inception_dim,
    'resnet': ResNet_dim,
}

TRAIN_SPLIT = (0.2, 1)
TEST_SPLIT = (0, 0.2)
