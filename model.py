"""
Shared model definition and CIFAR-10 normalisation constants.

The architecture is ResNet-18 adapted for 32x32 CIFAR images:
  - 3x3 stride-1 stem instead of the default 7x7 stride-2 conv
  - the initial max-pool removed (32x32 is too small to downsample early)
  - the final fully-connected layer resized to `num_classes`

Every stage in the pipeline (training, quantization, pruning, evaluation)
imports build_model from here so there is a single source of truth.
"""
import torch.nn as nn
from torchvision.models import resnet18

# CIFAR-10 channel mean/std (standard values)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def build_model(num_classes=10):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
