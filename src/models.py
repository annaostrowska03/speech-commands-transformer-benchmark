import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=12):
    """
    Loads a pre-trained ResNet18 and modifies it for 1-channel spectrogram input
    and 12-class speech command classification.
    """
    model = models.resnet18(weights="IMAGENET1K_V1")

    existing_layer = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=existing_layer.out_channels,
        kernel_size=existing_layer.kernel_size,
        stride=existing_layer.stride,
        padding=existing_layer.padding,
        bias=existing_layer.bias
    )

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model