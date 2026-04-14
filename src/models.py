import inspect
import torch.nn as nn
from torchvision import models

def get_resnet18_model(
    num_classes=12,
    input_channels=1,
    use_pretrained=True,
    freeze_backbone=False,
    dropout=0.0,
):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False


    existing_layer = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=input_channels, 
        out_channels=existing_layer.out_channels,
        kernel_size=existing_layer.kernel_size,
        stride=existing_layer.stride,
        padding=existing_layer.padding,
        bias=(existing_layer.bias is not None)
    )

    if use_pretrained and input_channels == 1:
        model.conv1.weight.data.copy_(existing_layer.weight.data.mean(dim=1, keepdim=True))
        if existing_layer.bias is not None and model.conv1.bias is not None:
            model.conv1.bias.data.copy_(existing_layer.bias.data)

    num_ftrs = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes),
        )
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    if freeze_backbone:
        for param in model.conv1.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


MODEL_BUILDERS = {
    "resnet18": get_resnet18_model,
}


def get_available_models():
    return sorted(MODEL_BUILDERS.keys())


def register_model(model_name, builder):
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if not callable(builder):
        raise TypeError("builder must be callable")
    if model_name in MODEL_BUILDERS:
        raise ValueError(f"Model '{model_name}' is already registered")
    MODEL_BUILDERS[model_name] = builder


def get_model(model_name, **model_kwargs):
    if model_name not in MODEL_BUILDERS:
        available = ", ".join(get_available_models())
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")

    builder = MODEL_BUILDERS[model_name]
    signature = inspect.signature(builder)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return builder(**model_kwargs)

    filtered_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in signature.parameters
    }
    return builder(**filtered_kwargs)