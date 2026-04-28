import inspect
import torch.nn as nn
from torchvision import models

try:
    from transformers import ASTConfig, ASTForAudioClassification
except ImportError:
    ASTConfig = None
    ASTForAudioClassification = None


class RepeatInputChannels(nn.Module):
    """Wrap a model and repeat input channels to match expected channel count."""

    def __init__(self, base_model, in_channels=1, out_channels=3):
        super().__init__()
        self.base_model = base_model
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        if x.shape[1] == self.out_channels:
            return self.base_model(x)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Unexpected input channels: got {x.shape[1]}, expected {self.in_channels} or {self.out_channels}"
            )

        repeats = self.out_channels // self.in_channels
        if self.in_channels * repeats != self.out_channels:
            raise ValueError(
                f"Cannot evenly repeat from {self.in_channels} channels to {self.out_channels} channels"
            )
        x = x.repeat(1, repeats, 1, 1)
        return self.base_model(x)


class ASTLogitsWrapper(nn.Module):
    """Wrap ASTForAudioClassification so forward returns logits directly."""

    def __init__(self, ast_model):
        super().__init__()
        self.ast_model = ast_model

    def forward(self, *args, **kwargs):
        outputs = self.ast_model(*args, **kwargs)
        return outputs.logits


def _require_transformers_for_ast():
    if ASTConfig is None or ASTForAudioClassification is None:
        raise ImportError(
            "The 'transformers' package is required to use the AST model. "
            "Install it with `pip install transformers`."
        )


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


def get_resnet18_no_audio_tweaks_model(
    num_classes=12,
    input_channels=1,
    use_pretrained=True,
    freeze_backbone=False,
    dropout=0.0,
):
    """Image-style ResNet18 baseline without audio-specific stem adaptation.

    Keeps the original 3-channel conv1 and repeats 1-channel spectrogram input to RGB-like 3-channel input.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes),
        )
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    return RepeatInputChannels(model, in_channels=input_channels, out_channels=3)


def get_mobilenetv2_model(
    num_classes=12,
    input_channels=1,
    use_pretrained=True,
    freeze_backbone=False,
    dropout=0.0,
):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if use_pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    existing_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=input_channels,
        out_channels=existing_layer.out_channels,
        kernel_size=existing_layer.kernel_size,
        stride=existing_layer.stride,
        padding=existing_layer.padding,
        bias=(existing_layer.bias is not None),
    )

    if use_pretrained and input_channels == 1:
        model.features[0][0].weight.data.copy_(existing_layer.weight.data.mean(dim=1, keepdim=True))
        if existing_layer.bias is not None and model.features[0][0].bias is not None:
            model.features[0][0].bias.data.copy_(existing_layer.bias.data)

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=max(0.0, float(dropout))),
        nn.Linear(num_ftrs, num_classes),
    )

    if freeze_backbone:
        for param in model.features[0][0].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def get_ast_model(
    num_classes=12,
    use_pretrained=True,
    freeze_backbone=False,
    dropout=0.0,
    model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
):
    _require_transformers_for_ast()

    if use_pretrained:
        model = ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        config = ASTConfig(num_labels=num_classes)
        model = ASTForAudioClassification(config)

    if dropout > 0.0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            model.classifier,
        )

    if freeze_backbone:
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return ASTLogitsWrapper(model)


MODEL_BUILDERS = {
    "resnet18": get_resnet18_model,
    "resnet18_no_audio_tweaks": get_resnet18_no_audio_tweaks_model,
    "mobilenetv2": get_mobilenetv2_model,
    "ast": get_ast_model,
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
