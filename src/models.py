"""Model definitions and checkpoint loaders for the three branches.

The published ThreatVisionAI ensemble fuses these three with weighted soft
voting (raw=0.50, wavelet=0.40, vit=0.10) at inference time. See
evaluate_ensemble.py.
"""

from torch import nn
from torchvision.models import resnet18

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


def build_resnet18(num_classes, pretrained=False):
    if pretrained:
        m = resnet18(weights="DEFAULT")
    else:
        m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_vit_tiny(num_classes, pretrained=False):
    if not _HAS_TIMM:
        raise ImportError("timm is required for ViT-Tiny. pip install timm")
    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def load_checkpoint(model, path, device):
    """Load a state dict from `path`. Handles both raw state_dict and wrapped
    checkpoints saved as {'model_state_dict': ...} or {'state_dict': ...}.
    """
    import torch
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    return model
