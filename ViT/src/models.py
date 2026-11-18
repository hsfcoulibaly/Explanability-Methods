"""Model builders. Both start from ImageNet-pretrained weights (timm)."""
import torch
import timm

from .data import NUM_CLASSES


def build_resnet(name="resnet50", pretrained=True, num_classes=NUM_CLASSES):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def build_deit(name="deit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def build_model(arch, pretrained=True):
    if arch == "cnn":
        return build_resnet(pretrained=pretrained)
    if arch == "vit":
        return build_deit(pretrained=pretrained)
    raise ValueError(f"arch must be 'cnn' or 'vit', got {arch!r}")


def is_vit(model):
    return hasattr(model, "blocks") and hasattr(model, "cls_token")


def gradcam_target_layer(model):
    """Layer Grad-CAM hooks into.
    CNN: last bottleneck of layer4.
    ViT: norm1 of the last block. (The block OUTPUT gets ~zero gradient for
    patch tokens because only CLS reaches the head, so hook its input side.)
    """
    if is_vit(model):
        return model.blocks[-1].norm1
    return model.layer4[-1]


def set_fused_attn(model, enabled):
    """Fused attention hides the attention matrix; turn it off for rollout."""
    for blk in getattr(model, "blocks", []):
        blk.attn.fused_attn = enabled


def save_checkpoint(model, path, **extra):
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(arch, path, device="cpu"):
    model = build_model(arch, pretrained=False)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
