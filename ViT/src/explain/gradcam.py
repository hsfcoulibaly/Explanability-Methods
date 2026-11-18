"""Grad-CAM for both a ResNet and a ViT (no extra library needed)."""
import torch
import torch.nn.functional as F

from ..models import gradcam_target_layer, is_vit


def _vit_tokens_to_grid(t):
    """(B, 1+N, C) tokens -> (B, C, h, w) grid, dropping the CLS token."""
    t = t[:, 1:, :]
    B, N, C = t.shape
    h = w = int(N ** 0.5)
    return t.transpose(1, 2).reshape(B, C, h, w)


def gradcam(model, x, target=None):
    """Returns (saliency (B,H,W) in [0,1], target classes (B,)).

    target: None -> explain the predicted class.
    """
    model.eval()
    layer = gradcam_target_layer(model)
    store = {}

    def fwd_hook(_, __, out):
        out.retain_grad()
        store["act"] = out

    h = layer.register_forward_hook(fwd_hook)
    try:
        x = x.clone().requires_grad_(True)  # so a ViT norm1 output carries grad
        logits = model(x)
        if target is None:
            target = logits.argmax(1)
        model.zero_grad(set_to_none=True)
        logits.gather(1, target.view(-1, 1)).sum().backward()
        act, grad = store["act"], store["act"].grad
    finally:
        h.remove()

    if is_vit(model):
        act, grad = _vit_tokens_to_grid(act), _vit_tokens_to_grid(grad)

    weights = grad.mean(dim=(2, 3), keepdim=True)          # GAP over space
    cam = F.relu((weights * act).sum(1, keepdim=True))      # (B,1,h,w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach(), target.detach()
