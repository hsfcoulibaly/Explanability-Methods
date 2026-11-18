"""Attention rollout (Abnar & Zuidema, 2020) for timm ViT/DeiT.

Class-agnostic: it shows where the CLS token gathers information from,
not evidence for a specific class. Keep that in mind when comparing it
to Grad-CAM / IG in the report.
"""
import torch
import torch.nn.functional as F

from ..models import is_vit, set_fused_attn


@torch.no_grad()
def attention_rollout(model, x, target=None, head_fusion="mean", discard_ratio=0.0):
    """Returns (saliency (B,H,W) in [0,1], predicted classes (B,))."""
    assert is_vit(model), "attention rollout is ViT-only"
    model.eval()
    attns = []
    # attn_drop's output IS the softmaxed attention matrix (B, heads, N, N)
    hooks = [blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o))
             for blk in model.blocks]
    set_fused_attn(model, False)
    try:
        logits = model(x)
    finally:
        set_fused_attn(model, True)
        for h in hooks:
            h.remove()

    B, _, N, _ = attns[0].shape
    eye = torch.eye(N, device=x.device).expand(B, N, N)
    result = eye.clone()
    for a in attns:
        a = a.mean(1) if head_fusion == "mean" else a.max(1).values  # (B,N,N)
        if discard_ratio > 0:  # optionally drop weakest links
            flat = a.view(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            idx = flat.topk(k, dim=1, largest=False).indices
            flat.scatter_(1, idx, 0)
        a = 0.5 * a + 0.5 * eye             # account for residual connection
        a = a / a.sum(-1, keepdim=True)
        result = a @ result

    mask = result[:, 0, 1:]                  # CLS row, patch columns
    s = int(mask.shape[1] ** 0.5)
    mask = mask.reshape(B, 1, s, s)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    mask = mask - mask.amin(dim=(1, 2), keepdim=True)
    mask = mask / (mask.amax(dim=(1, 2), keepdim=True) + 1e-8)
    pred = logits.argmax(1) if target is None else target
    return mask, pred
