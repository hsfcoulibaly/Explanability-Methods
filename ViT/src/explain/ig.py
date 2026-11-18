"""Integrated Gradients via Captum, applied identically to both models."""
import torch
import torchvision.transforms.functional as TF
from captum.attr import IntegratedGradients


def make_baseline(x, kind="black"):
    """Team decision: use the SAME baseline for both models.
    'black' = zero image in pixel space (not zero in normalized space).
    """
    if kind == "black":
        from ..data import IMAGENET_MEAN, IMAGENET_STD
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return ((0 - mean) / std).expand_as(x).contiguous()
    if kind == "blur":
        return TF.gaussian_blur(x, kernel_size=51, sigma=15.0)
    raise ValueError(kind)


def integrated_gradients(model, x, target=None, n_steps=32, baseline="black",
                         internal_batch_size=16):
    """Returns (saliency (B,H,W) in [0,1], target classes (B,))."""
    model.eval()
    if target is None:
        with torch.no_grad():
            target = model(x).argmax(1)
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, baselines=make_baseline(x, baseline), target=target,
                        n_steps=n_steps, internal_batch_size=internal_batch_size)
    sal = attr.abs().sum(1)                  # aggregate over color channels
    sal = sal - sal.amin(dim=(1, 2), keepdim=True)
    sal = sal / (sal.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return sal.detach(), target
