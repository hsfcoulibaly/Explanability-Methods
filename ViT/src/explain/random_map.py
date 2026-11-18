"""Random saliency map: the reference floor for deletion/insertion."""
import torch


def random_map(model, x, target=None):
    """Same signature as every explainer. Uniform noise, class-agnostic."""
    if target is None:
        with torch.no_grad():
            target = model.eval()(x).argmax(1)
    B, _, H, W = x.shape
    return torch.rand(B, H, W, device=x.device), target
