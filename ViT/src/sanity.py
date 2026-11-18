"""Model-randomization sanity check (Adebayo et al., 2018), cascading top-down.

Re-initialize layers from the head downward, cumulatively. After each stage,
recompute the saliency (for the ORIGINAL predicted class) and measure Spearman
rank correlation with the original map. A faithful method's correlation should
drop towards 0; a method that stays high is not explaining the learned weights.

Usage (Colab, from ViT/):
    python -m src.sanity --arch vit --ckpt $CKPT_DIR/vit.pt --out $CKPT_DIR/sanity_vit.json
"""
import argparse
import copy
import json

import torch
import torch.nn as nn

from .data import get_eval_subset
from .explain import get_explainer
from .models import load_checkpoint, is_vit


def stages(model):
    """Ordered (name, [modules]) from the output end to the input end."""
    if is_vit(model):
        s = [("head", [model.head])]
        s += [(f"blocks.{i}", [model.blocks[i]]) for i in reversed(range(len(model.blocks)))]
        s += [("patch_embed", [model.patch_embed])]
        return s
    return [("fc", [model.fc]), ("layer4", [model.layer4]), ("layer3", [model.layer3]),
            ("layer2", [model.layer2]), ("layer1", [model.layer1]),
            ("conv1+bn1", [model.conv1, model.bn1])]


def reinit(modules):
    for top in modules:
        for m in top.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()


def _ranks(t):  # (B,N) -> ranks (B,N)
    return t.argsort(1).argsort(1).float()


def spearman(a, b):
    """a, b: (B,H,W) -> (B,) Spearman correlation per image."""
    ra, rb = _ranks(a.flatten(1)), _ranks(b.flatten(1))
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-8)


def cascading_randomization(model, x, explainer, seed=0):
    """Returns {'original': 1.0, stage_name: mean spearman, ...}."""
    torch.manual_seed(seed)
    base, target = explainer(model, x)
    base, target = base.detach(), target.detach()
    rand_model = copy.deepcopy(model)
    out = {}
    for name, mods in stages(rand_model):
        reinit(mods)
        sal, _ = explainer(rand_model, x, target)
        out[name] = spearman(base, sal.detach()).mean().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "vit"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--n-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-root", default="./data")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(a.arch, a.ckpt, device)
    methods = a.methods or (["gradcam", "ig", "rollout"] if a.arch == "vit" else ["gradcam", "ig"])
    loader = get_eval_subset(a.data_root, a.batch_size)

    results = {}
    for name in methods:
        per_batch, n = [], 0
        for x, _ in loader:
            if n >= a.n_images:
                break
            per_batch.append(cascading_randomization(model, x.to(device), get_explainer(name)))
            n += x.shape[0]
        results[name] = {k: sum(d[k] for d in per_batch) / len(per_batch) for k in per_batch[0]}
        print(name, {k: round(v, 3) for k, v in results[name].items()})
    json.dump(results, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
