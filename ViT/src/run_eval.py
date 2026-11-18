"""Run deletion/insertion for one model over several explainers; merge into a JSON.

Usage (Colab, from ViT/):
    python -m src.run_eval --arch cnn --ckpt $CKPT_DIR/cnn.pt --out $CKPT_DIR/results.json
    python -m src.run_eval --arch vit --ckpt $CKPT_DIR/vit.pt --out $CKPT_DIR/results.json
    # mean-removal ablation:
    python -m src.run_eval --arch vit --ckpt ... --removal mean --out ...
    # quick check: --max-batches 4
"""
import argparse
import json
import os

import torch

from .data import get_eval_subset
from .explain import get_explainer
from .metrics import evaluate_explainer
from .models import load_checkpoint

DEFAULT_METHODS = {
    "cnn": ["gradcam", "ig", "random"],
    "vit": ["gradcam", "ig", "rollout", "random"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "vit"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="JSON file; results are merged into it")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--removal", choices=["blur", "mean"], default="blur")
    p.add_argument("--max-batches", type=int, default=None, help="None = full 1000 images")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(a.arch, a.ckpt, device)
    loader = get_eval_subset(a.data_root, a.batch_size)

    store = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name in a.methods or DEFAULT_METHODS[a.arch]:
        r = evaluate_explainer(model, loader, get_explainer(name), device,
                               max_batches=a.max_batches, removal=a.removal)
        r.update(test_acc=ckpt.get("test_acc"), epochs_saved=ckpt.get("epoch"))
        store[f"{a.arch}/{name}/{a.removal}"] = r
        print(f"{a.arch} {name} [{a.removal}]: del {r['deletion_auc']:.4f} "
              f"ins {r['insertion_auc']:.4f} (n={r['n']})")
        json.dump(store, open(a.out, "w"), indent=2)  # save after every pair


if __name__ == "__main__":
    main()
