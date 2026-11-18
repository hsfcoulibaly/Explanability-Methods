"""Exploratory data analysis for CIFAR-100 (no model needed; runs on CPU).

Run from the repo root:   python EDA/eda.py
Writes figures to figures/eda_*.png and numbers to EDA/eda_summary.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ViT"))
from src.data import (EVAL_SUBSET_SEED, EVAL_SUBSET_SIZE, IMAGENET_MEAN,  # noqa: E402
                      IMAGENET_STD, eval_subset_indices)

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(ROOT, "ViT", "data")

train = datasets.CIFAR100(DATA, train=True, download=True)
test = datasets.CIFAR100(DATA, train=False, download=True)
names = train.classes
ytr, yte = np.array(train.targets), np.array(test.targets)
Xtr, Xte = train.data, test.data                       # (N,32,32,3) uint8
sub_idx = np.array(eval_subset_indices())
ysub = yte[sub_idx]

summary = {"train_size": len(ytr), "test_size": len(yte), "n_classes": len(names),
           "image_shape_raw": list(Xtr.shape[1:])}

# 1. class balance ---------------------------------------------------------
ctr, cte = np.bincount(ytr, minlength=100), np.bincount(yte, minlength=100)
csub = np.bincount(ysub, minlength=100)
summary["train_per_class"] = [int(ctr.min()), int(ctr.max())]
summary["test_per_class"] = [int(cte.min()), int(cte.max())]
summary["eval_subset_per_class"] = {"min": int(csub.min()), "max": int(csub.max()),
                                    "mean": float(csub.mean()), "classes_missing": int((csub == 0).sum())}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].bar(range(100), ctr, color="#4C78A8", label="train (500 each)")
ax[0].set_ylim(0, 600); ax[0].set_xlabel("class index"); ax[0].set_ylabel("images")
ax[0].set_title("CIFAR-100 train: perfectly balanced"); ax[0].legend(loc="lower right")
ax[1].bar(range(100), csub, color="#F58518")
ax[1].axhline(csub.mean(), color="k", ls="--", lw=1, label=f"mean = {csub.mean():.0f}")
ax[1].set_xlabel("class index"); ax[1].set_ylabel("images")
ax[1].set_title(f"Fixed eval subset (n={EVAL_SUBSET_SIZE}, seed {EVAL_SUBSET_SEED})")
ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/eda_class_balance.png", dpi=150); plt.close()

# 2. what the models actually see: 32px original vs 224px upscaled -----------
from PIL import Image  # noqa: E402
rng = np.random.default_rng(1)
pick = rng.choice(len(Xte), 6, replace=False)
fig, ax = plt.subplots(2, 6, figsize=(12, 4.6))
for j, i in enumerate(pick):
    im = Image.fromarray(Xte[i])
    ax[0, j].imshow(im); ax[0, j].set_title(names[yte[i]], fontsize=9)
    ax[1, j].imshow(im.resize((224, 224), Image.BILINEAR))
ax[0, 0].set_ylabel("original 32x32"); ax[1, 0].set_ylabel("input 224x224")
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
plt.suptitle("Images are tiny; 224px inputs are upscaled blur, so heatmaps look blocky", fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG}/eda_upscaling.png", dpi=150); plt.close()

# 3. pixel statistics vs ImageNet normalization ---------------------------------
mean, std = Xtr.reshape(-1, 3).mean(0) / 255, Xtr.reshape(-1, 3).std(0) / 255
summary["cifar100_mean"] = mean.round(4).tolist(); summary["cifar100_std"] = std.round(4).tolist()
summary["imagenet_mean"] = list(IMAGENET_MEAN); summary["imagenet_std"] = list(IMAGENET_STD)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
w = 0.35; x = np.arange(3)
ax.bar(x - w / 2, mean, w, label="CIFAR-100 mean"); ax.bar(x + w / 2, IMAGENET_MEAN, w, label="ImageNet mean")
ax.set_xticks(x); ax.set_xticklabels(["R", "G", "B"]); ax.legend(); ax.set_title("Channel means (close: normalization is OK)")
plt.tight_layout(); plt.savefig(f"{FIG}/eda_channel_stats.png", dpi=150); plt.close()

# 4. is the eval subset representative? (brightness / contrast vs full test set) ---
def feats(X):
    g = X.mean(-1)
    return g.mean((1, 2)) / 255, g.std((1, 2)) / 255
b_all, c_all = feats(Xte); b_sub, c_sub = feats(Xte[sub_idx])
summary["subset_vs_test"] = {"brightness_mean": [float(b_all.mean()), float(b_sub.mean())],
                             "contrast_mean": [float(c_all.mean()), float(c_sub.mean())]}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, (al, su), t in zip(ax, [(b_all, b_sub), (c_all, c_sub)], ["brightness", "contrast (std)"]):
    a.hist(al, bins=40, density=True, alpha=.5, label="full test (10k)")
    a.hist(su, bins=40, density=True, alpha=.5, label="eval subset (1k)")
    a.set_title(t); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/eda_subset_representativeness.png", dpi=150); plt.close()

json.dump(summary, open(os.path.join(ROOT, "EDA", "eda_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
