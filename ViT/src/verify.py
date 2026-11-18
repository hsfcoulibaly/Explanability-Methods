"""Verification + explanation analyses on top of the shared faithfulness metric.

Subcommands (run from ViT/, see notebooks/analysis.ipynb):

  collect   per-image deletion/insertion scores for every explainer, for the
            real saliency map AND its inverted copy (1 - map), plus the
            mean curves and the maps themselves.  Resumable (skips done files).
  summarize bootstrap CIs, paired difference vs random, inverted-map control,
            correct-vs-wrong split, method agreement; writes tables + figures.
  hooks     Grad-CAM target-layer ablation on the ViT (why norm1?).
  perclass  per-class accuracy of both checkpoints on the full test set.

Why the inverted-map control: if ranking cells by 1 - saliency does not make
deletion worse and insertion worse, the metric is not measuring the map.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_datasets, get_eval_subset
from .explain import get_explainer
from .metrics import N_STEPS, deletion_insertion
from .models import is_vit, load_checkpoint

METHODS = {"cnn": ["gradcam", "ig", "random"], "vit": ["gradcam", "ig", "rollout", "random"]}
POOL = 8  # saved maps are pooled to 28x28 (one value per metric cell)


# ----------------------------------------------------------------- collect --
def collect(arch, ckpt, out_dir, data_root="./data", batch_size=16, max_batches=None,
            removal="blur", methods=None, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(arch, ckpt, device)
    os.makedirs(out_dir, exist_ok=True)
    for name in methods or METHODS[arch]:
        path = os.path.join(out_dir, f"scores_{arch}_{name}_{removal}.npz")
        if os.path.exists(path):
            print("skip", path)
            continue
        explainer = get_explainer(name)
        acc = {k: [] for k in ["del", "ins", "del_inv", "ins_inv", "pred", "y", "maps"]}
        curves = {k: 0.0 for k in ["del", "ins", "del_inv", "ins_inv"]}
        n = 0
        for i, (x, y) in enumerate(get_eval_subset(data_root, batch_size)):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            sal, target = explainer(model, x)
            sal = sal.to(device)
            r = deletion_insertion(model, x, sal, target, removal=removal)
            ri = deletion_insertion(model, x, 1.0 - sal, target, removal=removal)
            acc["del"].append(r["deletion_auc"]); acc["ins"].append(r["insertion_auc"])
            acc["del_inv"].append(ri["deletion_auc"]); acc["ins_inv"].append(ri["insertion_auc"])
            for k, v in [("del", r["deletion_curve"]), ("ins", r["insertion_curve"]),
                         ("del_inv", ri["deletion_curve"]), ("ins_inv", ri["insertion_curve"])]:
                curves[k] = curves[k] + v.sum(0).numpy()
            acc["pred"].append(target.cpu()); acc["y"].append(y)
            acc["maps"].append(F.avg_pool2d(sal.unsqueeze(1), POOL).squeeze(1).cpu().half())
            n += len(x)
        out = {k: torch.cat(v).numpy() for k, v in acc.items()}
        for k, v in curves.items():
            out[f"curve_{k}"] = v / n
        np.savez_compressed(path, **out)
        print(f"{arch} {name}: del {out['del'].mean():.4f} ins {out['ins'].mean():.4f} | "
              f"inverted del {out['del_inv'].mean():.4f} ins {out['ins_inv'].mean():.4f} (n={n})")


# --------------------------------------------------------------- summarize --
def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, 1000):  # chunked to bound memory
        idx = rng.integers(0, len(x), (min(1000, n_boot - s), len(x)))
        means[s:s + len(idx)] = x[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load(d, arch, name, removal):
    p = os.path.join(d, f"scores_{arch}_{name}_{removal}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def summarize(scores_dir, fig_dir, res_dir, removal="blur"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import rankdata

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    D = {(a, m): _load(scores_dir, a, m, removal) for a in METHODS for m in METHODS[a]}
    D = {k: v for k, v in D.items() if v is not None}

    rows, inv_rows, split_rows = [], [], []
    for (a, m), d in D.items():
        lo_d, hi_d = boot_ci(d["del"]); lo_i, hi_i = boot_ci(d["ins"])
        row = dict(model=a, method=m, deletion=d["del"].mean(), del_lo=lo_d, del_hi=hi_d,
                   insertion=d["ins"].mean(), ins_lo=lo_i, ins_hi=hi_i, n=len(d["del"]))
        if (a, "random") in D and m != "random":
            r = D[(a, "random")]
            dd, di = d["del"] - r["del"], d["ins"] - r["ins"]  # same images -> paired
            (dl, dh), (il, ih) = boot_ci(dd), boot_ci(di)
            row.update(del_minus_random=dd.mean(), del_diff_lo=dl, del_diff_hi=dh,
                       ins_minus_random=di.mean(), ins_diff_lo=il, ins_diff_hi=ih,
                       frac_images_better_del=float((dd < 0).mean()),
                       frac_images_better_ins=float((di > 0).mean()))
        rows.append(row)
        inv_rows.append(dict(model=a, method=m, deletion=d["del"].mean(),
                             deletion_inverted=d["del_inv"].mean(), insertion=d["ins"].mean(),
                             insertion_inverted=d["ins_inv"].mean(),
                             passes=bool(d["del_inv"].mean() > d["del"].mean() and
                                         d["ins_inv"].mean() < d["ins"].mean())))
        ok = d["pred"] == d["y"]
        for lab, mask in [("correct", ok), ("wrong", ~ok)]:
            if mask.sum():
                split_rows.append(dict(model=a, method=m, group=lab, n=int(mask.sum()),
                                       deletion=d["del"][mask].mean(), insertion=d["ins"][mask].mean()))
    pd.DataFrame(rows).round(4).to_csv(f"{res_dir}/faithfulness_ci.csv", index=False)
    pd.DataFrame(inv_rows).round(4).to_csv(f"{res_dir}/inverted_control.csv", index=False)
    pd.DataFrame(split_rows).round(4).to_csv(f"{res_dir}/correct_vs_wrong.csv", index=False)

    # method agreement: mean per-image Spearman between pooled maps
    agree = {}
    for a in METHODS:
        ms = [m for m in METHODS[a] if (a, m) in D]
        R = {m: rankdata(D[(a, m)]["maps"].reshape(len(D[(a, m)]["maps"]), -1).astype(np.float32), axis=1)
             for m in ms}
        mat = np.eye(len(ms))
        for i, p in enumerate(ms):
            for j, q in enumerate(ms):
                if i < j:
                    A, B = R[p] - R[p].mean(1, keepdims=True), R[q] - R[q].mean(1, keepdims=True)
                    c = (A * B).sum(1) / (np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-9)
                    mat[i, j] = mat[j, i] = c.mean()
        agree[a] = (ms, mat)
    json.dump({a: {"methods": ms, "spearman": mat.round(4).tolist()} for a, (ms, mat) in agree.items()},
              open(f"{res_dir}/method_agreement.json", "w"), indent=2)

    colors = {"gradcam": "#4C78A8", "ig": "#F58518", "rollout": "#54A24B", "random": "#888888"}
    # fig 1: curves
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
    xs = np.linspace(0, 1, N_STEPS + 1)
    for r, a in enumerate(["cnn", "vit"]):
        for m in METHODS[a]:
            if (a, m) not in D:
                continue
            ax[r, 0].plot(xs, D[(a, m)]["curve_del"], color=colors[m], label=m)
            ax[r, 1].plot(xs, D[(a, m)]["curve_ins"], color=colors[m], label=m)
        ax[r, 0].set_title(f"{a.upper()} deletion (lower = better)")
        ax[r, 1].set_title(f"{a.upper()} insertion (higher = better)")
        ax[r, 0].set_ylabel("mean target-class probability")
        ax[r, 0].legend(fontsize=8)
    for a_ in ax[1]:
        a_.set_xlabel("fraction of cells changed")
    plt.tight_layout(); plt.savefig(f"{fig_dir}/curves_{removal}.png", dpi=150); plt.close()

    # fig 2: forest plot of paired difference vs random
    df = pd.DataFrame(rows)
    df = df[df.method != "random"]
    if len(df):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for k, (col, title) in enumerate([("del", "deletion: method - random (negative = better)"),
                                          ("ins", "insertion: method - random (positive = better)")]):
            for y_, (_, r_) in enumerate(df.iterrows()):
                lo, hi = r_[f"{col}_diff_lo"], r_[f"{col}_diff_hi"]
                mid = r_[f"{col}_minus_random"]
                ax[k].errorbar(mid, y_, xerr=[[mid - lo], [hi - mid]], fmt="o", color=colors[r_.method])
            ax[k].axvline(0, color="k", lw=1); ax[k].set_yticks(range(len(df)))
            ax[k].set_yticklabels([f"{a}/{m}" for a, m in zip(df.model, df.method)])
            ax[k].set_title(title, fontsize=9)
        plt.tight_layout(); plt.savefig(f"{fig_dir}/faithfulness_vs_random_ci.png", dpi=150); plt.close()

    # fig 3: inverted-map control
    ir = pd.DataFrame(inv_rows)
    if len(ir):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        lab = [f"{a}/{m}" for a, m in zip(ir.model, ir.method)]
        x = np.arange(len(ir)); w = 0.38
        ax[0].bar(x - w / 2, ir.deletion, w, label="map"); ax[0].bar(x + w / 2, ir.deletion_inverted, w, label="inverted map")
        ax[1].bar(x - w / 2, ir.insertion, w, label="map"); ax[1].bar(x + w / 2, ir.insertion_inverted, w, label="inverted map")
        ax[0].set_title("deletion AUC (inverted should be HIGHER)", fontsize=9)
        ax[1].set_title("insertion AUC (inverted should be LOWER)", fontsize=9)
        for a_ in ax:
            a_.set_xticks(x); a_.set_xticklabels(lab, rotation=40, ha="right", fontsize=8); a_.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{fig_dir}/inverted_control.png", dpi=150); plt.close()

    # fig 4: method agreement
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for k, a in enumerate(["cnn", "vit"]):
        if a not in agree:
            continue
        ms, mat = agree[a]
        im = ax[k].imshow(mat, vmin=-0.2, vmax=1, cmap="viridis")
        ax[k].set_xticks(range(len(ms))); ax[k].set_xticklabels(ms); ax[k].set_yticks(range(len(ms))); ax[k].set_yticklabels(ms)
        for i in range(len(ms)):
            for j in range(len(ms)):
                ax[k].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax[k].set_title(f"{a.upper()}: mean Spearman between maps", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/method_agreement.png", dpi=150); plt.close()
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(inv_rows).round(4).to_string())


# ------------------------------------------------------------------- hooks --
def hook_ablation(ckpt, out_json, fig_path=None, data_root="./data", n_images=256, batch_size=16):
    """Grad-CAM on the ViT with different target layers. The last block's OUTPUT
    should give ~zero gradient on patch tokens (only CLS reaches the head)."""
    import importlib
    gc_mod = importlib.import_module(".explain.gradcam", package=__package__)  # the package re-exports the function under the same name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint("vit", ckpt, device)
    assert is_vit(model)
    variants = {"blocks[-1].norm1 (ours)": lambda m: m.blocks[-1].norm1,
                "blocks[-1] output": lambda m: m.blocks[-1],
                "blocks[-2] output": lambda m: m.blocks[-2],
                "blocks[-4] output": lambda m: m.blocks[-4],
                "blocks[-1].norm2": lambda m: m.blocks[-1].norm2}
    orig = gc_mod.gradcam_target_layer
    res = {}
    try:
        for name, getter in variants.items():
            gc_mod.gradcam_target_layer = getter
            d, ins, flat = [], [], 0
            seen = 0
            for x, _ in get_eval_subset(data_root, batch_size):
                if seen >= n_images:
                    break
                x = x.to(device)
                sal, t = gc_mod.gradcam(model, x)
                flat += int((sal.flatten(1).amax(1) < 1e-6).sum())  # all-zero maps
                r = deletion_insertion(model, x, sal.to(device), t)
                d.append(r["deletion_auc"]); ins.append(r["insertion_auc"]); seen += len(x)
            d, ins = torch.cat(d), torch.cat(ins)
            res[name] = dict(deletion=d.mean().item(), insertion=ins.mean().item(),
                             degenerate_maps=flat, n=len(d))
            print(name, res[name])
    finally:
        gc_mod.gradcam_target_layer = orig
    json.dump(res, open(out_json, "w"), indent=2)
    if fig_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(res)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].barh(names, [res[n]["deletion"] for n in names]); ax[0].set_title("deletion (lower = better)", fontsize=9)
        ax[1].barh(names, [res[n]["insertion"] for n in names]); ax[1].set_title("insertion (higher = better)", fontsize=9)
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()


# --------------------------------------------------------------- per class --
@torch.no_grad()
def per_class(cnn_ckpt, vit_ckpt, out_json, fig_path=None, data_root="./data", batch_size=256):
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = get_datasets(data_root)
    loader = DataLoader(test, batch_size=batch_size, num_workers=2)
    out = {"classes": test.classes}
    for arch, ck in [("cnn", cnn_ckpt), ("vit", vit_ckpt)]:
        model, _ = load_checkpoint(arch, ck, device)
        hit = np.zeros(100); tot = np.zeros(100)
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu().numpy()
            y = y.numpy()
            np.add.at(tot, y, 1); np.add.at(hit, y, (p == y))
        out[arch] = (hit / tot).tolist()
        out[f"{arch}_overall"] = float(hit.sum() / tot.sum())
    json.dump(out, open(out_json, "w"), indent=2)
    if fig_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c, v = np.array(out["cnn"]), np.array(out["vit"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(c, v, s=12); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set_xlabel("ResNet-50 class accuracy"); ax[0].set_ylabel("DeiT-Small class accuracy")
        ax[0].set_title(f"per-class accuracy (overall {out['cnn_overall']:.3f} vs {out['vit_overall']:.3f})", fontsize=9)
        worst = np.argsort((c + v) / 2)[:12]
        w = 0.4; x = np.arange(len(worst))
        ax[1].bar(x - w / 2, c[worst], w, label="ResNet-50"); ax[1].bar(x + w / 2, v[worst], w, label="DeiT-Small")
        ax[1].set_xticks(x); ax[1].set_xticklabels([out["classes"][i] for i in worst], rotation=60, ha="right", fontsize=8)
        ax[1].set_title("12 hardest classes (mean of both models)", fontsize=9); ax[1].legend()
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    print({k: out[k] for k in ["cnn_overall", "vit_overall"]})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--arch", choices=["cnn", "vit"], required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--removal", choices=["blur", "mean"], default="blur")
    c.add_argument("--max-batches", type=int, default=None)
    c.add_argument("--batch-size", type=int, default=16)
    c.add_argument("--data-root", default="./data")
    s = sub.add_parser("summarize")
    s.add_argument("--scores-dir", required=True)
    s.add_argument("--fig-dir", required=True)
    s.add_argument("--res-dir", required=True)
    s.add_argument("--removal", default="blur")
    h = sub.add_parser("hooks")
    h.add_argument("--ckpt", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--fig", default=None)
    h.add_argument("--n-images", type=int, default=256)
    h.add_argument("--data-root", default="./data")
    q = sub.add_parser("perclass")
    q.add_argument("--cnn-ckpt", required=True)
    q.add_argument("--vit-ckpt", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--fig", default=None)
    q.add_argument("--data-root", default="./data")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.arch, a.ckpt, a.out_dir, a.data_root, a.batch_size, a.max_batches, a.removal)
    elif a.cmd == "summarize":
        summarize(a.scores_dir, a.fig_dir, a.res_dir, a.removal)
    elif a.cmd == "hooks":
        hook_ablation(a.ckpt, a.out, a.fig, a.data_root, a.n_images)
    else:
        per_class(a.cnn_ckpt, a.vit_ckpt, a.out, a.fig, a.data_root)


if __name__ == "__main__":
    main()
