"""Deletion / insertion faithfulness (Petsiuk et al., 2018). Shared by both models.

Deletion:  start from the image, progressively replace the most important
           regions with the "removed" version; lower AUC = more faithful.
Insertion: start from the "removed" image, progressively restore the most
           important regions; higher AUC = more faithful.

Team decisions live in the defaults below. Change them in ONE place.
  CELL     -- regions are ranked/removed in CELLxCELL blocks. 8 does not
              align with the ViT's 16px patches, so neither model is favored.
  N_STEPS  -- number of points on the curve.
  REMOVAL  -- what "removed" means: 'blur' (default) or 'mean'.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CELL = 8
N_STEPS = 50
REMOVAL = "blur"


def removed_image(x, kind=REMOVAL):
    if kind == "blur":
        return TF.gaussian_blur(x, kernel_size=51, sigma=15.0)
    if kind == "mean":  # zero in normalized space = dataset mean color
        return torch.zeros_like(x)
    raise ValueError(kind)


def _cell_order(saliency, cell):
    """(B,H,W) saliency -> (B, n_cells) cell indices, most important first."""
    pooled = F.avg_pool2d(saliency.unsqueeze(1), cell).flatten(1)
    return pooled.argsort(dim=1, descending=True), pooled.shape[1]


@torch.no_grad()
def deletion_insertion(model, x, saliency, target, cell=CELL, n_steps=N_STEPS,
                       removal=REMOVAL, batch_size=64):
    """Returns dict with per-image 'deletion_auc', 'insertion_auc' (B,)
    and the full curves (B, n_steps+1) of target-class softmax probability.
    """
    model.eval()
    B, _, H, W = x.shape
    gh, gw = H // cell, W // cell
    order, n_cells = _cell_order(saliency, cell)
    x_removed = removed_image(x, removal)
    counts = torch.linspace(0, n_cells, n_steps + 1).round().long()

    curves = {"deletion": [], "insertion": []}
    for b in range(B):
        # rank[c] = position of cell c in the importance order
        rank = torch.empty(n_cells, dtype=torch.long, device=x.device)
        rank[order[b]] = torch.arange(n_cells, device=x.device)
        # masks[k] = 1 where cell is among the top counts[k] cells
        masks = (rank.unsqueeze(0) < counts.to(x.device).unsqueeze(1)).float()
        masks = masks.view(-1, 1, gh, gw)
        masks = F.interpolate(masks, size=(H, W), mode="nearest")  # (S,1,H,W)

        for mode in ("deletion", "insertion"):
            if mode == "deletion":
                batch = x[b] * (1 - masks) + x_removed[b] * masks
            else:
                batch = x_removed[b] * (1 - masks) + x[b] * masks
            probs = []
            for i in range(0, batch.shape[0], batch_size):
                logits = model(batch[i:i + batch_size])
                probs.append(logits.softmax(1)[:, target[b]])
            curves[mode].append(torch.cat(probs))

    out = {}
    xs = counts.float() / n_cells
    for mode in ("deletion", "insertion"):
        c = torch.stack(curves[mode])                       # (B, S)
        out[f"{mode}_curve"] = c.cpu()
        out[f"{mode}_auc"] = torch.trapezoid(c, xs.to(c.device), dim=1).cpu()
    return out


def evaluate_explainer(model, loader, explainer, device, max_batches=None, **metric_kw):
    """Run one model-explainer pair over the eval subset; return mean AUCs."""
    dels, ins = [], []
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        sal, target = explainer(model, x)
        r = deletion_insertion(model, x, sal.to(device), target, **metric_kw)
        dels.append(r["deletion_auc"])
        ins.append(r["insertion_auc"])
    dels, ins = torch.cat(dels), torch.cat(ins)
    return {"deletion_auc": dels.mean().item(), "deletion_std": dels.std().item(),
            "insertion_auc": ins.mean().item(), "insertion_std": ins.std().item(),
            "n": len(dels)}
