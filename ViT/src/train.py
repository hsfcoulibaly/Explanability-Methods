"""Shared fine-tuning loop for both architectures.

Usage (Colab):
    python -m src.train --arch vit --epochs 5 --lr 1e-4 \
        --out /content/drive/MyDrive/xai/vit.pt
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn

from .data import get_loaders
from .models import build_model, save_checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(arch, epochs, lr, batch_size, weight_decay, out, data_root, seed, warmup_epochs=1):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, pretrained=True).to(device)
    train_loader, test_loader = get_loaders(data_root, batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):  # linear warmup, cosine decay
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"epoch {epoch} step {i}/{len(train_loader)} loss {loss.item():.3f}")

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: train loss {running / len(train_loader):.3f} "
              f"test acc {acc:.4f} ({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, out, arch=arch, epoch=epoch, test_acc=acc)
            print(f"  saved -> {out}")
    print(f"best test acc: {best_acc:.4f}")
    return best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "vit"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-4 for vit, 3e-4 for cnn")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    lr = a.lr if a.lr is not None else (1e-4 if a.arch == "vit" else 3e-4)
    train(a.arch, a.epochs, lr, a.batch_size, a.weight_decay, a.out, a.data_root, a.seed)


if __name__ == "__main__":
    main()
