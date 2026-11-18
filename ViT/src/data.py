"""Shared data pipeline: CIFAR-100 at 224x224 with ImageNet normalization.

Both models MUST use these loaders so the comparison is fair.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
NUM_CLASSES = 100

# Fixed subset of the test set used for ALL explanation + faithfulness runs.
EVAL_SUBSET_SIZE = 1000
EVAL_SUBSET_SEED = 0


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def test_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_datasets(root="./data"):
    train = datasets.CIFAR100(root, train=True, download=True, transform=train_transform())
    test = datasets.CIFAR100(root, train=False, download=True, transform=test_transform())
    return train, test


def get_loaders(root="./data", batch_size=128, num_workers=2):
    train, test = get_datasets(root)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def eval_subset_indices(n=EVAL_SUBSET_SIZE, seed=EVAL_SUBSET_SEED, total=10000):
    """Same indices on every machine, every run."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n, replace=False)).tolist()


def get_eval_subset(root="./data", batch_size=32, num_workers=2):
    test = datasets.CIFAR100(root, train=False, download=True, transform=test_transform())
    subset = Subset(test, eval_subset_indices())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def denormalize(x):
    """Normalized tensor (B,3,H,W) or (3,H,W) -> [0,1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
