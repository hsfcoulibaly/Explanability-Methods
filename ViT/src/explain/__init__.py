"""Every explainer has the same signature:
    explainer(model, x, target=None) -> (saliency (B,H,W) in [0,1], target (B,))
so metrics.py can treat them interchangeably.
"""
from .gradcam import gradcam
from .ig import integrated_gradients
from .rollout import attention_rollout
from .random_map import random_map

EXPLAINERS = {
    "random": random_map,           # reference floor, both models
    "gradcam": gradcam,
    "ig": integrated_gradients,
    "rollout": attention_rollout,   # ViT only
}


def get_explainer(name):
    return EXPLAINERS[name]
