# Quantitative Faithfulness Comparison of Explainability Methods Across CNN and Vision Transformer Architectures

A ResNet-50 and a DeiT-Small are fine-tuned on CIFAR-100 under an identical pipeline, explained with the same methods, and scored with one shared deletion/insertion faithfulness metric on a fixed 1,000-image evaluation subset.

| Model | Explainers |
|---|---|
| ResNet-50 (`cnn`) | Grad-CAM, Integrated Gradients, random baseline |
| DeiT-Small (`vit`) | Grad-CAM, Integrated Gradients, attention rollout, random baseline |

## Layout

```
ViT/
  requirements.txt
  notebooks/
    vit_train.ipynb      Colab: train DeiT, evaluate, sanity check
    cnn_train.ipynb      Colab: train ResNet-50, evaluate, sanity check
  src/                   shared code, used by BOTH models (run from ViT/)
    data.py              CIFAR-100 at 224x224, fixed 1,000-image eval subset
    models.py            ResNet-50 / DeiT-Small builders (timm), checkpoints
    train.py             fine-tuning loop (AdamW, warmup + cosine, label smoothing)
    metrics.py           deletion / insertion AUC
    run_eval.py          score model x explainer pairs -> results.json
    sanity.py            cascading model-randomization check
    explain/             gradcam.py, ig.py, rollout.py, random_map.py
EDA/  Results/  figures/ notes and outputs for the report
```

`dataset/` is unused: CIFAR-100 is downloaded automatically by torchvision into `ViT/data/`.

## Running (Google Colab, GPU runtime)

Open `ViT/notebooks/cnn_train.ipynb` or `vit_train.ipynb`, set the repo URL in the clone cell, and run top to bottom. Checkpoints and results are saved to Google Drive.

Or from a terminal inside `ViT/`:

```bash
pip install -r requirements.txt

# 1. fine-tune
python -m src.train --arch cnn --epochs 5 --out ckpt/cnn.pt
python -m src.train --arch vit --epochs 5 --out ckpt/vit.pt

# 2. faithfulness scores (drop --max-batches for the full 1,000 images)
python -m src.run_eval --arch cnn --ckpt ckpt/cnn.pt --out Results/results.json --max-batches 4
python -m src.run_eval --arch vit --ckpt ckpt/vit.pt --out Results/results.json --max-batches 4

# 3. mean-color removal ablation
python -m src.run_eval --arch vit --ckpt ckpt/vit.pt --out Results/results.json --removal mean

# 4. randomization sanity check
python -m src.sanity --arch vit --ckpt ckpt/vit.pt --out Results/sanity_vit.json
```

## Shared design decisions (change in one place)

- Data and preprocessing: `src/data.py`. Both models use the same loaders and the same 1,000 test images (seed 0).
- Deletion/insertion: `src/metrics.py` (8x8-pixel cells, 50 steps, Gaussian-blur removal by default).
- Integrated Gradients: black baseline, 32 steps, absolute value summed over channels.

## Status

Code implemented and smoke-tested on CPU with random inputs. Fine-tuning and the final results on Colab are in progress; see `Results/`.
