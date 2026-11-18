# Quantitative Faithfulness Comparison of Explainability Methods Across CNN and Vision Transformer Architectures

A ResNet-50 and a DeiT-Small are fine-tuned on CIFAR-100 under an identical pipeline, explained with the same methods, and scored with one shared deletion/insertion faithfulness metric on a fixed 1,000-image evaluation subset.

**Question:** do explanation methods built for CNNs (Grad-CAM, Integrated Gradients) still tell the truth on Vision Transformers, compared with a ViT-specific method (attention rollout)?

| Model | Explainers |
|---|---|
| ResNet-50 (`cnn`) | Grad-CAM, Integrated Gradients, random baseline |
| DeiT-Small (`vit`) | Grad-CAM, Integrated Gradients, attention rollout, random baseline |

## Layout

```
ViT/
  requirements.txt
  notebooks/
    run_all.ipynb        Colab: train both models, score all methods (start here)
    analysis.ipynb       Colab: EDA, verification, explanations -> EDA/ figures/ Results/
    vit_train.ipynb      Colab: DeiT only (older)
    cnn_train.ipynb      Colab: ResNet-50 only (older)
  src/                   shared code, used by BOTH models (run from ViT/)
    data.py              CIFAR-100 at 224x224, fixed 1,000-image eval subset
    models.py            ResNet-50 / DeiT-Small builders (timm), checkpoints
    train.py             fine-tuning loop (AdamW, warmup + cosine, label smoothing)
    metrics.py           deletion / insertion AUC
    run_eval.py          score model x explainer pairs -> results.json
    sanity.py            cascading model-randomization check
    verify.py            CIs, inverted-map control, Grad-CAM hook ablation, per-class accuracy
    explain/             gradcam.py, ig.py, rollout.py, random_map.py
EDA/                     eda.py (CIFAR-100 EDA), eda_summary.json, per-class accuracy
figures/                 every figure (eda_*.png, curves, CIs, heatmaps, sanity, ...)
Results/                 faithfulness.csv, CI / control tables, final_report.tex and PDF
```

`dataset/` is unused: CIFAR-100 is downloaded automatically by torchvision into `ViT/data/`.

## Running (Google Colab, GPU runtime)

Order: `run_all.ipynb` (train + scores), then `analysis.ipynb` (EDA + verification). The analysis notebook zips its outputs; unzip at the repo root and push.

1. Open `ViT/notebooks/run_all.ipynb` in Colab (File > Open notebook > GitHub > `hsfcoulibaly/Explanability-Methods`), pick an A100/L4 GPU, Run all. It clones the repo, trains both models (3 epochs), runs all scores, and saves checkpoints and `results.json` to Google Drive (`xai-cnn-vit/`). Finished steps are skipped, so after a disconnect rerun the setup cell and Run all.
2. Open `ViT/notebooks/analysis.ipynb` and run it to produce the EDA, verification and explanation outputs.

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

## Results so far

Fine-tuned for 3 epochs; test accuracy: ResNet-50 **82.3%**, DeiT-Small **87.9%**. Faithfulness on the fixed 1,000 images (blur removal; deletion lower = better, insertion higher = better). Full table incl. mean-color removal: `Results/faithfulness.csv`.

| Model | Method | Deletion | Insertion |
|---|---|---|---|
| ResNet-50 | Grad-CAM | 0.189 | 0.543 |
| ResNet-50 | IG | 0.208 | 0.478 |
| ResNet-50 | random | 0.217 | 0.217 |
| DeiT-Small | Grad-CAM | 0.432 | 0.471 |
| DeiT-Small | IG | 0.337 | 0.580 |
| DeiT-Small | rollout | 0.362 | 0.548 |
| DeiT-Small | random | 0.433 | 0.431 |

Read each method against its own random baseline: ViTs tolerate blurred cells better, so absolute ViT scores are higher. Grad-CAM beats random clearly on the CNN but is close to random on the ViT; IG is the most consistent across both. With mean-color removal the insertion ranking holds but deletion is unreliable (gray fill is out of distribution).

## Status

- Done: pipeline, training, all scores (blur and mean removal), EDA (`EDA/`, `figures/eda_*`), report draft (`Results/final_report.tex` / `.pdf`).
- To run in `analysis.ipynb`: confidence intervals, inverted-map control, randomization sanity check, Grad-CAM hook ablation, per-class accuracy, qualitative heatmaps. Then fill the red `[...]` sections of the report.
- Not tracked in git: `ViT/data/` (CIFAR-100 downloads itself).

## Compiling the report

```bash
cd Results
pdflatex final_report.tex && pdflatex final_report.tex
```
