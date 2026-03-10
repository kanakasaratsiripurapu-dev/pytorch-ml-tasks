# PyTorch ML Tasks — Submission Guide

## Overview

This repository contains **four self-contained PyTorch tasks**, each following the `pytorch_task_v1` protocol. Every task lives in its own directory as a single `task.py` file that trains, evaluates, and self-validates without any external test harness.

| Task ID | Series | Level | Algorithm |
|---|---|---|---|
| `linreg_lvl1_raw_tensors` | Linear Regression | 1 | Univariate LR with manual gradient descent |
| `logreg_lvl2_multiclass_softmax` | Logistic Regression | 2 | 3-class softmax via `nn.Module` + Adam |
| `cluster_lvl1_kmeans` | Clustering | 1 | K-Means from scratch with k-means++ init |
| `ae_lvl3_vae` | Autoencoders | 3 | VAE with reparameterisation trick on MNIST |

---

## Protocol: `pytorch_task_v1`

Each task exposes **nine required functions**:

| Function | Purpose |
|---|---|
| `get_task_metadata()` | Returns a dict with task name, type, algorithm, and description |
| `set_seed(seed)` | Sets reproducibility seeds (torch, numpy, CUDA) |
| `get_device()` | Returns `torch.device` (CUDA if available, else CPU) |
| `make_dataloaders(...)` | Builds training and validation data with an 80/20 split |
| `build_model(...)` | Constructs and returns the model (on device) |
| `train(model, ...)` | Runs the training loop; returns a history dict |
| `evaluate(model, ...)` | Computes metrics (loss, accuracy, R2, etc.) on a data split |
| `predict(model, ...)` | Returns predictions and/or probabilities |
| `save_artifacts(model, metrics, output_dir)` | Saves model weights, metrics JSON, and plots |

### Main block contract

Every `task.py` includes an `if __name__ == '__main__':` block that:

1. Seeds all RNGs for reproducibility
2. Loads data (train + validation splits)
3. Builds and trains the model
4. Evaluates on **both** train and validation splits
5. Prints all metrics
6. Runs quality-threshold assertions (e.g. R2 > 0.9, macro-F1 > 0.85)
7. Exits with code **0** on success or **1** on failure

---

## How to Run

### Prerequisites

```
pip install -r requirements.txt
```

Dependencies: `torch>=2.0`, `torchvision>=0.15`, `numpy>=1.24`, `matplotlib>=3.7`, `scikit-learn>=1.2`

### Running a single task

```bash
cd tasks
python linreg_lvl1_raw_tensors/task.py
python logreg_lvl2_multiclass_softmax/task.py
python cluster_lvl1_kmeans/task.py
python ae_lvl3_vae/task.py
```

Each script prints training progress, final metrics, and a PASS/FAIL summary. Exit code 0 means all checks passed.

### Running all tasks

```bash
cd tasks
for d in linreg_lvl1_raw_tensors logreg_lvl2_multiclass_softmax cluster_lvl1_kmeans ae_lvl3_vae; do
    echo "--- $d ---"
    python "$d/task.py" || echo "FAILED: $d"
    echo
done
```

---

## Task Details

### 1. `linreg_lvl1_raw_tensors` — Linear Regression (Raw Tensors)

- **What it does**: Fits `y = 2x + 3 + noise` using pure tensor operations — no `torch.nn`, no `torch.optim`, no autograd.
- **Gradients**: Computed by hand: `dJ/d(theta_0) = (2/m) * sum(h - y)`, `dJ/d(theta_1) = (2/m) * sum((h - y) * x)`
- **Data**: 500 synthetic samples, 80/20 split
- **Quality checks**: R2 > 0.9, MSE < 1.0, parameter error < 1.0 for both theta_0 and theta_1
- **Artifacts**: `model.pth`, `metrics.json`, `linreg_lvl1_loss.png`

### 2. `logreg_lvl2_multiclass_softmax` — Softmax Regression (Multiclass)

- **What it does**: Trains a single linear layer (`nn.Module`) on 3-class synthetic blobs using `CrossEntropyLoss` and Adam.
- **Math**: Docstring includes LaTeX for the softmax function and cross-entropy loss.
- **Data**: 600 samples from `make_blobs`, standardised, 80/20 split
- **Quality checks**: val macro-F1 > 0.85, val accuracy > 0.80, train accuracy > 0.80, accuracy gap < 0.10
- **Artifacts**: `model.pth`, `metrics.json`, `logreg_lvl2_boundary.png` (decision boundary contour)

### 3. `cluster_lvl1_kmeans` — K-Means (From Scratch)

- **What it does**: Implements Lloyd's algorithm with k-means++ initialisation and vectorised distance computation via `torch.cdist`.
- **Math**: Docstring includes the within-cluster SSE objective.
- **Data**: 800 synthetic blob samples (K=4), 80/20 train/val split
- **Quality checks**: Inertia converged (non-increasing), within 5% of sklearn KMeans, val ARI > 0.8, converged in < 100 iterations
- **Artifacts**: `centroids.pth`, `metrics.json`, `kmeans_inertia.png`, `kmeans_clusters.png`

### 4. `ae_lvl3_vae` — Variational Autoencoder

- **What it does**: Trains a VAE on MNIST with the reparameterisation trick. Loss = pixel-wise BCE (reconstruction) + KL divergence (regularisation).
- **Math**: Docstring includes LaTeX for ELBO, reparameterisation, and the closed-form KL for diagonal Gaussians.
- **Data**: MNIST (auto-downloaded), 85/15 train/val split
- **Quality checks**: Loss decreased over training, no NaN in generated samples, no posterior collapse (KL > 0.1), val/train loss ratio < 1.3, val reconstruction loss < 100
- **Artifacts**: `model.pth`, `metrics.json`, `vae_reconstructions.png`, `vae_generated_samples.png`, `vae_loss_curves.png`

---

## Repository Structure

```
tasks/
  requirements.txt
  .gitignore
  SUBMISSION.md            # this file
  linreg_lvl1_raw_tensors/
    task.py
  logreg_lvl2_multiclass_softmax/
    task.py
  cluster_lvl1_kmeans/
    task.py
  ae_lvl3_vae/
    task.py
```

Generated artifacts (`output/`, `data/`, `*.pth`, `*.png`) are git-ignored.

---

## Submission Checklist

- [x] Each task is a **single self-contained** `task.py` file
- [x] All 9 required protocol functions are implemented in every task
- [x] `if __name__ == '__main__'` block trains, evaluates on train+val, prints metrics, asserts thresholds, and uses `sys.exit`
- [x] Math formulas included in docstrings (LaTeX where required)
- [x] Train/validation split in every task (80/20 or 85/15)
- [x] No separate test files — the script itself is the test
- [x] All tasks exit 0 when quality checks pass
- [x] Artifacts saved to `./output/` (model weights, metrics JSON, plots)
- [x] `requirements.txt` provided with pinned minimum versions
- [x] `.gitignore` excludes generated files
