# Classification of Eye Diseases from Color Fundus Images

A reproducible deep learning research pipeline for **multi-class retinal disease classification** from colour fundus images, developed as a **B.Tech Major Project** at **Jadavpur University**.

> **Research-use only.** This project is not a clinical diagnostic device and must not be used for patient care.

---

## Team

| Name | Role |
|---|---|
| Gunjan Basak | Team Member |
| Chirantan Biswas | Team Member |
| Subhajit Gayen | Team Member |

**Supervisor:** Dr. Pawan Kumar Singh, Jadavpur University

---

## Dataset

10-class colour fundus image classification (Folder-based structure — no CSV files):

| Class | Train Folders (70%) | Val Folders (15%) | Test Folders (15%) | Total |
|---|:---:|:---:|:---:|:---:|
| Central Serous Chorioretinopathy | 280 | 60 | 60 | 400 |
| Diabetic Retinopathy | 280 | 60 | 60 | 400 |
| Disc Edema | 280 | 60 | 60 | 400 |
| Glaucoma | 280 | 60 | 60 | 400 |
| Healthy | 280 | 60 | 60 | 400 |
| Macular Scar | 280 | 60 | 60 | 400 |
| Myopia | 280 | 60 | 60 | 400 |
| Pterygium | 280 | 60 | 60 | 400 |
| Retinal Detachment | 280 | 60 | 60 | 400 |
| Retinitis Pigmentosa | 280 | 60 | 60 | 400 |
| **Total** | **2,800** | **600** | **600** | **4,000** |

Equally balanced split across all 10 disease classes (4,000 total images).

---

## Objectives

- Build reproducible data preparation and evaluation pipelines directly from folder directories.
- Establish fair CNN baselines (EfficientNet-B0, ResNet-50) before proposing novel architecture.
- Evaluate models with Macro F1, Balanced Accuracy, ROC-AUC, Cohen's Kappa, Sensitivity, and Specificity.
- Use Grad-CAM explainability to inspect model behaviour on fundus images.

---

## Repository Layout

```
configs/       Experiment configuration YAML files
data/
  train/       Training images (280 per class subfolder)
  val/         Validation images (60 per class subfolder)
  test/        Test images (60 per class subfolder)
docs/          Proposal, literature review, experiment records
scripts/       Repeatable CLI helpers (e.g. build_balanced_folders.py)
src/
  data/        Dataset class and preprocessing/augmentations
  losses/      Focal loss and weighted cross-entropy
  models/      timm backbone wrappers and novel CBAM + Fusion architectures
  utils/       Grad-CAM explainability
  metrics.py   Medical evaluation metrics
  train.py     Training entry point
  evaluate.py  Test-set evaluation entry point
tests/         Unit tests for all modules
outputs/       Local checkpoints, logs, and figures (not versioned)
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -e .[dev]

# 3. Build physical balanced 70/15/15 train/val/test folders
python scripts/build_balanced_folders.py

# 4. Run training
python -m src.train --config configs/config.yaml

# 5. Evaluate best model on test set
python -m src.evaluate --checkpoint outputs/best_model.pth --config configs/config.yaml
```

---

## Research Workflow

1. Literature review -> identify baseline models and gaps
2. Train and document baseline models (EfficientNet, BiomedCLIP)
3. Propose novel hybrid architecture (BiomedCLIP + CBAM Dual Attention + Feature Pyramid Fusion)
4. Multi-view Test-Time Augmentation (TTA) and multi-model probability blending
5. Controlled ablations and Grad-CAM interpretability

---

## License

MIT. Check every dataset's licence and citation requirements separately before use.
