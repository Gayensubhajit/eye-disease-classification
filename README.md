# Classification of Eye Diseases from Color Fundus Images

A reproducible deep learning research pipeline for **multi-class retinal disease classification** from colour fundus images, developed as a **7th–8th Semester B.Tech Major Project** at **Jadavpur University**.

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

10-class colour fundus image classification:

| Class | Train | Val | Test |
|---|---|---|---|
| Central Serous Chorioretinopathy | 81 | 10 | 10 |
| Diabetic Retinopathy | 1207 | 151 | 151 |
| Disc Edema | 101 | 13 | 13 |
| Glaucoma | 1079 | 135 | 135 |
| Healthy | 819 | 103 | 102 |
| Macular Scar | 356 | 44 | 44 |
| Myopia | 400 | 50 | 50 |
| Pterygium | 13 | 2 | 2 |
| Retinal Detachment | 100 | 12 | 13 |
| Retinitis Pigmentosa | 111 | 14 | 14 |
| **Total** | **4,267** | **534** | **534** |

Stratified 80/10/10 split from 5,335 original images.

---

## Objectives

- Build reproducible data preparation and evaluation pipelines.
- Establish fair CNN baselines (EfficientNet-B0, ResNet-50) before proposing novel architecture.
- Evaluate models with Macro F1, ROC-AUC, Cohen's Kappa, Sensitivity, and Specificity.
- Use Grad-CAM explainability to inspect model behaviour on fundus images.

---

## Repository Layout

```
configs/       Experiment configuration YAML files
data/          Split manifests (train/val/test CSVs — images not versioned)
docs/          Proposal, literature review, experiment records
scripts/       Repeatable CLI helpers (e.g. create_splits.py)
src/
  data/        Dataset class and preprocessing/augmentations
  losses/      Focal loss and weighted cross-entropy
  models/      timm backbone wrappers
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

# 3. Generate train/val/test splits
python scripts/create_splits.py --config configs/config.yaml

# 4. Run training
python -m src.train --config configs/config.yaml

# 5. Evaluate best model on test set
python -m src.evaluate --checkpoint outputs/best_model.pth --config configs/config.yaml
```

---

## Research Workflow

1. Literature review → identify baseline models and gaps
2. Train and document EfficientNet-B0 baseline
3. Identify a measured limitation from baseline results
4. Add one justified architectural improvement
5. Run controlled ablations, report confidence intervals

---

## License

MIT. Check every dataset's licence and citation requirements separately before use.
