# Project Context

## Project

Classification of Eye Diseases from Color Fundus Images

## University

Jadavpur University

## Supervisor

Dr. Pawan Kumar Singh

## Duration

Department of Information Technology, Jadavpur University

---

## Objective

Develop a novel deep learning architecture for multi-class eye disease classification from color fundus images.

The objective is NOT to reproduce an existing paper.

The proposed model should improve upon existing methods through architectural improvements, preprocessing, attention mechanisms, feature fusion, or training strategy.

---

## Repository

eye-disease-classification

---

## Team

- Gunjan Basak
- Chirantan Biswas
- Subhajit Gayen

---

## Tech Stack

Python

PyTorch

OpenCV

Albumentations

NumPy

Pandas

scikit-learn

Matplotlib

timm

Grad-CAM

Git

GitHub

Jupyter Notebook

---

## Current Stage

Repository setup complete.

Currently performing:

- Literature review
- Dataset collection
- Dataset analysis
- Image preprocessing

No model has been finalized.

---

## Coding Style

- Modular code
- PEP8
- Type hints when practical
- Small reusable functions
- Clear docstrings

---

## Folder Structure

src/

models/

training/

losses/

utils/

preprocessing/

data/

experiments/

configs/

docs/

---

## Important Rules

Never hardcode paths.

Never commit datasets.

Never commit model checkpoints.

Always compare against baselines.

Every experiment must be reproducible.

Every improvement must be measurable.

---

## Research Philosophy

We first reproduce baseline models.

Only then propose a novel architecture.

Every architectural change must be justified by literature or experiments.

Avoid inventing complexity without evidence.

---

---

## Current State-of-the-Art Benchmarks (September 2026)

### 1. Primary 10-Class Retinal Disease Classification (Core Research Deliverable)
- **Dataset:** 4,000 balanced color fundus images (2,800 train / 600 val / 600 test) across 10 disease categories.
- **Top Architecture:** BiomedCLIP + CBAM Feature Fusion with CLAHE preprocessing (`biomedclip_cbam_fusion`).
- **Best Performance:** **91.00% Macro Accuracy**, **91.30% Macro F1-Score**, **0.9904 Macro ROC-AUC**.

### 2. External Literature SOTA Benchmark (Kaggle 4-Class Eye Diseases)
- **Dataset:** 4,217 images (2,949 train / 845 val / 423 test) across Cataract, Diabetic Retinopathy, Glaucoma, Normal.
- **Target to Beat:** Alsohemi & Dardouri, *Journal of Imaging* (MDPI, Aug 2025) — **95.12%**.
- **Our Model:** Quad-Ensemble MS-TTA (BiomedCLIP 31% + ConvNeXt-v2 26% + ResNet-50d 23% + EfficientNet-B3 20%).
- **Verified Result:** **95.74% Test Accuracy** (405/423 correct), **95.70% Macro F1**, **0.9929 ROC-AUC**, **0.9326 Cohen's Kappa**.
- **Diabetic Retinopathy:** **100.0% Sensitivity, 100.0% Specificity, 100.0% F1** (0 errors).
- **Report:** Formally documented in `docs/Benchmark_Progress_Update.pdf`.
