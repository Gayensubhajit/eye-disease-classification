# Experiment Log

All model experiments must be recorded here to track progress toward the final project report.

| ID | Date | Model Architecture | Key Changes | Val Macro F1 | Test Acc | Test Macro F1 | Test ROC-AUC | Cohen's Kappa | Decision / Insights |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-001** | 2026-08-14 | **EfficientNet-B0** | Baseline (Focal Loss, Cosine LR, 140/130/130 balanced folders) | **85.10%** | **83.38%** | **83.11%** | **97.74%** | **0.9115** | Strong baseline established. Glaucoma (F1: 51.9%) and Macular Scar (F1: 72.1%) are hardest to separate. |
| **EXP-002** | Pending | **BiomedCLIP (ViT-B/16)** | Medical Foundation Model fine-tuning (`configs/biomedclip.yaml`) | TBD | TBD | TBD | TBD | TBD | Next planned experiment. |
