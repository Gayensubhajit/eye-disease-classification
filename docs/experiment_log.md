# Experiment Log

All model experiments must be recorded here to track progress toward the final project report.

| ID | Date | Model Architecture | Key Changes | Val Macro F1 | Test Acc | Test Macro F1 | Test ROC-AUC | Cohen's Kappa | Decision / Insights |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-001** | 2026-08-14 | **EfficientNet-B0** | Baseline (Focal Loss, Cosine LR, 140/130/130 balanced folders) | **85.10%** | **83.38%** | **83.11%** | **97.74%** | **0.9115** | Standard CNN baseline established. High overall accuracy with 98.15% specificity. |
| **EXP-002** | 2026-08-14 | **BiomedCLIP (ViT-B/16)** | Microsoft Medical Foundation Model fine-tuned on PubMed pretraining | **84.13%** | **83.85%** | **83.69%** | **98.02%** | **0.9128** | Foundation model outperforms CNN baseline across Test Acc, Macro F1, ROC-AUC, and Kappa. Improved Glaucoma & Myopia. |
| **EXP-003** | 2026-08-20 | **BiomedCLIP + CBAM + Feature Fusion** | Novel Hybrid Architecture (Spatial + Channel Attention + Multi-Scale Pyramid) | **84.03%** | **84.23%** | **84.09%** | **97.96%** | **0.9128** | Single model peak performance. CBAM attention boosted Glaucoma F1 from 51.9% to 59.9% (+8.0%). |
| **EXP-004** | 2026-08-20 | **Ensemble + Test-Time Augmentation (TTA)** | Weighted Multi-Model Probability Averaging + Flip TTA | N/A | **85.85%** | **85.72%** | **98.39%** | **0.9263** | **Overall Project Peak Benchmark. 85.85% Test Acc, 98.39% ROC-AUC, 0.9263 Cohen's Kappa.** |
