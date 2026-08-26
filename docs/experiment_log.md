# Experiment Log

All model experiments must be recorded here to track progress toward the final project report.

| ID | Date | Model Architecture | Key Changes | Val Macro F1 | Test Acc | Test Macro F1 | Test ROC-AUC | Cohen's Kappa | Decision / Insights |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-001** | 2026-08-14 | **EfficientNet-B0** | Baseline (Focal Loss, Cosine LR, 35% train split, 224x224) | 85.10% | 83.38% | 83.11% | 97.74% | 0.9115 | Standard CNN baseline established. High specificity (98.15%). |
| **EXP-002** | 2026-08-14 | **BiomedCLIP (ViT-B/16)** | Microsoft Medical Foundation Model fine-tuned on PubMed pretraining | 84.13% | 83.85% | 83.69% | 98.02% | 0.9128 | Foundation model outperforms CNN baseline across Test Acc, Macro F1, and Kappa. |
| **EXP-003** | 2026-08-20 | **BiomedCLIP + CBAM + Feature Fusion** | Novel Hybrid Architecture (Spatial + Channel Attention + Multi-Scale Pyramid) | 84.03% | 84.23% | 84.09% | 97.96% | 0.9128 | Single model peak for 224x224. CBAM boosted Glaucoma F1 to 59.9%. |
| **EXP-004** | 2026-08-20 | **Ensemble + TTA (224x224)** | Weighted Multi-Model Probability Averaging + 4-View Flip TTA | N/A | 85.85% | 85.72% | 98.39% | 0.9263 | Multi-model averaging peak on 35% train split. |
| **EXP-005** | 2026-08-26 | **EfficientNet-B3 + High-Res ($384\times 384$) + CLAHE** 🏆 | **70/15/15 Balanced Split (2,800 train), $384\times 384$ Resolution, LAB CLAHE, Focal Loss** | **89.55%** | **90.17%** | **90.03%** | **98.91%** | **0.9628** | 🏆 **NEW OVERALL SOTA BENCHMARK! First time crossing 90% threshold. Macular Scar (+14%), CSCR (+5%), Retinal Detachment (100%), Disc Edema (98.3%). Beats published EyeFusionNet IEEE 2023 (89.2%).** |
