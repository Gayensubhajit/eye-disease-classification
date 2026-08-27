# Experiment Log

All model experiments must be recorded here to track progress toward the final project report.

| ID | Date | Model Architecture | Key Changes | Val Macro F1 | Test Acc | Test Macro F1 | Test ROC-AUC | Cohen's Kappa | Decision / Insights |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-001** | 2026-08-14 | **EfficientNet-B0** | Baseline (Focal Loss, Cosine LR, 35% train split, 224x224) | 85.10% | 83.38% | 83.11% | 97.74% | 0.9115 | Standard CNN baseline established. High specificity (98.15%). |
| **EXP-002** | 2026-08-14 | **BiomedCLIP (ViT-B/16)** | Microsoft Medical Foundation Model fine-tuned on PubMed pretraining | 84.13% | 83.85% | 83.69% | 98.02% | 0.9128 | Foundation model outperforms CNN baseline across Test Acc, Macro F1, and Kappa. |
| **EXP-003** | 2026-08-20 | **BiomedCLIP + CBAM + Feature Fusion** | Novel Hybrid Architecture (Spatial + Channel Attention + Multi-Scale Pyramid, 224x224) | 84.03% | 84.23% | 84.09% | 97.96% | 0.9128 | Single model peak for 224x224 on 35% split. |
| **EXP-004** | 2026-08-20 | **Ensemble + TTA (224x224)** | Weighted Multi-Model Probability Averaging + 4-View Flip TTA | N/A | 85.85% | 85.72% | 98.39% | 0.9263 | Multi-model averaging peak on 35% train split. |
| **EXP-005** | 2026-08-26 | **EfficientNet-B3 + High-Res ($384\times 384$) + CLAHE** | 70/15/15 Balanced Split (2,800 train), $384\times 384$ Resolution, LAB CLAHE, Focal Loss | 89.55% | 90.17% | 90.03% | 98.91% | 0.9628 | First time crossing 90% threshold. Beats published EyeFusionNet IEEE 2023 (89.2%). |
| **EXP-006** | 2026-08-27 | **BiomedCLIP + CBAM Dual Attention (Retrained on 70% Split)** | Fine-tuned Novel Architecture on 2,800 balanced images with CLAHE | 87.45% | 87.83% | 87.83% | 98.94% | 0.9447 | High Glaucoma sensitivity (65% recall, 63.93% F1). |
| **EXP-007** | 2026-08-27 | **Mega-Ensemble (EffNet-B3 384 + BiomedCLIP-CBAM) + 4-View TTA** | Blended High-Res CNN + Multimodal Foundation Model + Flip TTA | N/A | 90.50% | 90.44% | 0.9923 | 0.9704 | First ensemble crossing 90.5%. |
| **EXP-008** | 2026-08-27 | **ConvNeXt-Small + High-Res ($384\times 384$) + CLAHE** 🏆 | **Modern 7x7 Depthwise ConvNet, 50M params, 384x384, Focal Loss ($\gamma=2.0$)** | **88.16%** | **90.50%** | **90.48%** | **0.9902** | **0.9663** | 🏆 **NEW ALL-TIME HIGHEST SINGLE-MODEL SOTA! Glaucoma F1 jumped to 68.25% (71.67% sensitivity). CSCR (96.0%), DR (95.0%), Disc Edema (99.2%), RD (100%), Pterygium (100%).** |
| **EXP-009** | 2026-08-27 | **Triple-Threat Mega-Ensemble (ConvNeXt-Small + EffNet-B3 + BiomedCLIP-CBAM) + 4-View TTA** 🏆 | **Blended Heterogeneous Paradigms (Modern ConvNet + Mobile CNN + ViT) + Flip TTA** | N/A | **90.50%** | **90.40%** | **0.9929** | **0.9710** | 🏆 **PEAK ROC-AUC (99.29%) & HIGHEST KAPPA (0.9710). Diabetic Retinopathy F1 reached 95.87% (96.67% sensitivity).** |
