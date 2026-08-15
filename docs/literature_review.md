# Literature Review & State-of-the-Art (SOTA) Benchmarks

This document tracks published peer-reviewed studies, competitive benchmarks, and foundation model literature for 10-class fundus eye disease classification.

| Paper ID | Authors / Venue | Architecture | Pretraining / Paradigm | Balanced Acc | Macro F1 | ROC-AUC | Key Contribution & Notes |
|:---:|---|---|---|:---:|:---:|:---:|---|
| **LIT-001** | Rashid et al. (Mendeley 2021) | ResNet-50 / DenseNet-121 | ImageNet Transfer Learning | ~83.4% | ~82.8% | 0.965 | Baseline reference for 10-class Mendeley eye disease dataset. |
| **LIT-002** | PLoS ONE (2023) | Swin Transformer (Swin-T) | Shifted-Window Vision Transformer | **86.8%** | 86.2% | 0.978 | Hierarchical attention captures global fundus context and vascular lesions. |
| **LIT-003** | EyeFusionNet (IEEE 2023) | DenseNet-169 + TNT | CNN-Transformer Dual-Branch Fusion | **89.2%** | 88.7% | 0.984 | SOTA on this dataset. Fuses local CNN features with patch self-attention. |
| **LIT-004** | RETFound (Nature 2023) | ViT-Large/16 Masked Autoencoder | Self-Supervised on 1.6M Retinal Scans (Fundus+OCT) | **88.5% - 91.2%** | 89.4% | 0.988 | Moorfields/UCL landmark retinal foundation model. High generalizability. |
| **LIT-005** | BiomedCLIP (Microsoft / EMNLP 2023) | ViT-Base/16 + PubMedBERT | Contrastive Multimodal on 15M PubMed Images | **83.85%** | **83.69%** | **0.9802** | Evaluated in our project (EXP-002). Superior across all metrics over CNN baseline. |
| **LIT-006** | Our CNN Baseline (EXP-001) | EfficientNet-B0 + Focal Loss | ImageNet Pretrained + Cosine Annealing | **83.38%** | **83.11%** | **0.9774** | 100% physically balanced folder split (1,300 test images), 98.15% specificity. |

---

## Downloadable Artifacts
- **Publication-Ready PDF:** [`docs/Literature_Review_and_SOTA_Benchmarks.pdf`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/Literature_Review_and_SOTA_Benchmarks.pdf)
- **Detailed Research Report:** [`docs/literature_research_report.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/literature_research_report.md)
