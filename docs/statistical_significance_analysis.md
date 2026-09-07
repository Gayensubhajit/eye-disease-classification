# Empirical Statistical Rigor & Hypothesis Testing Report

> **Project:** Retinal Disease Classification System  
> **Institution:** Department of Information Technology, Jadavpur University  
> **Supervisor:** Dr. Pawan Kumar Singh  
> **Methodology:** 1,000-iteration Bootstrap 95% Confidence Intervals & McNemar's Paired Discordance Test (Two-Sided Exact Binomial)

---

## 1. 10-Class Unified Benchmark (600 Untouched Test Images)

### 95% Bootstrap Confidence Intervals

| Architecture | Test Accuracy (95% CI) | Macro F1-Score (95% CI) | Macro ROC-AUC (95% CI) | Cohen's $\kappa$ (95% CI) |
|---|:---:|:---:|:---:|:---:|
| ConvNeXt-Small 384 | **90.17%** [87.83%, 92.50%] | 90.15% [87.98%, 92.37%] | 0.9915 [0.9884, 0.9943] | 0.8907 [0.8646, 0.9165] |
| EfficientNet-B3 384 | **91.00%** [88.67%, 93.17%] | 90.97% [88.86%, 92.97%] | 0.9907 [0.9870, 0.9939] | 0.9000 [0.8740, 0.9239] |
| ResNet-50d 384 | **88.17%** [85.67%, 90.67%] | 88.17% [85.95%, 90.43%] | 0.9904 [0.9867, 0.9935] | 0.8685 [0.8405, 0.8961] |
| BiomedCLIP-CBAM 224 | **89.00%** [86.50%, 91.34%] | 88.98% [86.63%, 91.14%] | 0.9898 [0.9858, 0.9934] | 0.8778 [0.8497, 0.9037] |
| ViT-Base-384 | **89.17%** [86.50%, 91.67%] | 89.15% [86.78%, 91.38%] | 0.9910 [0.9879, 0.9940] | 0.8796 [0.8499, 0.9072] |
| **Unified Quad Ensemble (SOTA)** | **91.83%** [89.67%, 93.83%] | 91.78% [89.75%, 93.76%] | 0.9923 [0.9890, 0.9949] | 0.9093 [0.8850, 0.9314] |

### McNemar's Paired Significance Tests (Unified Quad Ensemble vs Standalone Backbones)

| Compared Model | Ensemble Only Wins ($b$) | Model Only Wins ($c$) | Continuity $\chi^2$ | Exact Two-Sided $p$-value | Statistical Significance ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-Small 384 | +13 | -3 | 5.06 | 2.1271e-02 | **$p < 0.05$ (Significant)** |
| EfficientNet-B3 384 | +6 | -1 | 2.29 | 1.2500e-01 | Not Significant ($p \ge 0.05$) |
| ResNet-50d 384 | +28 | -6 | 12.97 | 1.9513e-04 | **$p < 0.001$ (Highly Significant)** |
| BiomedCLIP-CBAM 224 | +26 | -9 | 7.31 | 5.9881e-03 | **$p < 0.01$ (Very Significant)** |
| ViT-Base-384 | +23 | -7 | 7.50 | 5.2229e-03 | **$p < 0.01$ (Very Significant)** |

---

## 2. 4-Class Kaggle Benchmark (423 Untouched Test Images)

### 95% Bootstrap Confidence Intervals

| Architecture | Test Accuracy (95% CI) | Macro F1-Score (95% CI) | Macro ROC-AUC (95% CI) | Cohen's $\kappa$ (95% CI) |
|---|:---:|:---:|:---:|:---:|
| BiomedCLIP + CBAM | **94.56%** [92.20%, 96.45%] | 94.49% [92.26%, 96.50%] | 0.9915 [0.9857, 0.9967] | 0.9275 [0.8959, 0.9527] |
| ConvNeXt-Small v2 | **94.33%** [91.96%, 96.69%] | 94.25% [91.84%, 96.47%] | 0.9928 [0.9878, 0.9967] | 0.9243 [0.8926, 0.9556] |
| ResNet-50d | **92.91%** [90.30%, 95.27%] | 92.78% [90.04%, 95.28%] | 0.9866 [0.9790, 0.9933] | 0.9054 [0.8700, 0.9369] |
| EfficientNet-B3 | **92.91%** [90.31%, 95.27%] | 92.73% [90.17%, 95.23%] | 0.9895 [0.9829, 0.9950] | 0.9054 [0.8707, 0.9369] |
| DenseNet-121 | **93.14%** [90.78%, 95.51%] | 92.98% [90.57%, 95.26%] | 0.9883 [0.9811, 0.9943] | 0.9085 [0.8764, 0.9398] |
| ViT-Base-384 | **93.14%** [90.54%, 95.51%] | 93.03% [90.53%, 95.32%] | 0.9904 [0.9845, 0.9954] | 0.9086 [0.8739, 0.9399] |
| **ResNet-Integrated Quad Ensemble (SOTA)** | **95.74%** [93.62%, 97.64%] | 95.70% [93.63%, 97.55%] | 0.9929 [0.9875, 0.9972] | 0.9432 [0.9148, 0.9684] |

### McNemar's Paired Significance Tests (ResNet-Integrated Quad Ensemble vs Standalone Backbones)

| Compared Model | Ensemble Only Wins ($b$) | Model Only Wins ($c$) | Continuity $\chi^2$ | Exact Two-Sided $p$-value | Statistical Significance ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|
| BiomedCLIP + CBAM | +8 | -3 | 1.45 | 2.2656e-01 | Not Significant ($p \ge 0.05$) |
| ConvNeXt-Small v2 | +6 | -0 | 4.17 | 3.1250e-02 | **$p < 0.05$ (Significant)** |
| ResNet-50d | +12 | -0 | 10.08 | 4.8828e-04 | **$p < 0.001$ (Highly Significant)** |
| EfficientNet-B3 | +15 | -3 | 6.72 | 7.5378e-03 | **$p < 0.01$ (Very Significant)** |
| DenseNet-121 | +13 | -2 | 6.67 | 7.3853e-03 | **$p < 0.01$ (Very Significant)** |
| ViT-Base-384 | +11 | -0 | 9.09 | 9.7656e-04 | **$p < 0.001$ (Highly Significant)** |