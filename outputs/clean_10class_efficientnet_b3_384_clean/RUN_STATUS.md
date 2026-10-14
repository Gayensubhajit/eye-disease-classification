# Clean 10-Class Benchmark Run Status: EfficientNet-B3 (384×384 CLAHE)

## 1. Run Summary & Provenance
- **Model Architecture**: `efficientnet_b3` (pretrained compound-scaled CNN)
- **Input Resolution**: $384 \times 384$ pixels
- **Preprocessing**: Circular Fundus Crop + CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Objective Loss**: Unweighted Multi-Class Cross-Entropy
- **Epoch Budget**: 20 uninterrupted epochs (completed in 46.52 minutes)
- **Random Seed**: 42 (deterministic CuBLAS and PyTorch configuration)
- **Batch Size**: 4
- **Initial Learning Rate**: $3 \times 10^{-4}$ (Cosine annealing to minimum $1.85 \times 10^{-6}$ over 20 epochs)
- **Weight Decay**: $1 \times 10^{-4}$
- **Test Set Quarantine**: Strictly enforced (`evaluate_test: false`, `--no-test-eval`; test partition remains untouched)
- **Checkpoint Location**: `outputs/clean_10class_efficientnet_b3_384_clean/best_model.pth`
- **Checkpoint SHA-256**: `0a93355dbc07dd2c989a55901e5b0fb8281d51dbc836bb3f0e1eb3a7e00fc3aa`
- **Best Validation Epoch**: Epoch 15 (selected by peak validation Macro-F1: 86.47% during training)

---

## 2. Learning-Rate Schedule Audit
The learning rate was recorded at every epoch and evaluated against the theoretical cosine annealing formulation:
$$\text{LR}_t = \frac{1}{2} \text{LR}_{\text{base}} \left(1 + \cos\left(\frac{(t-1) \pi}{T_{\max}}\right)\right)$$
- **Observed Base LR**: $0.00030000$ (Epoch 1)
- **Observed Final LR**: $0.00000185$ (Epoch 20)
- **Maximum Absolute Discrepancy**: $5.42 \times 10^{-20}$
- **Audit Result**: **PERFECT COSINE ANNEALING CONVERGENCE** (zero scheduler-resume anomalies or step skews).

---

## 3. Deterministic Validation Reconciliation Metrics ($N=562$)
Evaluated via `scripts/reconcile_val_artifacts.py` on the leak-free validation partition `data_clean/val`:

| Metric | Reconciled Value | Mathematical Identity Verification |
| :--- | :--- | :--- |
| **Validation Overall Accuracy** | **87.37%** (491 / 562) | $\sum \text{diag}(\text{CM}) / 562 = 491 / 562$ (Exact match) |
| **Validation Balanced Accuracy** | **88.37%** | $\frac{1}{10} \sum \text{Recall}_c$ (Exact match) |
| **Validation Macro-F1** | **86.32%** | $\frac{1}{10} \sum \text{F1}_c$ (Exact match) |
| **Validation Macro-Precision** | **85.58%** | $\frac{1}{10} \sum \text{Precision}_c$ (Exact match) |
| **Validation Macro-Specificity** | **98.47%** | $\frac{1}{10} \sum \text{Specificity}_c$ (Exact match) |

---

## 4. 3-Way Per-Class Validation Comparison

| Disease Class | Support | EffNet-B0 Baseline F1 | ConvNeXt-Small Clean F1 | EffNet-B3 Clean F1 | B3 vs B0 (pp) | B3 vs Conv (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Central Serous Chorioretinopathy [Color Fundus] | 11 | 80.00% | 86.96% | 76.92% | -3.08% | -10.03% |
| Diabetic Retinopathy | 191 | 96.06% | 96.88% | 95.51% | -0.55% | -1.36% |
| Disc Edema | 15 | 96.55% | 100.00% | 93.33% | -3.22% | -6.67% |
| Glaucoma | 124 | 80.34% | 83.27% | **85.37%** | **+5.02%** | **+2.10%** |
| Healthy | 105 | 84.35% | 88.48% | 84.02% | -0.33% | -4.46% |
| Macular Scar | 47 | 79.55% | 80.46% | 74.36% | -5.19% | -6.10% |
| Myopia | 37 | 72.97% | 78.38% | 75.32% | **+2.35%** | -3.05% |
| Pterygium | 2 | 100.00% | 100.00% | 100.00% | +0.00% | +0.00% |
| Retinal Detachment | 13 | 96.00% | 96.00% | 88.89% | -7.11% | -7.11% |
| Retinitis Pigmentosa | 17 | 88.24% | 91.43% | **89.47%** | **+1.24%** | -1.95% |
| **Validation Macro-F1** | **562** | **87.41%** | **90.18%** | **86.32%** | **-1.09%** | **-3.86%** |
| **Validation Balanced Accuracy**| **562** | **87.62%** | **90.12%** | **88.37%** | **+0.75%** | **-1.76%** |
| **Validation Overall Accuracy** | **562** | **87.01%** | **89.50%** | **87.37%** | **+0.36%** | **-2.14%** |
| **Correct Predictions** | **562** | **489** | **503** | **491** | **+2** | **-12** |

---

## 5. Key Clinical Insights & Diagnostic Trade-offs
1. **Glaucoma Leadership**: EfficientNet-B3 achieves the strongest Glaucoma score across all tested backbones:
   - Recall: **84.68%** (105 / 124 detected, vs 75.81% in B0 and 82.26% in ConvNeXt).
   - Precision: **86.07%** (105 / 122).
   - F1-Score: **85.37%** (+5.02 pp over B0, +2.10 pp over ConvNeXt).
   - *Clinical Interpretation*: High resolution (384×384) + CLAHE allows compound scaling to resolve the neuroretinal rim thinning and subtle cup-to-disc ratio alterations critical for glaucoma detection.
2. **Retinitis Pigmentosa Sensitivity**:
   - Recall reached **100.00%** (17 / 17 detected), outperforming B0 (88.24%) and ConvNeXt (94.12%).
3. **Macular Scar & Retinal Detachment Compression**:
   - Macular Scar recall dropped to 61.70% (29/47, with 6 confused as Diabetic Retinopathy and 3 as CSCR), though precision remained high at 93.55%.
   - Retinal Detachment maintained 92.31% recall (12/13), but precision decreased to 85.71% due to 2 false positives.
4. **Pterygium Statistical Scarcity**:
   - Validation support is only $N=2$. The 100% score (2/2 detected) is preserved across all models, but provides negligible statistical evidence for generalization.

