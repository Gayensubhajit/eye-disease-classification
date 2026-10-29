# Clean 10-Class Benchmark Milestone v4 — Freeze & Audit Report

**Date of Audit:** October 21, 2026  
**Auditor:** Subhajit Gayen & Antigravity IDE  
**Dataset:** Audited Clean 10-Class Validation Cohort (`data_clean/val`, N=562)  
**Quarantine Status:** Independent Test Split (`data_clean/test`, N=563) strictly quarantined and unread.

---

## 1. Executive Summary & Certified Scoreboard

All metrics below are computed directly in pure FP32 from the verified, preserved validation predictions on `data_clean/val` ($N=562$).

### A. Active Clean Champions (Milestone v4)

| Model / Ensemble Configuration | Checkpoint Path | SHA-256 Checksum | Validation Split | N | Accuracy | Macro-F1 | Balanced Acc | Evaluation Precision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal-Weight Quad Ensemble** (Champion Ensemble) | Ensemble of 4 Checkpoints Below | `[0.25, 0.25, 0.25, 0.25]` | `data_clean/val` | 562 | **89.68%** (504/562) | **89.58%** | **89.36%** | Pure FP32 |
| **ConvNeXt-Small** (Champion Standalone) | `outputs/clean_10class_convnext_small_384_clean/best_model.pth` | `238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d` | `data_clean/val` | 562 | **89.50%** (503/562) | **90.18%** | **90.12%** | Pure FP32 |

### B. Constituent Backbones of the Quad Ensemble

| Constituent Backbone | Input Resolution & Preprocessing | Checkpoint Path | SHA-256 Checksum | Standalone Acc | Standalone Macro-F1 | Standalone Bal Acc |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **ConvNeXt-Small** | 384x384 CLAHE + Crop Fundus | `outputs/clean_10class_convnext_small_384_clean/best_model.pth` | `238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d` | 89.50% (503/562) | 90.18% | 90.12% |
| **EfficientNet-B3** | 384x384 CLAHE + Crop Fundus | `outputs/clean_10class_efficientnet_b3_384_clean/best_model.pth` | `0a93355dbc07dd2c989a55901e5b0fb8281d51dbc836bb3f0e1eb3a7e00fc3aa` | 87.54% (492/562) | 86.47% | 88.46% |
| **ResNet-50d** | 384x384 CLAHE + Crop Fundus | `outputs/clean_10class_resnet50d_384_clean/best_model.pth` | `110a01d6b4544376bd26f6e21857d0307f5187a86ccc0cc11aca790caf25346f` | 87.01% (489/562) | 84.57% | 85.85% |
| **BiomedCLIP + CBAM** | 224x224 Conservative Transform | `outputs/clean_10class_biomedclip_cbam_fusion/best_model.pth` | `b323dfbe245e87508dd2e10a94104dcbee5c3e72325ea5f8167add9d7dee73aa` | 86.65% (487/562) | 83.72% | 81.66% |

### C. Separate Historical Benchmark (Uncleaned Split, CLAIM 2024 Non-Comparable)

| Historical Benchmark | Dataset Split | Sample N | Accuracy | Macro-F1 | Status / Caveat |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Historical Quad Ensemble** | Original Uncleaned `data/test` | 600 | 91.83% | 91.78% | **Superseded & Non-Comparable.** Contains duplicate fundus images and unverified patient cross-contamination. |

---

## 2. Certified Per-Class Metrics Table

Computed from `outputs/clean_10class_quad_ensemble_eval/val_predictions_equal_weights.csv` and `outputs/clean_10class_convnext_small_384_clean/val_predictions_reconciled.csv`.

| Clinical Class Name | Validation Support ($N$) | Quad Precision | Quad Recall | Quad Specificity | Quad F1 | ConvNeXt Precision | ConvNeXt Recall | ConvNeXt Specificity | ConvNeXt F1 | Delta (Quad vs ConvNeXt F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Central Serous Chorioretinopathy** | 11 | 76.92% | 90.91% | 99.46% | 83.33% | 83.33% | 90.91% | 99.64% | 86.96% | -3.63 pp |
| **Diabetic Retinopathy** | 191 | 96.39% | 97.91% | 98.11% | 97.14% | 96.37% | 97.38% | 98.11% | 96.88% | +0.26 pp |
| **Disc Edema** | 15 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 pp |
| **Glaucoma** | 124 | **87.60%** | **85.48%** | **96.58%** | **86.53%** | 84.30% | 82.26% | 95.66% | 83.27% | **+3.26 pp** |
| **Healthy** | 105 | 81.51% | 92.38% | 95.19% | 86.61% | 85.71% | 91.43% | 96.50% | 88.48% | -1.87 pp |
| **Macular Scar** | 47 | 94.29% | 70.21% | 99.61% | 80.49% | 87.50% | 74.47% | 99.03% | 80.46% | +0.03 pp |
| **Myopia** | 37 | 78.79% | 70.27% | 98.67% | 74.29% | 78.38% | 78.38% | 98.48% | 78.38% | -4.09 pp |
| **Pterygium** | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 pp |
| **Retinal Detachment** | 13 | 100.00% | 92.31% | 100.00% | 96.00% | 100.00% | 92.31% | 100.00% | 96.00% | 0.00 pp |
| **Retinitis Pigmentosa** | 17 | 88.89% | 94.12% | 99.63% | 91.43% | 88.89% | 94.12% | 99.63% | 91.43% | 0.00 pp |
| **Macro Average** | **562** | **90.44%** | **89.36%** | **98.72%** | **89.58%** | **90.45%** | **90.13%** | **98.71%** | **90.18%** | **-0.60 pp** |

---

## 3. Discrepancy Reconciliation & Mutually Exclusive Error Audit

### A. Correction of Previous Error Reporting Discrepancy
In earlier conversational notes, central-macula confusion was cited as accounting for "14 errors" while itemizing only 12 cases.
**Audit Finding:** The true total misclassifications for ground-truth Macular Scar is **exactly 14 errors**. The previous conversational summary accidentally omitted two single-occurrence confusion pairs:
- `Macular Scar -> Myopia: 1`
- `Macular Scar -> Retinitis Pigmentosa: 1`
With these two verified from `val_predictions_equal_weights.csv`, the sum equals exactly 14.

### B. Mutually Exclusive Error Partitioning (No Overlapping Categories)
To prevent cross-category double counting (e.g., miscounting `Macular Scar -> Glaucoma` as both a macular error and an optic nerve error), all 58 errors are strictly partitioned by their **True Ground-Truth Class**:

```
Total Quad Ensemble Errors: Exactly 58 / 562 (10.32%)

1. True Glaucoma (18 errors / 31.0% of all errors):
   ├── Glaucoma -> Healthy:                                  14
   ├── Glaucoma -> Myopia:                                    3
   └── Glaucoma -> Macular Scar:                              1

2. True Macular Scar (14 errors / 24.1% of all errors):
   ├── Macular Scar -> Diabetic Retinopathy:                  5
   ├── Macular Scar -> Glaucoma:                              3
   ├── Macular Scar -> Central Serous Chorioretinopathy:      2
   ├── Macular Scar -> Healthy:                               2
   ├── Macular Scar -> Myopia:                                1
   └── Macular Scar -> Retinitis Pigmentosa:                  1

3. True Myopia (11 errors / 19.0% of all errors):
   ├── Myopia -> Glaucoma:                                    5
   ├── Myopia -> Healthy:                                     5
   └── Myopia -> Macular Scar:                                1

4. True Healthy (8 errors / 13.8% of all errors):
   ├── Healthy -> Glaucoma:                                   6
   ├── Healthy -> Central Serous Chorioretinopathy:           1
   └── Healthy -> Myopia:                                     1

5. True Diabetic Retinopathy (4 errors / 6.9% of all errors):
   ├── Diabetic Retinopathy -> Glaucoma:                      1
   ├── Diabetic Retinopathy -> Healthy:                       1
   ├── Diabetic Retinopathy -> Myopia:                        1
   └── Diabetic Retinopathy -> Retinitis Pigmentosa:          1

6. True Central Serous Chorioretinopathy (1 error / 1.7%):
   └── CSCR -> Diabetic Retinopathy:                          1

7. True Retinal Detachment (1 error / 1.7%):
   └── Retinal Detachment -> Myopia:                          1

8. True Retinitis Pigmentosa (1 error / 1.7%):
   └── Retinitis Pigmentosa -> Diabetic Retinopathy:          1

9. True Disc Edema:                                           0 errors (100% precision & recall)
10. True Pterygium:                                           0 errors (100% precision & recall)

Top 4 Disease Classes: Glaucoma (18) + Macular Scar (14) + Myopia (11) + Healthy (8) = 51 / 58 errors (87.93%).
Remaining 6 Classes Combined: DR (4) + CSCR (1) + RD (1) + RP (1) + Disc Edema (0) + Pterygium (0) = 7 / 58 errors (12.07%).
Total Sum: Exactly 58 errors across 562 validation samples.
```

### C. Scientific Caution on Causal Inferences
*While class confusions are mathematically exact, **confusion counts alone do not prove anatomical causation** (e.g., cup-to-disc ratio or foveal texture) without pixel-level localization, saliency/Grad-CAM validation, or expert ophthalmologist confirmation.*

---

## 4. Audit of Ensemble Fusion-Weight Code & Selection Status

1. **Fixed Equal Weights `[0.25, 0.25, 0.25, 0.25]` (Predeclared Primary Champion):**
   - Requires zero validation label optimization.
   - Completely parameter-free and immune to post-processing over-adaptation.
   - Correct count: 504/562 (89.68% accuracy, 89.58% Macro-F1).
2. **5-Fold Cross-Validated OOF Weights `[0.500, 0.093, 0.117, 0.289]` (Internal Exploratory):**
   - Correct count: 503/562 (89.50% accuracy, 89.64% Macro-F1).
   - Under direct simplex optimization, the out-of-fold performance matches ConvNeXt alone (503 samples).
3. **Full-Validation In-Sample Optimization `[0.500, 0.000, 0.175, 0.325]` (Exploratory Ceiling Only):**
   - Yields in-sample 509/562 (90.57% accuracy, 90.94% Macro-F1).
   - **Audit Statement:** This score is strictly an in-sample upper bound. It is not an evaluation metric and must never be reported as a validated benchmark score.

---

## 5. Explicit Limitations & Governance Record

1. **Lack of Patient-Level Identifiers:**
   The original public data sources do not provide de-identified patient IDs. While perceptual hashing, DINOv2 embedding similarity, and exact SHA-256 deduplication eliminated identical and near-identical images across splits, potential bilateral imaging (left and right eyes of the same subject in train and val) cannot be verified.
2. **Prior Model-Selection Exposure:**
   All 4 backbone models selected their `best_model.pth` checkpoints based on validation performance during training. Exploratory post-processing on this same cohort must be interpreted with caution.
3. **Severe Class Imbalance on Minority Classes:**
   Validation sample sizes for certain pathologies are very small (Pterygium $N=2$, CSCR $N=11$, Retinal Detachment $N=13$, Disc Edema $N=15$). A single misclassification in these classes causes large percentage swings in F1.

---

## 6. Scientific Recommendation on Swin Transformer Experiment

### Complementary Error Analysis:
- Total validation errors across the 4 constituent models:
  - ConvNeXt: 59 errors
  - EfficientNet: 70 errors
  - ResNet: 73 errors
  - BiomedCLIP: 75 errors
- **Shared Failure Modes:** Only **25 out of 562 samples** are misclassified by all 4 models simultaneously. The theoretical oracle ceiling across the 4 models is **537/562 (95.55%)**.
- **Weakest Backbone:** ResNet-50d is the weakest constituent (73 errors, 84.57% Macro-F1) and shares the highest error correlation with EfficientNet-B3 (47 shared errors, Jaccard 0.490).
- **Recommendation:**
  A **Swin-Transformer (`swin_base_patch4_window12_384`) experiment is scientifically justified** solely as an architectural replacement candidate for ResNet-50d to break the shared error correlation between the CNN backbones.
  *Decision Protocol:* The Swin-Transformer must be evaluated on `data_clean/val` using the exact same FP32 protocol and original augmentations. It will only be adopted if it provides complementary non-correlated predictions that improve the fixed ensemble beyond 504/562 correct without compromising Glaucoma recall.
