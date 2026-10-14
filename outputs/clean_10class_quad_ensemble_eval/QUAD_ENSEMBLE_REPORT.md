# Clean 10-Class Quad Ensemble Evaluation & Comparison Report (Pure FP32 Re-Verification)

## 1. Executive Summary & Verification Policy
This report presents the re-verified evaluation of the **Unified Quad Ensemble** on the duplicate/source-group audited validation partition (`data_clean/val`, $N=562$ images) across all 10 clinical eye disease classes.

All four constituent backbones were evaluated under a strictly uniform **pure FP32 inference policy** (matching their training configs with `use_amp: false` and disabling half-precision autocasting):
1. **ConvNeXt-Small** ($384\times 384$ + CLAHE, Crop Fundus) | Checkpoint SHA-256: `238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d`
2. **EfficientNet-B3** ($384\times 384$ + CLAHE, Crop Fundus) | Checkpoint SHA-256: `0a93355dbc07dd2c989a55901e5b0fb8281d51dbc836bb3f0e1eb3a7e00fc3aa`
3. **ResNet-50d** ($384\times 384$ + CLAHE, Crop Fundus) | Checkpoint SHA-256: `110a01d6b4544376bd26f6e21857d0307f5187a86ccc0cc11aca790caf25346f`
4. **BiomedCLIP + CBAM Fusion** ($224\times 224$, Conservative Aug) | Checkpoint SHA-256: `b323dfbe245e87508dd2e10a94104dcbee5c3e72325ea5f8167add9d7dee73aa`

100% concordance of image file paths, true ground-truth labels, and class column ordering was asserted across all 562 samples. Checkpoints and test set partitions remain untouched and quarantined.

---

## 2. Re-Verified Benchmark Comparison ($N=562$)

*Ensemble weight search objective: Direct Macro-F1 maximization across the simplex, with accuracy used strictly as a tie-breaker.*

| Architecture / Ensemble Configuration | Status | Normalized Simplex Weights $[ \text{Conv, Eff, Res, Bio} ]$ | Validation Accuracy | Validation Balanced Accuracy | Validation Macro-F1 | Correct Samples ($N=562$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 Baseline** | Benchmark Reference | — | 87.01% | 87.62% | 87.41% | 489 / 562 |
| **BiomedCLIP + CBAM Fusion** | Standalone Component | — | 86.65% | 81.66% | 83.72% | 487 / 562 |
| **ResNet-50d** | Standalone Component | — | 87.01% | 85.85% | 84.57% | 489 / 562 |
| **EfficientNet-B3** | Standalone Component | — | 87.54% | 88.46% | 86.47% | 492 / 562 |
| **ConvNeXt-Small** | **Standalone Leader** | — | **89.50%** | **90.12%** | **90.18%** | **503 / 562** |
| **Quad Ensemble: Original Documented Weights** | Historical Baseline | $[0.143, 0.429, 0.286, 0.143]$ | 88.79% | 89.18% | 87.91% | 499 / 562 |
| **Quad Ensemble: Predeclared Equal Weights** | **Candidate Ensemble** | $[0.250, 0.250, 0.250, 0.250]$ | **89.68%** | 89.36% | **89.58%** | **504 / 562** |
| **Quad Ensemble: 5-Fold Cross-Validated (OOF)** | Internal Exploratory Analysis | Mean: $[0.500, 0.093, 0.117, 0.289]$ | 89.50% | 89.73% | 89.64% | 503 / 562 |
| **Quad Ensemble: Full-Validation Optimized** | **Exploratory Only** | $[0.500, 0.000, 0.175, 0.325]$ | *90.57%* | *91.13%* | *90.94%* | *509 / 562* |

---

## 3. Scientific Analysis of Findings

### A. Defensible Standalone Finding
* **ConvNeXt-Small** remains the leading standalone model on the audited validation split, achieving **90.18% Macro-F1** and **89.50% Accuracy** (503 / 562 correct).

### B. Predeclared Equal-Weight Ensemble vs. ConvNeXt-Small
* **Accuracy Concordance**: ConvNeXt-Small correctly classifies 503 samples ($89.50\%$), while the Predeclared Equal-Weight Quad correctly classifies 504 samples ($89.68\%$, $+1$ net correct sample).
* **Macro-F1 Dynamic**: Equal-Weight Quad achieves **89.58% Macro-F1** vs. **90.18%** for ConvNeXt-Small ($-0.60$ pp). The equal-weight quad does **not** establish an overall improvement over ConvNeXt-Small.
* **Statistical Significance**: McNemar's paired test yields $p = 1.0000$ (10 ConvNeXt-only correct vs. 11 Quad-only correct; $\chi^2_{\text{continuity}} = 0.0000$), demonstrating that overall accuracy is statistically indistinguishable.
* **Clinical Trade-off**:
  * *Glaucoma Rescue*: Equal-Weight Quad surges Glaucoma F1 from **83.27% to 86.53% (+3.26 pp)**, correctly diagnosing 6 difficult glaucoma cases where ConvNeXt failed.
  * *Myopia / Macular Scar*: ConvNeXt's sharp high-frequency edge filters perform better on thin myopic crescents and scar textures (Myopia F1: 78.38% vs 74.29%).

### C. Corrected Weight Selection & 5-Fold Cross-Validation
* Under direct Macro-F1 maximization across 177,041 unique simplex weight vectors with accuracy tie-breaking:
  * Full-validation in-sample optimization selects $[0.500, 0.000, 0.175, 0.325]$, yielding **90.94% Macro-F1** and 509/562 correct ($90.57\%$). This is strictly **Exploratory Only** due to in-sample selection bias.
  * 5-Fold Stratified Cross-Validation produces an aggregated Out-of-Fold (OOF) performance of **89.64% Macro-F1** and **89.50% Accuracy** (503/562 correct).
  * **Critical Caveat**: This 5-fold cross-validation is labelled as **Internal Exploratory Analysis** because the underlying backbone checkpoints (`best_model.pth`) were already selected based on peak validation Macro-F1 on this exact same 562-image cohort during training.

### D. Dataset Provenance: Duplicate/Source-Group Audited (Not Complete Leak-Free Proof)
* While `data_clean/` has undergone rigorous deduplication (DINOv2 embedding cosine similarity, perceptual hash, exact SHA-256) and source-group clustering to eliminate near-duplicates, **complete patient-level leakage elimination cannot be claimed**.
* Because the original public data source lacked patient-level identifiers, multiple fundus photographs from the same patient (e.g., bilateral imaging, temporal follow-up) cannot be mathematically verified as independent across partitions. The dataset is therefore accurately characterized as **duplicate/source-group audited**.

### E. Relationship to Historical Leaky Benchmark (91.83%)
* The historical Quad Ensemble reported **91.83% accuracy** on `data/test` ($N=600$).
* That historical result was obtained on an uncleaned split known to contain patient-identity cross-contamination and duplicated fundus images.
* Therefore, the clean validation results (**89.68% equal weights**, **89.64% cross-validated OOF**, **90.18% standalone ConvNeXt**) cannot be claimed as proving superiority over the historical 91.83% score. They represent fundamentally different experimental regimes under CLAIM 2024 reporting standards.

---

## 4. Re-Verified Per-Class Diagnostic Profile ($N=562$)

| Disease Class | Support | ConvNeXt | EffNet-B3 | ResNet-50d | BiomedCLIP | Equal Weights | 5-Fold CV (OOF) | Exploratory (Full Val) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Central Serous Chorioretinopathy | 11 | **86.96%** | 76.92% | 72.00% | 70.00% | 83.33% | 83.33% | **86.96%** |
| Diabetic Retinopathy | 191 | 96.88% | 95.51% | 96.35% | 96.37% | **97.14%** | **97.14%** | **97.14%** |
| Disc Edema | 15 | **100.00%** | 93.33% | 93.33% | 74.07% | **100.00%** | **100.00%** | **100.00%** |
| **Glaucoma** | **124** | 83.27% | 85.37% | 85.48% | 83.33% | **86.53%** | 83.95% | 85.71% |
| Healthy | 105 | **88.48%** | 84.55% | 83.84% | 84.35% | 86.61% | 88.18% | **88.99%** |
| Macular Scar | 47 | 80.46% | 74.36% | 70.27% | 76.40% | 80.49% | 82.35% | **83.72%** |
| Myopia | 37 | 78.38% | 76.32% | 72.22% | 69.57% | 74.29% | 73.97% | **79.45%** |
| Pterygium *(Severe Scarcity: N=2)* | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Retinal Detachment | 13 | **96.00%** | 88.89% | 85.71% | 91.67% | **96.00%** | **96.00%** | **96.00%** |
| Retinitis Pigmentosa | 17 | **91.43%** | 89.47% | 86.49% | **91.43%** | **91.43%** | **91.43%** | **91.43%** |
| **Validation Macro-F1** | **562** | **90.18%** | **86.47%** | **84.57%** | **83.72%** | **89.58%** | **89.64%** | ***90.94%*** |
| **Validation Accuracy** | **562** | **89.50%** | **87.54%** | **87.01%** | **86.65%** | **89.68%** | **89.50%** | ***90.57%*** |
| **Correct Predictions** | **562** | **503** | **492** | **489** | **487** | **504** | **503** | ***509*** |
