# Clean 10-Class Benchmark & Quad-Architecture Ensemble Evaluation Report
**Authoritative Final Re-Verified Results under CLAIM 2024 / STARD Guidelines**

---

## Executive Summary

This report establishes the final, re-verified comparative analysis between our leading standalone architectures and multi-backbone ensembles on the **duplicate/source-group audited** validation partition (`data_clean/val`, $N=562$ images) across all 10 clinical eye disease classes.

### Verification Standards Enforced
1. **Strict Pure FP32 Numerical Policy**: All backbone inference passes were re-executed using pure 32-bit floating-point precision (`use_amp: false`, `torch.cuda.amp.autocast` disabled) matching training and eliminating numerical drift.
2. **Deterministic Data Concordance**: 100% concordance of image file paths, true ground-truth labels, and class column ordering was asserted across all 562 validation samples.
3. **Quarantine Preserved**: Test set evaluation (`data_clean/test`, $N=563$) and all further model training remain strictly paused.
4. **Corrected Optimization Objective**: Ensemble weight searches directly maximize Macro-F1 across the simplex, utilizing accuracy strictly as a tie-breaker.

---

## 1. Unified Benchmark Performance Summary ($N=562$)

| Architecture / Ensemble Strategy | Model Category | Simplex Weights $[ \text{Conv, Eff, Res, Bio} ]$ | Accuracy | Balanced Accuracy | Macro-F1 | Correct Samples | Interpretation & Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BiomedCLIP + CBAM Fusion** | Standalone Component | — | 86.65% | 81.66% | 83.72% | 487 / 562 | Zero-shot clinical prior fusion |
| **ResNet-50d** | Standalone Component | — | 87.01% | 85.85% | 84.57% | 489 / 562 | Strong multi-scale feature backbone |
| **EfficientNet-B3** | Standalone Component | — | 87.54% | 88.46% | 86.47% | 492 / 562 | High Glaucoma specificity (85.37% F1) |
| **ConvNeXt-Small** | **Standalone Leader** | — | **89.50%** | **90.12%** | **90.18%** | **503 / 562** | **Primary Defensible Standalone Model** |
| **Quad: Original Documented Weights** | Historical Baseline | $[0.143, 0.429, 0.286, 0.143]$ | 88.79% | 89.18% | 87.91% | 499 / 562 | Sub-optimal (-2.27 pp F1 vs ConvNeXt) |
| **Quad: Predeclared Equal Weights** | **Candidate Ensemble** | $[0.250, 0.250, 0.250, 0.250]$ | **89.68%** | 89.36% | **89.58%** | **504 / 562** | **Higher Accuracy (+1 sample); Glaucoma +3.26 pp** |
| **Quad: 5-Fold Cross-Validated (OOF)** | Internal Exploratory Analysis | Mean: $[0.500, 0.093, 0.117, 0.289]$ | 89.50% | 89.73% | 89.64% | 503 / 562 | Corrected direct Macro-F1 selection |
| **Quad: Full-Validation Optimized** | **Exploratory Only** | $[0.500, 0.000, 0.175, 0.325]$ | *90.57%* | *91.13%* | *90.94%* | *509 / 562* | *In-Sample Selection Optimism* |

---

## 2. Granular Per-Class Diagnostic Profile ($N=562$)

| Disease Category | Cohort Support | ConvNeXt-Small | EffNet-B3 | ResNet-50d | BiomedCLIP | Equal-Weight Quad (Candidate) | 5-Fold CV (OOF Exploratory) | Exploratory In-Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Central Serous Chorioretinopathy | 11 | **86.96%** | 76.92% | 72.00% | 70.00% | 83.33% | 83.33% | **86.96%** |
| Diabetic Retinopathy | 191 | 96.88% | 95.51% | 96.35% | 96.37% | **97.14%** | **97.14%** | **97.14%** |
| Disc Edema | 15 | **100.00%** | 93.33% | 93.33% | 74.07% | **100.00%** | **100.00%** | **100.00%** |
| **Glaucoma** | **124** | 83.27% | 85.37% | 85.48% | 83.33% | **86.53%** | 83.95% | 85.71% |
| Healthy | 105 | **88.48%** | 84.55% | 83.84% | 84.35% | 86.61% | 88.18% | **88.99%** |
| Macular Scar | 47 | 80.46% | 74.36% | 70.27% | 76.40% | 80.49% | 82.35% | **83.72%** |
| Myopia | 37 | **78.38%** | 76.32% | 72.22% | 69.57% | 74.29% | 73.97% | **79.45%** |
| Pterygium *(Severe Scarcity Caution)* | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Retinal Detachment | 13 | **96.00%** | 88.89% | 85.71% | 91.67% | **96.00%** | **96.00%** | **96.00%** |
| Retinitis Pigmentosa | 17 | **91.43%** | 89.47% | 86.49% | **91.43%** | **91.43%** | **91.43%** | **91.43%** |
| **Validation Macro-F1** | **562** | **90.18%** | **86.47%** | **84.57%** | **83.72%** | **89.58%** | **89.64%** | ***90.94%*** |
| **Validation Accuracy** | **562** | **89.50%** | **87.54%** | **87.01%** | **86.65%** | **89.68%** | **89.50%** | ***90.57%*** |
| **Correct Predictions** | **562** | **503** | **492** | **489** | **487** | **504** | **503** | ***509*** |

---

## 3. Methodological & Rigorous Scientific Limitations

To maintain the highest standard of scientific integrity for publication (conforming to CLAIM 2024 and STARD guidelines), the following four limitations must be explicitly articulated:

### Limitation 1: Checkpoint-Selection Overlap in Cross-Validation (Internal Exploratory Analysis)
* 5-fold cross-validation of ensemble weights on `data_clean/val` isolates the weight-fitting step across folds, yielding an aggregated Out-of-Fold (OOF) Macro-F1 of **89.64%** (and accuracy of 89.50%, 503/562 correct).
* However, **this remains an internal exploratory analysis rather than an independent confirmation of the modeling pipeline**.
* The individual backbone checkpoints (`best_model.pth` for ConvNeXt, EffNet-B3, ResNet-50d, and BiomedCLIP) were previously chosen based on peak validation performance across this exact same 562-image cohort during training. A completely independent evaluation would require nested cross-validation trained from scratch or a strictly quarantined, pre-registered test partition.

### Limitation 2: Cleaned Test Set Prior Exposure
* The cleaned test partition (`data_clean/test`, $N=563$) was evaluated during earlier exploratory baseline iterations and initial data-leakage auditing prior to protocol locking.
* Under strict pre-registration standards, it cannot now be framed as an untouched, virgin confirmatory test partition. Any future evaluation on `data_clean/test` must be reported with explicit disclosure of prior model exposure.

### Limitation 3: Dataset Characterization as Duplicate/Source-Group Audited (Not Complete Leak-Free Proof)
* While `data_clean/` has undergone thorough multi-metric deduplication (removing identical files, high-similarity DINOv2 vision representations, and conflicting source labels), **complete patient-level leakage elimination cannot be verified**.
* The original Mendeley/Zenodo dataset releases lacked clinical metadata and patient IDs. Therefore, eyes belonging to the same subject that exhibit distinct photographic angles, illuminations, or disease stages may still span partition boundaries. The dataset must be described as **duplicate/source-group audited** rather than claiming absolute absence of data leakage.

### Limitation 4: Strict Non-Comparability with Historical Leaky Quad (91.83%)
* The historical Quad Ensemble reported **91.83% accuracy** on `data/test` ($N=600$).
* That historical result was obtained on an uncleaned partition with documented cross-set image duplicates and patient identity overlap.
* Therefore, the clean benchmark results (**89.68% Equal-Weight Quad**, **89.64% Cross-Validated Quad**, **90.18% ConvNeXt-Small**) must not be claimed as proving superiority over the historical 91.83% score, nor should the historical score be treated as a clinically valid metric. They represent fundamentally different experimental regimes.

---

## 4. Defensible Findings & Manuscript Synthesis

The empirical evidence supports two primary findings:

1. **Standalone Leader**:
   * **ConvNeXt-Small** leads all standalone models with **90.18% validation Macro-F1** and **89.50% accuracy** (503 / 562 correct).
   * It provides the strongest single-model balance across both common and fine-texture retinal pathologies (CSCR: 86.96%, Myopia: 78.38%, Healthy: 88.48%).

2. **Predeclared Equal-Weight Quad Ensemble**:
   * Achieves **89.68% accuracy** (504 / 562 correct) and **89.58% Macro-F1**.
   * While its 89.58% Macro-F1 does **not** establish an overall improvement over ConvNeXt-Small (90.18%), the equal-weight quad serves as a useful candidate consensus architecture that substantially bolsters sensitivity on sight-threatening **Glaucoma** (F1 surges from 83.27% to **86.53%**, +3.26 pp; rescuing 6 false negatives).
   * McNemar's paired test ($p = 1.0000$) confirms that overall accuracy between ConvNeXt-Small and the Equal-Weight Quad is statistically indistinguishable.

