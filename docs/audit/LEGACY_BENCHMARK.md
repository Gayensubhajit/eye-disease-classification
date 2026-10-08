# Legacy Benchmark Documentation (Frozen Baseline)

**Project:** Classification of Eye Diseases from Color Fundus Images  
**Status:** **IMMUTABLE LEGACY EXPERIMENT**  
**Branch:** `main` (Archived commit `e5cf731`)  
**Active Work Branch:** `research-clean-split`  
**Date:** October 2026  

---

## 1. Executive Protocol Notice

This document preserves the exact methodological specifications, composition, performance figures, and known limitations of the **Legacy 4,000-Image Balanced Fundus Benchmark**.

In accordance with strict research integrity protocols:
- **No legacy data files are modified or deleted** (`data/train`, `data/val`, `data/test`).
- **All existing checkpoints, cached predictions, and evaluation scripts are preserved**.
- **All reported numbers (91.83% ensemble, 91.00% EfficientNet-B3, 96.00% conformal coverage) remain valid historical measurements of this specific benchmark as constructed.**
- All future data repair and un-augmented generalization experiments are isolated on the `research-clean-split` branch.

---

## 2. Dataset Construction & Augmentation Pipeline

### Construction History
The legacy dataset was synthesized via `scripts/build_balanced_folders.py` from the raw *Eye Disease Image Dataset* (Mendeley Data / Kaggle):
1. **Target:** 4,000 color fundus images partitioned into:
   - `data/train`: 2,800 images (280 per class across 10 classes, 70%)
   - `data/val`: 600 images (60 per class across 10 classes, 15%)
   - `data/test`: 600 images (60 per class across 10 classes, 15%)
2. **Original Class Imbalance:**
   - Glaucoma: 1,349 raw images
   - Healthy: 1,024 raw images
   - Myopia: 500 raw images
   - Diabetic Retinopathy: 382 raw images (in original selection pool)
   - Macular Scar: 288 raw images
   - Retinitis Pigmentosa: 139 raw images
   - Disc Edema: 127 raw images
   - Retinal Detachment: 125 raw images
   - Central Serous Chorioretinopathy (CSCR): 101 raw images
   - Pterygium: **17 raw images**
3. **Augmentation Procedure:**
   - Classes with fewer than 400 images were inflated using:
     - Pre-augmented images provided in the Kaggle upload.
     - Offline Albumentations transforms: Horizontal Flip ($p=0.5$), Vertical Flip ($p=0.5$), Random Rotate 90 ($p=0.5$), Affine scale/translation/rotation ($p=0.7$), ColorJitter ($p=0.5$).
   - **Methodological Defect:** Augmentation was applied to the combined pool *before* partitioning into train, val, and test. Consequently, synthetic twins of individual source retinas were distributed across splits.

---

## 3. Audited Duplication & Leakage Profile

As forensic analysis revealed (`docs/audit/PROJECT_AUDIT.md`, `docs/audit/leakage_audit.md`):

| Finding | Metric | Physical Impact |
|---|---|---|
| **Exact MD5 Duplicates** | **11 images** | Bit-for-bit identical files between Train and Test splits |
| **Exact MD5 Val Duplicates** | **6 images** | Bit-for-bit identical files between Train and Val splits |
| **Perceptual dHash Near-Duplicates** | **86 images (14.3%)** | Test images sharing visual features with Train images |
| **Minority Class Contamination** | **15% – 22%** | Pterygium (21.7%), Disc Edema (20.0%), CSCR (16.7%) |
| **Patient-Level Isolation** | **NOT VERIFIED** | Filenames are anonymized (`img_0000.jpg`); no patient MRNs or laterality |

### Consequence on Minority Class Metrics
This pre-split synthetic duplication directly explains why five minority classes (Pterygium, CSCR, Disc Edema, Retinal Detachment, Retinitis Pigmentosa) exhibited **100.0% sensitivity** across multiple backbones. The models evaluated test images that shared anatomical features with images seen during training.

---

## 4. Frozen Legacy Experimental Results

These results represent the audited baseline on the constructed 4,000-image dataset:

### A. Standalone Backbones (600 Held-Out Test Images)
- **EfficientNet-B3:** 91.00% (546/600), Macro-F1: 90.96%
- **ConvNeXt-Small:** 89.17% (535/600), Macro-F1: 89.14%
- **ResNet-50d:** 88.50% (531/600), Macro-F1: 88.42%
- **ViT-Base-384:** 88.33% (530/600), Macro-F1: 88.29%
- **BiomedCLIP-CBAM:** 88.17% (529/600), Macro-F1: 88.11%

### B. Ensembles & Fusion
- **Fixed Weighted Quad Ensemble:** **91.83%** (551/600), Macro-F1: **91.78%**
  - Bootstrap 95% CI: Accuracy **[89.67%, 93.83%]**, Macro-F1 **[89.75%, 93.76%]**
  - Weights: EffNet-B3 (0.35), ConvNeXt-S (0.25), ResNet-50d (0.20), BiomedCLIP-CBAM (0.20)
- **Uniform Average:** 91.33% (548/600)
- **Stacking MLP Meta-Learner:** 91.17% (547/600)
- **Hierarchical Specialist (G/H/M):** 90.50% (543/600)
- **Oracle Upper Bound (Ceiling):** 94.83% (569/600)

### C. Statistical Significance (Exact McNemar Paired Tests)
- vs. ResNet-50d: $p = 0.0003$ (Significant)
- vs. ViT-Base-384: $p = 0.0052$ (Significant)
- vs. BiomedCLIP-CBAM: $p = 0.0059$ (Significant)
- vs. ConvNeXt-Small: $p = 0.0210$ (Significant)
- vs. EfficientNet-B3: $p = 0.1250$ (**Not significant** at $\alpha=0.05$)

### D. Calibration & Uncertainty (Nested Val-A / Val-B Protocol)
- **Temperature Scaling ($T^* = 0.8889$ on Val-A):** ECE improved from 5.20% to 3.05%
- **Split Conformal Prediction ($lpha=0.05$ on Val-B):**
  - Empirical Coverage: **96.00%** (576/600)
  - Average Prediction Set Size: **1.17** classes
  - Singleton Rate: **84.83%** (509/600)
  - Singleton Precision: **96.46%** (491/509 correct)

---

## 5. Role in Future Scientific Manuscript

In the final research manuscript, this experiment will be presented transparently as:
> *"The Legacy Benchmark: Evaluating multi-architecture ensemble fusion on a balanced public dataset construction."*

Followed immediately by:
> *"Leakage-Controlled Benchmark: Re-evaluating model generalization after eliminating cross-split duplication and enforcing patient/raw image isolation."*

This before-and-after analysis elevates the research from a standard benchmark paper to an impactful study on methodological rigor and leakage vulnerability in ophthalmic deep learning.
