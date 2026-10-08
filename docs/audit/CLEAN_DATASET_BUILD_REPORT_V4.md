# Clean Dataset Construction & Post-Build Validation Report (V4)

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Controlled Physical Construction & Post-Build Integrity Audit of `data_clean/`  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `main`  
**Authorization Commit Hash:** `9254432ade30fa90a9e2a701456dc3448e9b8fe2`  
**Frozen Manifest SHA-256:** `08bdf021f222392bf915aaf3627a4acb0ff0134aeab0fdd63301debe70f81a2c`  
**Interim Milestone:** `CLEAN_DATA_BENCHMARK_CONSTRUCTED_PENDING_FINAL_VALIDATION_FIX`  
**Final Milestone Verdict:** **`CLEAN_DATA_BENCHMARK_FROZEN`** (Post-Build Verification Flawlessly Resolved & Validated)  

---

## Executive Summary

Following explicit user authorization under the **Controlled `data_clean/` Construction — V4** protocol and subsequent independent methodology review, the physical dataset `data_clean/` was constructed from the frozen V4 manifest ([`outputs/audit/final_clean_split_manifest_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_manifest_v4.csv)).

Two concrete methodological and bookkeeping items identified during review were rigorously addressed:
1. **Elimination of Non-Image Stray Files:** All placeholder `.gitkeep` files were completely removed from `data_clean/`. The physical filesystem now strictly contains exactly 3,747 valid image files and zero non-image files. `.gitignore` was updated to cleanly exclude `data_clean/`.
2. **Rigorous Endpoint Inspection of All 86 V3 Pairs:** Replaced the previous reconciliation-string check with an independent physical endpoint inspection. Every image endpoint across all 86 V3 high-confidence relationships was queried against the physical filesystem: confirming zero cross-split leakage, zero intra-split duplicate retention, zero quarantine contamination, and exact 1:1 retention of representatives.
3. **Explicit Per-Class / Per-Split Manifest Assertion:** Independently verified all 30 class-split cells (10 classes $\times$ 3 partitions) against the frozen manifest with 100% exact correspondence.

All 3,747 retained fundus photographs were copied directly from the original raw archive, preserving raw pixel values, bit depth, and color gamuts without pre-split augmentation, resizing, or transformation.

---

## 1. Physical Directory Architecture & Inventory

The dataset is partitioned into standard `ImageFolder` hierarchy across 10 disease categories:

```text
data_clean/
├── train/          (2,622 images, 69.98%)
│   ├── Central Serous Chorioretinopathy [Color Fundus]/
│   ├── Diabetic Retinopathy/
│   ├── Disc Edema/
│   ├── Glaucoma/
│   ├── Healthy/
│   ├── Macular Scar/
│   ├── Myopia/
│   ├── Pterygium/
│   ├── Retinal Detachment/
│   └── Retinitis Pigmentosa/
├── val/            (562 images, 15.00%)
│   └── [10 disease class subdirectories]
└── test/           (563 images, 15.03%)
    └── [10 disease class subdirectories]
```

### Complete Physical Class Distribution & Manifest Alignment:

| Disease Class | Train | Val | Test | Total Images | Partition % | Manifest Match |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Diabetic Retinopathy** | 891 | 191 | 191 | 1,273 | 33.97% | **EXACT (3/3)** |
| **Glaucoma** | 579 | 124 | 124 | 827 | 22.07% | **EXACT (3/3)** |
| **Healthy** | 492 | 105 | 105 | 702 | 18.73% | **EXACT (3/3)** |
| **Macular Scar** | 217 | 47 | 47 | 311 | 8.30% | **EXACT (3/3)** |
| **Myopia** | 172 | 37 | 37 | 246 | 6.57% | **EXACT (3/3)** |
| **Retinitis Pigmentosa** | 81 | 17 | 17 | 115 | 3.07% | **EXACT (3/3)** |
| **Disc Edema** | 68 | 15 | 15 | 98 | 2.62% | **EXACT (3/3)** |
| **Retinal Detachment** | 63 | 13 | 14 | 90 | 2.40% | **EXACT (3/3)** |
| **Central Serous Chorioretinopathy [Color Fundus]** | 49 | 11 | 11 | 71 | 1.89% | **EXACT (3/3)** |
| **Pterygium** | 10 | 2 | 2 | 14 | 0.37% | **EXACT (3/3)** |
| **Total Physical Images** | **2,622** | **562** | **563** | **3,747** | **100.00%** | **30/30 MATCH** |

---

## 2. Independent Post-Build Physical Verification Suite

The physical build was validated by [`scripts/build_data_clean_v4.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/build_data_clean_v4.py) and logged to [`outputs/audit/data_clean_v4_validation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_validation.csv):

| Verification Check | Observed Value | Expected Value | Status |
|---|:---:|:---:|:---:|
| **Total Physical Image Files Copied** | **3,747** | 3,747 | **PASS** |
| **Non-Image Stray Files in Directory** | **0** | 0 | **PASS** |
| **Train Split Total Count** | **2,622** | 2,622 | **PASS** |
| **Validation Split Total Count** | **562** | 562 | **PASS** |
| **Test Split Total Count** | **563** | 563 | **PASS** |
| **Per-Class Distribution Cells Matching Manifest** | **30 / 30** | 30 | **PASS** |
| **Physical MD5 Cross-Split Overlap** | **0** | 0 | **PASS** |
| **Quarantine File Contamination** | **0** | 0 | **PASS** |
| **Excluded Burst Redundant Contamination** | **0** | 0 | **PASS** |
| **V3 Pairs Cross-Split Leakage (Endpoint Inspected)** | **0** | 0 | **PASS** |
| **V3 Pairs Both Endpoints Present** | **0** | 0 | **PASS** |
| **V3 Pairs Representative Retained Exactly 1** | **61** | 61 | **PASS** |
| **V3 Pairs Both Endpoints Excluded** | **25** | 25 | **PASS** |

---

## 3. Methodological V3 Endpoint Verification Details

To eliminate any circular reasoning, every pair from [`outputs/audit/v4_v3_reconciliation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_v3_reconciliation.csv) was inspected at the physical filesystem endpoint level:

```python
loc_a = physical_lookup.get(image_a)  # (split, class, path) or None
loc_b = physical_lookup.get(image_b)  # (split, class, path) or None
```

- **Cross-Split Leakage:** In 0 of the 86 relationships are both endpoints present in different partitions ($\text{Leakage} = 0$).
- **Intra-Split Duplicate Retention:** In 0 of the 86 relationships are both endpoints retained in the same partition. Coherent burst clusters strictly retain only the single best-quality representative ($\text{Both Present} = 0$).
- **Representative Retained (61 pairs):** Exactly 61 pairs involve a representative frame retained in `data_clean/` while the redundant burst frame was excluded from the copy.
- **Both Excluded (25 pairs):** Exactly 25 pairs involve endpoints that were both excluded (13 cross-class conflict quarantined pairs, 5 ambiguous chain quarantined pairs, and 7 pairs between two redundant burst frames in larger clusters).
- **Contamination Zero:** 0 quarantined images and 0 excluded burst frames were copied into `data_clean/`.

---

## 4. Strict Preprocessing & Augmentation Boundaries

1. **Pre-Split Augmentation Strictly Avoided:** All images in `data_clean/` are identical bit-for-bit copies of the original archive files.
2. **Post-Split Augmentation Policy:** Any data augmentation (random flips, affine rotations, color jitter) must be implemented strictly on-the-fly inside the PyTorch `Dataset` / `DataLoader` during training, applied exclusively to `train/`.
3. **Validation & Test Freezing:** Validation (`val/`) and test (`test/`) sets must remain completely un-augmented and evaluated exclusively with standard deterministic resizing/normalization.
4. **Legacy Benchmark Isolation:** The legacy directory `data/` (`data/train`, `data/val`, `data/test`), historical checkpoints, and legacy evaluation tables remain frozen and untouched for transparent comparative analysis in the research paper.

---

## 5. Scientific Grounding & Disclaimers

1. **Image-Level / Source-Group Isolation:** Because the upstream Mendeley archive (DOI `10.17632/s9bfhswzjb.1`) contains no patient or laterality identifiers, the benchmark enforces rigorous **image-level isolation and source-group separation**, but does not claim patient-level independence.
2. **Transformation Audit Scope:** The DINOv2 visual embedding search served as an independent transformation-robust audit against perceptual hash blind spots. It is not presented as mathematical proof that all possible source pairs in existence were identified.
3. **Small-Class Evaluation Caution:** Pterygium contains only 14 retained images (10 Train, 2 Val, 2 Test). Macro-averaged F1 and balanced accuracy are mandatory evaluation metrics to prevent minority class collapse.

---

## 6. Produced Build Artifacts

1. [`scripts/build_data_clean_v4.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/build_data_clean_v4.py): Controlled dataset build script with embedded SHA-256/MD5 verification, per-class manifest validation, and endpoint inspection.
2. [`outputs/audit/data_clean_v4_inventory.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_inventory.csv): Complete inventory of all 3,747 physical files, relative paths, disease classes, source-group IDs, byte sizes, MD5, and SHA-256 hashes.
3. [`outputs/audit/data_clean_v4_checksums.sha256`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_checksums.sha256): Standard `sha256sum`-compatible checksum catalog for all 3,747 images in `data_clean/`.
4. [`outputs/audit/data_clean_v4_validation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_validation.csv): Automated validation results table recording passing status across all 13 checks.
