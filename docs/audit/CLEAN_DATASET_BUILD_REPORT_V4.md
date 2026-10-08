# Clean Dataset Construction & Post-Build Validation Report (V4)

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Controlled Physical Construction & Post-Build Integrity Audit of `data_clean/`  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `main`  
**Authorization Commit Hash:** `9254432ade30fa90a9e2a701456dc3448e9b8fe2`  
**Frozen Manifest SHA-256:** `08bdf021f222392bf915aaf3627a4acb0ff0134aeab0fdd63301debe70f81a2c`  
**Final Milestone Verdict:** **`CLEAN_DATA_BENCHMARK_FROZEN`** (Dataset Built, Verified, and Ready for Clean Training Experiments)  

---

## Executive Summary

Following explicit user authorization under the **Controlled `data_clean/` Construction — V4** protocol, the clean benchmark directory `data_clean/` was physically instantiated from the frozen V4 manifest ([`outputs/audit/final_clean_split_manifest_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_manifest_v4.csv)).

All 3,747 retained fundus photographs were copied directly from the original raw archive, preserving raw pixel values, bit depth, and color gamuts without pre-split augmentation, resizing, or transformation. An independent physical verification suite was executed directly against the generated directory structure to confirm zero contamination, zero hash collisions, exact class counts, and zero cross-split leakage.

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

### Complete Physical Class Distribution:

| Disease Class | Train | Val | Test | Total Images | Partition % |
|---|:---:|:---:|:---:|:---:|:---:|
| **Diabetic Retinopathy** | 891 | 191 | 191 | 1,273 | 33.97% |
| **Glaucoma** | 579 | 124 | 124 | 827 | 22.07% |
| **Healthy** | 492 | 105 | 105 | 702 | 18.73% |
| **Macular Scar** | 217 | 47 | 47 | 311 | 8.30% |
| **Myopia** | 172 | 37 | 37 | 246 | 6.57% |
| **Retinitis Pigmentosa** | 81 | 17 | 17 | 115 | 3.07% |
| **Disc Edema** | 68 | 15 | 15 | 98 | 2.62% |
| **Retinal Detachment** | 63 | 13 | 14 | 90 | 2.40% |
| **Central Serous Chorioretinopathy [Color Fundus]** | 49 | 11 | 11 | 71 | 1.89% |
| **Pterygium** | 10 | 2 | 2 | 14 | 0.37% |
| **Total Physical Images** | **2,622** | **562** | **563** | **3,747** | **100.00%** |

---

## 2. Independent Post-Build Physical Verification

The physical build was validated by [`scripts/build_data_clean_v4.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/build_data_clean_v4.py) and logged to [`outputs/audit/data_clean_v4_validation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_validation.csv):

| Verification Check | Observed Value | Expected Value | Status |
|---|:---:|:---:|:---:|
| **Total Physical Files Copied** | **3,747** | 3,747 | **PASS** |
| **Train Split Physical Count** | **2,622** | 2,622 | **PASS** |
| **Validation Split Physical Count** | **562** | 562 | **PASS** |
| **Test Split Physical Count** | **563** | 563 | **PASS** |
| **Physical MD5 Cross-Split Overlap** | **0** | 0 | **PASS** |
| **Quarantined File Contamination** | **0** | 0 | **PASS** |
| **V3 Cross-Split Burst Leakage** | **0** | 0 | **PASS** |
| **SHA-256 Inventory Hash Verification** | **3,747 / 3,747** | 3,747 | **PASS** |

### Key Validation Findings:
1. **Zero Missing or Extraneous Files:** Exactly 3,747 files exist in `data_clean/`. Zero unexpected or duplicate files were found.
2. **Zero Cross-Split Hash Overlap:** Evaluating the MD5 sets of `data_clean/train`, `data_clean/val`, and `data_clean/test` yields zero intersection:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. **Zero Quarantine Contamination:** None of the 397 quarantined images (390 cross-class conflicts + 7 ambiguous chain images from `SRC_GROUP_0007`) are present in `data_clean/`.
4. **Zero Burst Copy Contamination:** None of the 243 excluded secondary burst frames exist in `data_clean/`.
5. **Zero V3 Burst Leakage:** All 86 high-confidence pairs discovered during the V3 DINOv2 audit are confirmed to have zero cross-split leakage.

---

## 3. Strict Preprocessing & Augmentation Boundaries

1. **Pre-Split Augmentation Strictly Avoided:** All images in `data_clean/` are identical bit-for-bit copies of the original archive files.
2. **Post-Split Augmentation Policy:** Any data augmentation (random flips, affine rotations, color jitter) must be implemented strictly on-the-fly inside the PyTorch `Dataset` / `DataLoader` during training, applied exclusively to `train/`.
3. **Validation & Test Freezing:** Validation (`val/`) and test (`test/`) sets must remain completely un-augmented and evaluated exclusively with standard deterministic resizing/normalization.
4. **Legacy Benchmark Isolation:** The legacy directory `data/` (`data/train`, `data/val`, `data/test`), historical checkpoints, and legacy evaluation tables remain frozen and untouched for transparent comparative analysis in the research paper.

---

## 4. Scientific Grounding & Disclaimers

1. **Image-Level / Source-Group Isolation:** Because the upstream Mendeley archive (DOI `10.17632/s9bfhswzjb.1`) contains no patient or laterality identifiers, the benchmark enforces rigorous **image-level isolation and source-group separation**, but does not claim patient-level independence.
2. **Transformation Audit Scope:** The DINOv2 visual embedding search served as an independent transformation-robust audit against perceptual hash blind spots. It is not presented as mathematical proof that all possible source pairs in existence were identified.
3. **Small-Class Evaluation Caution:** Pterygium contains only 14 retained images (10 Train, 2 Val, 2 Test). Macro-averaged F1 and balanced accuracy are mandatory evaluation metrics to prevent minority class collapse.

---

## 5. Produced Build Artifacts

1. [`scripts/build_data_clean_v4.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/build_data_clean_v4.py): Controlled dataset build script with embedded SHA-256 and MD5 verification.
2. [`outputs/audit/data_clean_v4_inventory.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_inventory.csv): Complete inventory of all 3,747 physical files, relative paths, disease classes, source-group IDs, byte sizes, MD5, and SHA-256 hashes.
3. [`outputs/audit/data_clean_v4_checksums.sha256`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_checksums.sha256): Standard `sha256sum`-compatible checksum catalog for all 3,747 images in `data_clean/`.
4. [`outputs/audit/data_clean_v4_validation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/data_clean_v4_validation.csv): Automated validation results table recording passing status across all 8 checks.
