# Final Raw Dataset Deduplication & Clean Split Manifest Audit

**Project:** Classification of Eye Diseases from Color Fundus Images  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Date:** October 2026  
**Branch:** `research-clean-split`  
**Evaluation Target:** Raw Mendeley/Kaggle Original Dataset Archive  
**Target Path:** `/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset`  
**Final Readiness Verdict:** **`READY_FOR_CLEAN_DATASET_BUILD`**  

---

## Executive Summary

A comprehensive, forensic deduplication audit and clean split construction were performed directly on the genuine raw images of the *Eye Disease Image Dataset*. 

Prior audits established that the legacy 4,000-image benchmark (`data/`) suffered from pre-split augmentation leakage. Furthermore, forensic inspection revealed that the upstream *Original Dataset* contains **470 duplicate image groups**, including **464 groups of exact duplicate files assigned to conflicting disease categories**.

To build an unassailable research benchmark, a strict deduplication policy was applied:
1. **Unique images** were retained.
2. **Same-class redundant copies** were deduplicated to a single representative image.
3. **Cross-class conflicting images** were completely excluded from the supervised benchmark.

This resulted in **4,387 clean, label-consistent, unaugmented source images**, which were deterministically partitioned (seed 42) into a stratified split of **3,073 Train (70.0%) / 657 Validation (15.0%) / 657 Test (15.0%)**. All 12 raw-image cross-split leakage checks passed with 100% compliance.

---

## 1. Raw Dataset Inventory

The raw filesystem was audited directly from disk without relying on secondary reports.

- **Base Directory:** `/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset`
- **Total Physical Files:** **5,335**
- **File Format:** 100% JPEG (`.jpg`)
- **Metadata Files:** None present in archive (no CSV, JSON, or text manifests)
- **EXIF Metadata:** 0 tags present across all inspected files (scrubbed prior to public release)

### Raw Filesystem Counts by Class:
1. *Central Serous Chorioretinopathy [Color Fundus]:* 101 images
2. *Diabetic Retinopathy:* 1,509 images
3. *Disc Edema:* 127 images
4. *Glaucoma:* 1,349 images
5. *Healthy:* 1,024 images
6. *Macular Scar:* 444 images
7. *Myopia:* 500 images
8. *Pterygium:* 17 images
9. *Retinal Detachment:* 125 images
10. *Retinitis Pigmentosa:* 139 images
- **Total Files:** **5,335**

---

## 2. Exact Duplicate Findings

Cryptographic MD5 hashing across all 5,335 files identified:
- **Total Unique MD5 Hashes:** **4,851**
- **Redundant Duplicate File Instances:** **484**
- **Group Size Distribution:**
  - `group_size = 1` (Unique): **4,381 hashes** (4,381 files)
  - `group_size = 2`: **457 hashes** (914 files)
  - `group_size = 3`: **12 hashes** (36 files)
  - `group_size = 4`: **1 hash** (4 files)

### Duplicate Classification:
- **`UNIQUE`:** 4,381 groups (4,381 files)
- **`SAME_CLASS_DUPLICATE`:** 6 groups (12 files)
  - Retain 1 representative per group, exclude 6 redundant duplicates.
- **`CROSS_CLASS_DUPLICATE`:** 464 groups (**942 files**)
  - Conflicting labels across classes; all 942 files excluded.

Full catalog saved in: [`outputs/audit/raw_exact_duplicate_groups.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/raw_exact_duplicate_groups.csv)

---

## 3. Cross-Class Label Conflict Findings

The audit confirmed that **464 groups of identical images carry conflicting disease labels** in the upstream Mendeley release:
- **Glaucoma $\leftrightarrow$ Healthy:** 192 duplicate groups (e.g., `Glaucoma134.jpg` is bit-for-bit identical to `Healthy357.jpg`).
- **Glaucoma $\leftrightarrow$ Myopia:** 142 duplicate groups.
- **Diabetic Retinopathy $\leftrightarrow$ Healthy:** 31 duplicate groups.
- **CSCR $\leftrightarrow$ Macular Scar:** 17 duplicate groups (e.g., `CSCR1.jpg` is bit-for-bit identical to `Macular Scar4.jpg`).
- **Disc Edema $\leftrightarrow$ Glaucoma:** 13 duplicate groups.
- **Macular Scar $\leftrightarrow$ Glaucoma:** 28 duplicate groups.
- **Three-Way Conflicts:** Several identical images appear in three classes simultaneously (e.g., `CSCR34.jpg` == `Glaucoma131.jpg` == `Macular Scar202.jpg`).

### Scientific Implication:
These images represent severe ground-truth label noise inherent to the public dataset. Retaining them or guessing a preferred label via majority voting would corrupt supervised training and evaluation.

---

## 4. Deduplication Policy

To establish an unassailable clean benchmark, the following strict resolution rules were enforced:

1. **`UNIQUE` $\rightarrow$ Eligible:**
   - 4,381 images are unique to a single file and a single disease label.
   - Status: `RETAINED` (`clean_dataset_eligible = True`).
2. **`SAME_CLASS_DUPLICATE` $\rightarrow$ Retain One Representative:**
   - For identical files within the same class, exactly one representative file (alphabetically first) was retained.
   - The redundant copy was excluded.
   - Status: 6 representatives `RETAINED_REPRESENTATIVE`; 6 files `EXCLUDED_SAME_CLASS_REDUNDANT`.
3. **`CROSS_CLASS_DUPLICATE` $\rightarrow$ Exclude Entire Group:**
   - Because no metadata or clinical records exist to identify which label is correct, the entire group was excluded from the supervised benchmark.
   - Status: 942 files marked `EXCLUDED_CROSS_CLASS_CONFLICT` (`clean_dataset_eligible = False`).

Full audit manifest saved in: [`outputs/audit/raw_dedup_manifest.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/raw_dedup_manifest.csv)

---

## 5. Post-Deduplication Class Distribution

| Disease Category | Raw Files on Disk | Cross-Class Conflicts Excluded | Same-Class Redundant Excluded | Clean Eligible Source Images | Eligible Retention % |
|---|:---:|:---:|:---:|:---:|:---:|
| **Central Serous Chorioretinopathy** | 101 | 21 | 0 | **80** | 79.2% |
| **Diabetic Retinopathy** | 1,509 | 59 | 2 | **1,448** | 96.0% |
| **Disc Edema** | 127 | 17 | 2 | **108** | 85.0% |
| **Glaucoma** | 1,349 | 358 | 1 | **990** | 73.4% |
| **Healthy** | 1,024 | 201 | 0 | **823** | 80.4% |
| **Macular Scar** | 444 | 92 | 0 | **352** | 79.3% |
| **Myopia** | 500 | 179 | 0 | **321** | 64.2% |
| **Pterygium** | 17 | 0 | 0 | **17** | **100.0%** |
| **Retinal Detachment** | 125 | 4 | 1 | **120** | 96.0% |
| **Retinitis Pigmentosa** | 139 | 11 | 0 | **128** | 92.1% |
| **TOTAL** | **5,335** | **942** | **6** | **4,387** | **82.2%** |

### Accounting Reconciliation:
$$\text{Eligible Clean Images (4,387)} + \text{Cross-Class Excluded (942)} + \text{Same-Class Redundant (6)} = 5,335 \text{ Raw Files}$$
The accounting balances with 100% mathematical precision.

---

## 6. Near-Duplicate Analysis

A pairwise perceptual hash analysis (64-bit dHash) was conducted across the 4,387 eligible clean source images ($pprox 9.62 \times 10^6$ pairwise comparisons):

- **Distance = 0 (dHash Collisions, Different MD5):** **1,250 pairs** (`POSSIBLE_NEAR_DUPLICATE`)
  - These represent images that share coarse 8x8 gradient profiles despite distinct raw pixel matrices (common in visually uniform normal or mild maculopathy fundus images).
- **Distance 1 to 2:** **27,830 pairs** (`REVIEW_REQUIRED`)
- **Distance 3 to 4:** 127,598 pairs (`REVIEW_REQUIRED`)
- **Distance > 4:** Remaining pairs are `CLEARLY_DISTINCT`.

Full candidate ledger saved in: [`outputs/audit/raw_near_duplicate_candidates.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/raw_near_duplicate_candidates.csv)

> [!NOTE]
> Perceptual distance alone is NOT proof that two images belong to the same patient or eye. Images were not deleted based on perceptual similarity, but are cataloged for transparent error analysis.

---

## 7. Proposed Clean Split Manifest

The 4,387 clean eligible source images were partitioned using:
- **Deterministic Random Seed:** `42`
- **Partition Ratio:** 70% Train / 15% Validation / 15% Test
- **Integer Allocation Formula:**
  $$N_{\text{val}} = \max(1, \text{round}(0.15 \times N)), \quad N_{\text{test}} = \max(1, \text{round}(0.15 \times N)), \quad N_{\text{train}} = N - N_{\text{val}} - N_{\text{test}}$$

### Exact Class Allocation:

| Disease Class | Eligible Images | Train (70%) | Validation (15%) | Test (15%) | Split Fractions |
|---|:---:|:---:|:---:|:---:|:---:|
| **Central Serous Chorioretinopathy** | 80 | 56 | 12 | 12 | 70.0% / 15.0% / 15.0% |
| **Diabetic Retinopathy** | 1,448 | 1,014 | 217 | 217 | 70.0% / 15.0% / 15.0% |
| **Disc Edema** | 108 | 76 | 16 | 16 | 70.4% / 14.8% / 14.8% |
| **Glaucoma** | 990 | 694 | 148 | 148 | 70.1% / 14.9% / 14.9% |
| **Healthy** | 823 | 577 | 123 | 123 | 70.1% / 14.9% / 14.9% |
| **Macular Scar** | 352 | 246 | 53 | 53 | 69.9% / 15.1% / 15.1% |
| **Myopia** | 321 | 225 | 48 | 48 | 70.1% / 15.0% / 15.0% |
| **Pterygium** | 17 | 11 | 3 | 3 | 64.7% / 17.6% / 17.6% |
| **Retinal Detachment** | 120 | 84 | 18 | 18 | 70.0% / 15.0% / 15.0% |
| **Retinitis Pigmentosa** | 128 | 90 | 19 | 19 | 70.3% / 14.8% / 14.8% |
| **TOTAL** | **4,387** | **3,073** (70.0%) | **657** (15.0%) | **657** (15.0%) | **100.0%** |

Split manifest saved in: [`outputs/audit/clean_split_manifest.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/clean_split_manifest.csv)

---

## 8. Split Leakage Verification

All 12 formal leakage tests were evaluated against the proposed clean manifest:

| Check ID | Description | Measured | Threshold | Verdict |
|---|---|:---:|:---:|:---:|
| `CHK_01_MD5_TRAIN_TEST_OVERLAP` | MD5 hash shared between Train and Test | **0** | 0 | **PASS** |
| `CHK_02_MD5_TRAIN_VAL_OVERLAP` | MD5 hash shared between Train and Val | **0** | 0 | **PASS** |
| `CHK_03_MD5_VAL_TEST_OVERLAP` | MD5 hash shared between Val and Test | **0** | 0 | **PASS** |
| `CHK_04_FILEPATH_CROSS_SPLIT_OVERLAP` | File path shared across any split | **0** | 0 | **PASS** |
| `CHK_05_EXCLUDED_PATHS_IN_MANIFEST` | Excluded file path present in manifest | **0** | 0 | **PASS** |
| `CHK_06_CROSS_CLASS_CONFLICT_MD5_IN_MANIFEST` | Cross-class conflicting MD5 present in manifest | **0** | 0 | **PASS** |
| `CHK_07_SAME_CLASS_REDUNDANT_PATH_IN_MANIFEST` | Redundant same-class duplicate path present in manifest | **0** | 0 | **PASS** |
| `CHK_08_AUGMENTED_ARCHIVE_CONTAMINATION` | Image sourced from Augmented Dataset folder | **0** | 0 | **PASS** |
| `CHK_09_TEST_PURITY` | Genuine unaugmented original images in test split | **657/657 (100.0%)** | 100.0% | **PASS** |
| `CHK_10_VAL_PURITY` | Genuine unaugmented original images in val split | **657/657 (100.0%)** | 100.0% | **PASS** |
| `CHK_11_TRAIN_PURITY` | Genuine unaugmented original images in train split | **3073/3073 (100.0%)** | 100.0% | **PASS** |
| `CHK_12_TOTAL_SPLIT_INTEGRITY` | Sum of train + val + test matches exact eligible count | **4,387** | 4,387 | **PASS** |

Verification log saved in: [`outputs/audit/clean_split_leakage_check.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/clean_split_leakage_check.csv)

---

## 9. Pterygium Limitation

- **Original Count:** Exactly 17 genuine raw images exist in the public repository.
- **Deduplication:** All 17 are unique images (0 cross-class conflicts, 0 same-class duplicates).
- **Split Allocation:** **11 Train, 3 Validation, 3 Test**.
- **Scientific Limitation:** With only 3 test images, each correct or incorrect prediction shifts measured sensitivity by **33.3 percentage points**. This sample size is too small for stable statistical estimation.
- **Reporting Protocol:** In future publications, Pterygium results must be reported with exact counts (e.g. 2/3 correct) and explicitly accompanied by a sample-size disclaimer. The manuscript will report both the full 10-class task and a secondary 9-class benchmark excluding Pterygium.

---

## 10. Patient-Level Metadata Limitation

> *"Patient-level isolation cannot be independently verified because the public source dataset does not provide patient identifiers or eye laterality metadata. The clean benchmark therefore enforces strict raw-image isolation."*

Under no circumstances should the research paper claim "patient-level split" or "zero patient leakage." The benchmark enforces image-level independence of genuine, unaugmented raw photographs.

---

## 11. Final PASS/FAIL Readiness Decision

### Decision: **`READY_FOR_CLEAN_DATASET_BUILD`**

### Evidence Summary for Readiness:
1. Every raw image on disk has been accounted for and fingerprinted with cryptographic MD5.
2. 942 images with cross-class conflicting ground truth labels have been purged.
3. 6 redundant same-class duplicate files have been deduplicated to single representatives.
4. Exactly 4,387 genuine original source images have been deterministically assigned to train, validation, and test splits.
5. All 12 raw-image cross-split leakage checks passed with zero errors.
6. The legacy 4,000-image dataset (`data/`) remains completely untouched and preserved.

The clean manifest is mathematically sealed and ready for directory creation (`data_clean/`).
