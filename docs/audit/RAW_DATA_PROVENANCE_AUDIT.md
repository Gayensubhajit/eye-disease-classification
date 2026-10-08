# Raw Data Provenance & Forensic Leakage Investigation Report

**Project:** Classification of Eye Diseases from Color Fundus Images  
**Investigation Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `research-clean-split`  
**Focus:** Forensic Investigation of the ORIGINAL Raw Dataset Source & Contamination Mechanism  

---

## Executive Summary

A forensic investigation was conducted into the upstream origin and curation pipeline of the *Eye Disease Image Dataset*. The investigation traced the dataset from its raw host repository through the offline balancing script (`scripts/build_balanced_folders.py`) to the legacy 4,000-image benchmark (`data/train`, `data/val`, `data/test`).

### Headline Findings:
1. **Upstream Source Identified:** The raw dataset originates from **Mendeley Data** (DOI: `10.17632/s9bfhswzjb.1`), compiled by **Mohammad Riadur Rashid, Shayla Sharmin, et al.** (Daffodil International University Health Informatics Lab, in partnership with Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital, Bangladesh), published in 2024 (*Data in Brief*).
2. **Total Absence of Patient Identifiers in Source:** In adherence to clinical confidentiality, the upstream authors **did not release patient identifiers, hospital accession numbers, visit timestamps, or eye laterality tags (Left Eye [OS] vs. Right Eye [OD])**. All files are sequentially named by disease category (e.g., `Pterygium1.jpg` to `Pterygium17.jpg`).
3. **Patient/Eye Grouping Cannot Be Fabricated:** Because patient IDs and laterality were permanently scrubbed at the source hospital collection stage, **individual patient grouping cannot be mathematically reconstructed from filenames or metadata**.
4. **Physical Location of Raw Archive:** The untouched raw dataset resides locally at `/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset`.
5. **Root Mechanism of Cross-Split Contamination:**
   - The original raw dataset contained severe class imbalance: Pterygium had only **17 unique images**, CSCR had 101, Disc Edema had 127, RD had 125, and RP had 139.
   - In `scripts/build_balanced_folders.py`, pre-augmented files from the Mendeley archive were pooled with original files, and additional Albumentations copies were synthesized to hit 400 images/class **before** partitioning into train (280), val (60), and test (60).
   - This caused synthetic twins of individual source eyes to be distributed across splits.
6. **Parent Retina Contamination Rates in Test Set:**
   - **Pterygium:** **100.0%** of test sources (15/15) are also in the training set.
   - **CSCR:** **97.6%** of test sources (41/42) are in the training set.
   - **Disc Edema:** **89.8%** of test sources (44/49) are in the training set.
   - **Retinal Detachment:** **84.1%** of test sources (37/44) are in the training set.
   - **Retinitis Pigmentosa:** **83.3%** of test sources (35/42) are in the training set.
7. **Mapping Artifact Produced:**
   - `outputs/audit/raw_to_processed_mapping.csv` successfully maps all 4,000 processed images back to their closest raw parent retina, with augmentation classification and source hashes.

---

## 1. Upstream Dataset Provenance & Metadata Details

| Item | Forensic Finding |
|---|---|
| **Original Dataset Title** | *Eye Disease Image Dataset* |
| **Primary Repository** | Mendeley Data |
| **DOI** | `10.17632/s9bfhswzjb.1` |
| **Published Article** | *Data in Brief* (Elsevier, 2024) |
| **Compilers / Authors** | Mohammad Riadur Rashid, Shayla Sharmin, et al. |
| **Clinical Affiliations** | Health Informatics Lab, Daffodil International University; Anwara Hamida Eye Hospital; B.N.S.B. Zahurul Haque Eye Hospital (Bangladesh) |
| **Kaggle Mirror** | Uploaded under title *Eye Disease Image Dataset* (e.g., `s9bfhswzjb/1` community mirror) |
| **Local Physical Raw Path** | `/mnt/windows/Users/subha/Downloads/Eye Disease Image Dataset/Eye Disease Image Dataset/` |

---

## 2. Upstream Metadata & Identifier Audit

| Attribute | Present in Upstream Source? | Evidence / Forensic Finding |
|---|---|---|
| **Patient IDs** | ❌ **ABSENT** | The source publication notes: *"every patient's data remains unknown, and the status of their illness is handled with the utmost confidentiality"*. |
| **Eye Identifiers** | ❌ **ABSENT** | No indicator exists to identify whether two images belong to the same person. |
| **Laterality (OD vs OS)** | ❌ **ABSENT** | No tag indicates whether an image is a Left Eye (OS) or Right Eye (OD). |
| **Acquisition IDs** | ❌ **ABSENT** | Camera serial numbers and session IDs are not present. |
| **EXIF Metadata** | ❌ **ABSENT** | All original JPEG images have 0 EXIF metadata tags (scrubbed). |
| **Metadata CSV** | ❌ **ABSENT** | The archive contains only folders of images (`Original Dataset` and `Augmented Dataset`). |
| **Original Filenames** | ✅ **PRESENT** | Named with disease prefix + integer index (e.g. `Pterygium1.jpg` to `Pterygium17.jpg`, `CSCR1.jpg` to `CSCR101.jpg`). |
| **Image Dimensions** | ✅ **PRESENT** | Original raw resolutions range from 2004×1690 to 2592×1944 pixels. |

---

## 3. Class Distribution: Raw vs. Processed 4,000-Image Benchmark

| Disease Category | Raw Unaugmented Images (`Original Dataset`) | Pre-Augmented in Mendeley (`Augmented Dataset`) | Distinct Raw Sources Used in 4,000 Benchmark | Target Split per Class (Train / Val / Test) |
|---|---|---|---|---|
| **Central Serous Chorioretinopathy (CSCR)** | 101 | 606 ($101 \times 6$) | 96 | 280 / 60 / 60 |
| **Diabetic Retinopathy** | 1,509 | 3,444 | 340 | 280 / 60 / 60 |
| **Disc Edema** | 127 | 762 ($127 \times 6$) | 116 | 280 / 60 / 60 |
| **Glaucoma** | 1,349 | 2,880 | 304 | 280 / 60 / 60 |
| **Healthy** | 1,024 | 2,676 | 298 | 280 / 60 / 60 |
| **Macular Scar** | 444 | 1,937 | 243 | 280 / 60 / 60 |
| **Myopia** | 500 | 2,251 | 263 | 280 / 60 / 60 |
| **Pterygium** | **17** | **102** ($17 \times 6$) | **17** | 280 / 60 / 60 |
| **Retinal Detachment** | 125 | 750 ($125 \times 6$) | 113 | 280 / 60 / 60 |
| **Retinitis Pigmentosa** | 139 | 834 ($139 \times 6$) | 124 | 280 / 60 / 60 |
| **TOTAL** | **5,335** | **16,242** | **1,914** | **2,800 / 600 / 600** |

---

## 4. Cross-Split Parent Retina Leakage Analysis

By indexing all 5,335 raw images and running perceptual dHash vector matching against the 4,000 processed images in `data/train`, `data/val`, and `data/test`, we tracked the exact distribution of parent retinas across splits:

| Disease Class | Distinct Raw Sources in Train | Distinct Raw Sources in Test | Shared Raw Sources (Cross-Split Twins) | % of Test Sources Contaminated |
|---|---|---|---|---|
| **Pterygium** | 17 | 15 | **15** | **100.0%** |
| **CSCR** | 92 | 42 | **41** | **97.6%** |
| **Disc Edema** | 110 | 49 | **44** | **89.8%** |
| **Retinal Detachment** | 103 | 44 | **37** | **84.1%** |
| **Retinitis Pigmentosa** | 105 | 42 | **35** | **83.3%** |
| **Macular Scar** | 195 | 55 | **27** | **49.1%** |
| **Myopia** | 206 | 55 | **25** | **45.5%** |
| **Glaucoma** | 228 | 56 | **13** | **23.2%** |
| **Healthy** | 220 | 56 | **11** | **19.6%** |
| **Diabetic Retinopathy** | 244 | 58 | **11** | **19.0%** |

### The Empirical Explanation for Reported 100% Sensitivity:
- In the five minority classes (Pterygium, CSCR, Disc Edema, Retinal Detachment, Retinitis Pigmentosa), between **83.3% and 100.0% of the test set consists of transformed copies of training eyes**.
- The neural networks were effectively tested on images they had already memorized during training, creating artificial 100.0% sensitivity.
- Conversely, in classes with hundreds of unique source retinas (Glaucoma, Healthy, Myopia), test contamination was significantly lower (19%–23%), revealing the models' true generalization performance and clinical confusion (e.g., 76.67% sensitivity in Glaucoma).

---

## 5. Augmentation Breakdown of the 4,000 Processed Images

Audited across all 4,000 rows in `outputs/audit/raw_to_processed_mapping.csv`:

| Augmentation Classification | Image Count | Percentage | Description |
|---|---|---|---|
| **Unaugmented (exact or recompressed)** | 1,489 | 37.2% | Exact raw images from `Original Dataset` |
| **Geometric Affine or Flip** | 1,695 | 42.4% | Horizontal/vertical flips, 90° rotations, affine scaling |
| **Color Jitter or Heavy Augmentation** | 725 | 18.1% | Brightness/contrast shifts, saturation jitter |
| **Synthetic / Distant** | 91 | 2.3% | Heavily distorted synthetic artifacts |
| **TOTAL** | **4,000** | **100.0%** | |

---

## 6. Answers to Core Provenance Questions

1. **What is the exact original dataset/source?**  
   *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`), collected by Daffodil International University with Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital.
2. **What is the original Kaggle dataset URL/name?**  
   Mirrored on Kaggle as *Eye Disease Image Dataset* under the Mendeley Data upload.
3. **Is there an upstream GitHub, paper, dataset repository, or metadata CSV?**  
   Published in *Data in Brief* (2024). There is no metadata CSV; the archive consists purely of JPEG image directories.
4. **Does the upstream source contain patient IDs, eye IDs, laterality, or acquisition IDs?**  
   **NO**. The source authors scrubbed all patient, eye, and laterality data for ethical privacy compliance.
5. **Can the current renamed images be mapped back to their original source filenames?**  
   **YES**. Successfully mapped via perceptual hash matching into `outputs/audit/raw_to_processed_mapping.csv`.
6. **Can the original patient/eye grouping be reconstructed?**  
   **NO**. Without original hospital master records, patient grouping cannot be recovered. Any attempt to fabricate patient groupings would be unscientific.
7. **Was offline augmentation performed before or after the current split?**  
   **BEFORE the split**. In `scripts/build_balanced_folders.py`, raw and augmented images were pooled and shuffled prior to partitioning into train/val/test.
8. **For every synthetic image, can we identify its source/original image?**  
   **YES**. Documented in `outputs/audit/raw_to_processed_mapping.csv`.

---

## 7. Recommended Clean Benchmark Construction Protocol

Since patient-level grouping is permanently unavailable at the source level, the most scientifically defensible benchmark must enforce **Strict Raw-Image Isolation**:

```text
                  5,335 ORIGINAL RAW IMAGES
                             │
                             ▼
               Identify & Deduplicate Raw Pool
                             │
                             ▼
         Stratified Partitioning (e.g., 70 / 15 / 15)
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       TRAIN POOL        VAL POOL        TEST POOL
            │                │                │
            ▼                ▼                ▼
     Online/Offline     NO SYNTHESIS     NO SYNTHESIS
      Augmentation      (Pure Unseen     (Pure Unseen
      (Train only)       Raw Images)      Raw Images)
            │
            ▼
     Model Training
```

### Key Principles for the Clean Split:
1. **Zero Raw Leakage:** No raw image (or any transformed derivative of it) present in the training set may ever appear in the validation or test sets.
2. **Untouched Evaluation Sets:** Validation and Test splits must consist **strictly of genuine, unaugmented raw images**.
3. **Augmentation Restricted to Training:** If minority classes need oversampling or synthetic expansion to balance loss gradients, it must occur **strictly on the training split after partitioning**.
4. **Transparent Limitation Statement:**
   > *"Because the upstream public dataset stripped patient identifiers and eye laterality, raw-image isolation was strictly enforced across train/val/test splits, though patient-level separation cannot be independently verified."*
