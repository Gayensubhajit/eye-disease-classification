# Cross-Split Near-Duplicate & Burst-Shot Audit Report

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Cross-Split Perceptual Near-Duplicate Verification of `clean_split_manifest.csv`  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `research-clean-split`  
**Final Readiness Verdict:** **`NOT_READY_FOR_DATA_CLEAN_BUILD`**  

---

## Executive Summary & Critical Finding

Following the exact cryptographic deduplication of the *Original Dataset* (which produced 4,387 eligible unique-MD5 source images), an independent, deep perceptual and structural audit was conducted across split boundaries in the proposed seed-42 manifest ([`outputs/audit/clean_split_manifest.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/clean_split_manifest.csv)).

### The Smoking Gun Discovery:
While **exact cryptographic MD5 leakage across splits is confirmed 0**, the cross-split perceptual audit uncovered **258 Tier 2 Highly Suspicious image pairs crossing split boundaries** (including **101 pairs between Train and Test**, **132 pairs between Train and Val**, and **25 pairs between Val and Test**).

Detailed computer-vision inspection (Normalized Cross-Correlation, ORB feature matching, and pixel difference) reveals that the upstream hospital dataset contains **consecutive burst exposures and un-indexed multi-capture shots of the same patient retina** saved under consecutive numerical filenames (e.g., `DR1098.jpg` $\leftrightarrow$ `DR1101.jpg`, `Glaucoma1014.jpg` $\leftrightarrow$ `Glaucoma1011.jpg`, `CSCR4.jpg` $\leftrightarrow$ `CSCR5.jpg`).

Because these burst shots have slightly different compression artifacts, minute illumination variations, or subtle eye saccades, their cryptographic MD5 hashes differed. Consequently, exact MD5 deduplication retained both shots, and pseudo-random shuffling partitioned them across Train and Test splits.

### Final Audit Decision:
**`NOT_READY_FOR_DATA_CLEAN_BUILD`**  
The clean split manifest cannot be used to generate `data_clean/` until these near-duplicate burst clusters are resolved, because evaluating `DR1101` in Test when `DR1098` was seen in Train re-introduces the identical generalization leakage that compromised the legacy benchmark.

---

## 1. Split Manifest Verification & Exact Checks

The proposed manifest was loaded and audited:
- **Train Split:** **3,073** images
- **Validation Split:** **657** images
- **Test Split:** **657** images
- **Total Manifest Size:** **4,387** images

### Exact Cryptographic Checks:
- `MD5 Train-Test Overlap:` **0** (**PASS**)
- `MD5 Train-Val Overlap:` **0** (**PASS**)
- `MD5 Val-Test Overlap:` **0** (**PASS**)
- `Filepath Cross-Split Overlap:` **0** (**PASS**)

All Tier 1 exact cryptographic checks remain completely clean.

---

## 2. Multi-Metric Perceptual Audit Methodology

All $4,469,571$ cross-split image pairs were audited using a tiered multi-scale feature pipeline:
1. **64-bit dHash (Lanczos horizontal gradient difference):** Evaluated across all pairs to identify coarse perceptual similarity ($d_{\text{dHash}} \le 2$).
2. **64-bit pHash (Discrete Cosine Transform low-frequency coefficients):** Evaluated across candidates to identify structural frequency alignment ($d_{\text{pHash}} \le 4$).
3. **Normalized Cross-Correlation (NCC):** Computed on $256 \times 256$ grayscale images to measure global structural correlation.
4. **Mean Absolute Pixel Difference:** Computed on $256 \times 256$ resolution ($0 \le \Delta_{\text{pixel}} \le 255$).
5. **ORB Keypoint & Descriptor Inlier Matching:** Evaluated using 500 ORB features per image and Brute-Force Hamming matcher with Lowe's ratio test ($0.75$).

---

## 3. Candidate Tier Classification & Definitions

| Tier | Classification | Criteria | Measured Count Across Splits | Leakage Status |
|---|---|---|:---:|---|
| **TIER 1** | **Confirmed Duplicate** | Identical MD5 or bitstream duplicate | **0** | **PASS** (Zero exact duplicates) |
| **TIER 2** | **Highly Suspicious** | Probable same source photograph or consecutive burst frame ($	ext{NCC} \ge 0.99$, $\Delta_{\text{pixel}} < 15.0$, or $\ge 15$ ORB inliers) | **258** | **FAIL / LEAKAGE DETECTED** (Must be repaired) |
| **TIER 3** | **Possible Similarity** | Coarse low hash distance ($d_{\text{dHash}} \le 2$), but distinct fine retinal structures | **14,921** | Natural anatomical similarity of human retinas |

---

## 4. Cross-Split Summary Table

Summary of candidate pairs by split boundary:

| Split Pair | Evaluated Cross-Split Pairs | Preliminary Candidates ($d_{\text{dHash}} \le 2$) | Tier 1 (Exact MD5) | Tier 2 (Highly Suspicious / Burst Leakage) | Tier 3 (Natural Similarity) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train $\leftrightarrow$ Validation** | 2,018,961 | 7,217 | **0** | **132** | 7,085 |
| **Train $\leftrightarrow$ Test** | 2,018,961 | 6,401 | **0** | **101** | 6,300 |
| **Validation $\leftrightarrow$ Test** | 431,649 | 1,561 | **0** | **25** | 1,536 |
| **TOTAL** | **4,469,571** | **15,179** | **0** | **258** | **14,921** |

---

## 5. Forensic Deep Dive into Tier 2 Pairs

Of the 258 Tier 2 pairs across splits:
- **135 pairs are SAME-CLASS near-duplicates** (burst shots within the same disease).
- **123 pairs are CROSS-CLASS near-duplicates** (reflecting un-purged subtle label conflicts in the raw dataset).

### Breakdown of Same-Class Tier 2 Pairs by Disease:
- **Glaucoma:** 60 pairs
- **Diabetic Retinopathy:** 28 pairs
- **Healthy:** 23 pairs
- **Myopia:** 10 pairs
- **Macular Scar:** 7 pairs
- **Central Serous Chorioretinopathy:** 3 pairs
- **Retinitis Pigmentosa:** 2 pairs
- **Disc Edema:** 1 pair
- **Retinal Detachment:** 1 pair
- **Pterygium:** **0 pairs** (Pterygium has 0 near-duplicates among its 17 images)

### Representative Smoking-Gun Cross-Split Pairs:

#### Case 1: Diabetic Retinopathy (`Train` $\leftrightarrow$ `Test`)
- **Train Image:** `Diabetic Retinopathy/DR1098.jpg`
- **Test Image:** `Diabetic Retinopathy/DR1101.jpg`
- **Metrics:** $d_{\text{dHash}} = 0$, $d_{\text{pHash}} = 0$, $\text{NCC} = 0.9993$, $\Delta_{\text{pixel}} = 2.36 / 255$, **ORB Keypoint Matches = 365**
- **Forensic Diagnosis:** Undeniable consecutive burst exposures of the identical retina. The vessel bifurcations, optic disc cup, and exudate clusters are geometrically identical down to 2.36 intensity values.

#### Case 2: Diabetic Retinopathy (`Train` $\leftrightarrow$ `Test`)
- **Train Image:** `Diabetic Retinopathy/DR897.jpg`
- **Test Image:** `Diabetic Retinopathy/DR896.jpg`
- **Metrics:** $d_{\text{dHash}} = 0$, $d_{\text{pHash}} = 2$, $\text{NCC} = 0.9997$, $\Delta_{\text{pixel}} = 5.42 / 255$, **ORB Keypoint Matches = 13**
- **Forensic Diagnosis:** Consecutive filenames in hospital accession. The same physical eye evaluated in Test after training on the immediate preceding exposure.

#### Case 3: Glaucoma (`Train` $\leftrightarrow$ `Test`)
- **Train Image:** `Glaucoma/Glaucoma1014.jpg`
- **Test Image:** `Glaucoma/Glaucoma1011.jpg`
- **Metrics:** $d_{\text{dHash}} = 0$, $d_{\text{pHash}} = 2$, $\text{NCC} = 0.9976$, $\Delta_{\text{pixel}} = 6.26 / 255$, **ORB Keypoint Matches = 225**
- **Forensic Diagnosis:** Same patient eye. 225 geometrically verified ORB keypoint inliers across optic disc rim and peripapillary atrophy.

#### Case 4: Central Serous Chorioretinopathy (`Train` $\leftrightarrow$ `Val`)
- **Train Image:** `Central Serous Chorioretinopathy [Color Fundus]/CSCR4.jpg`
- **Val Image:** `Central Serous Chorioretinopathy [Color Fundus]/CSCR5.jpg`
- **Metrics:** $d_{\text{dHash}} = 0$, $d_{\text{pHash}} = 0$, $\text{NCC} = 0.9999$, $\Delta_{\text{pixel}} = 2.33 / 255$, **ORB Keypoint Matches = 50**
- **Forensic Diagnosis:** Nearly 33% of pixels are bitwise identical; mean difference is 2.33. Validation loss and early stopping would be contaminated by memorizing CSCR4 in training.

---

## 6. Machine-Readable Audit Catalogs Produced

1. [`outputs/audit/cross_split_near_duplicate_candidates.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/cross_split_near_duplicate_candidates.csv) (15,179 rows):
   Every candidate pair with $d_{\text{dHash}} \le 2$ or ($d_{\text{dHash}} \le 4$ and $d_{\text{pHash}} \le 4$), recording filenames, classes, splits, MD5s, dHash distance, pHash distance, NCC, pixel difference, ORB inliers, tier classification, and rationale.
2. [`outputs/audit/cross_split_near_duplicate_summary.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/cross_split_near_duplicate_summary.csv) (3 rows):
   Formal summary table reporting candidate counts, Tier 1, Tier 2, and Tier 3 breakdowns for Train-Val, Train-Test, and Val-Test.

---

## 7. Required Remediation Protocol (Before Building `data_clean/`)

To eliminate this leakage pathway and reach `READY_FOR_DATA_CLEAN_BUILD`, we must apply **Perceptual Burst-Cluster Deduplication**:

1. **Construct a Connected Component Graph:**
   - Define an edge between any two images if they form a Tier 2 pair ($	ext{NCC} \ge 0.99$ and $\Delta_{\text{pixel}} < 15.0$, or $\ge 15$ ORB inliers).
2. **Resolve Clusters:**
   - For every cluster of connected images:
     - If all images in the cluster belong to the **same class**: retain **exactly ONE representative image** and exclude the redundant burst frames.
     - If images in the cluster span **different classes**: exclude the entire cluster (as they represent cross-class label ambiguity).
3. **Re-generate Split Manifest:**
   - Partition the resulting truly independent source photograph pool into Train (70%), Val (15%), Test (15%).
4. **Re-verify:**
   - Ensure Tier 1 = 0 AND Tier 2 = 0 across all split boundaries.

---

## 8. Final Decision

# **`NOT_READY_FOR_DATA_CLEAN_BUILD`**

### Rationale:
Convincing same-source-photograph burst pairs (101 in Train-Test, 132 in Train-Val) remain across split boundaries in `clean_split_manifest.csv`. The manifest must undergo perceptual cluster deduplication before directory creation.
