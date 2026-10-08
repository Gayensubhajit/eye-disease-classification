# Source-Group Resolution Audit & Split Manifest Report (V4)

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Source-Group Resolution Audit & Group-Level Stratified Split Rebuild  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `main`  
**Eligible Raw Pool:** 4,387 images (derived from 5,335 raw Mendeley archive images)  
**Retained Benchmark Pool:** 3,748 images  
**Final Split Scheme:** Group-Level Stratified Split (Train ≈ 70%, Val ≈ 15%, Test ≈ 15%)  
**Final Milestone Verdict:** **`NOT_READY_FOR_CLEAN_DATA_BUILD`** (Awaiting user review of V4 source-group resolution)  

---

## Executive Summary: The Architectural Transition to Source Groups

Prior audit phases (V1 to V3) revealed a fundamental methodological vulnerability in standard medical image splitting:
1. **The Hash Filter Blind Spot:** The initial dHash/pHash pre-filter ($d_{\text{dHash}} \le 2$ or $d_{\text{dHash}} \le 4, d_{\text{pHash}} \le 4$) missed subtle retinal burst sequences where minor camera angles, exposure shifts, or JPEG recompression elevated perceptual hash distances despite identical retinal vasculature.
2. **The V3 Discovery:** Transformation-robust DINOv2 feature embeddings uncovered **86 high-confidence capture sequence / burst relationships** crossing the V2 split boundaries (including **31 Train–Test pairs**, **44 Train–Val pairs**, and **8 cross-class label conflicts**).
3. **The Architectural Shift:** Because the public Mendeley archive (DOI `10.17632/s9bfhswzjb.1`) does not provide patient or laterality identifiers, creating a defensible benchmark requires transitioning from an **individual-image split** to a **source-group stratified split**.

In Audit V4, all discovered high-confidence source relationships across the entire eligible pool were unified into a **Source-Group Quarantine Graph**. Connected components were audited for internal coherence to prevent transitive chain over-merging. For coherent same-class groups, exactly one canonical representative was selected using an objective sharpness and illumination metric. Cross-class conflict groups were strictly quarantined with zero relabeling or majority voting. Finally, dataset partitioning was executed at the source-group level, ensuring that no source group spans split boundaries.

---

## 1. Unified Forensic Edge Extraction & Sources

The candidate edge graph was constructed from two complementary, high-confidence forensic sources across the complete 4,387-image eligible pool:

| Forensic Source | Description | Pairs Extracted |
|---|---|:---:|
| **Source A: V3 DINOv2 Cross-Split Audit** | All pairs classified as `CONFIRMED_SAME_SOURCE` or `HIGH_CONFIDENCE_CAPTURE_SEQUENCE` from the 384-dimensional DINOv2 nearest-neighbor search | 86 |
| **Source B: Complete-Pool V2 Audit** | All multi-image cluster relationships established via geometric RANSAC homography, FOV-masked SSIM, and filename sequence corroboration | 227 clusters |

### Robust Forensic Criteria
Image pairs were admitted as high-confidence source relationships if they satisfied:
- **`CONFIRMED_SAME_SOURCE`:** Masked SSIM $\ge 0.95$, Masked NCC $\ge 0.99$, and Mean Masked Pixel Difference $\le 6.0/255$.
- **`HIGH_CONFIDENCE_CAPTURE_SEQUENCE`:** Masked SSIM $\ge 0.90$ with adjacent filename sequencing ($|\Delta_{\text{num}}| \le 5$), OR OpenCV ORB with RANSAC geometric homography inliers $\ge 15$ (inlier ratio $\ge 0.20$), OR Masked SSIM $\ge 0.92$.

All weak bridge connections and ambiguous pairs were classified as `FALSE_POSITIVE_REVIEW` and excluded from triggering component mergers.

---

## 2. Source-Group Resolution & Coherence Audit

A connected-component graph $G = (V, E)$ was initialized over all $N = 4,387$ eligible images. Connected components were analyzed and categorized into four mutually exclusive operational statuses:

| Coherence Status | Group Count | Member Images | Action Taken |
|---|:---:|:---:|---|
| **`SINGLETON`** | 3,525 | 3,525 | Retained as atomic, independent source images |
| **`COHERENT_SOURCE_GROUP`** | 222 | 465 | 1 canonical representative retained; 243 redundant burst copies excluded |
| **`CHAIN_REVIEW_REQUIRED`** | 1 | 7 | Same-class triad/chain (`SRC_GROUP_0007`); 1 canonical representative retained; 6 copies excluded |
| **`CROSS_CLASS_CONFLICT_GROUP`** | 63 | 390 | **100% QUARANTINED**; completely excluded from the clean benchmark |
| **Total** | **3,811** | **4,387** | **3,748 Retained; 639 Excluded/Quarantined** |

### Audit of Multi-Image Groups:
- **Same-Class Coherent Groups (222 groups, 465 images):** Consist of simple duplicate pairs (size 2: 213 groups) and coherent triads/burst sequences (size 3–5: 9 groups). Pairwise DINOv2 cosine similarities within these groups average $0.9614$.
- **Chain Review Required (1 group, 7 images):** `SRC_GROUP_0007` consists of `Retinal Detachment54, 55, 56, 57, 58, 59, 64`. Pairwise DINOv2 similarities range from $0.8615$ (between endpoints) to $0.9821$ (between adjacent frames), with a mean of $0.9364$. Because all 7 images belong to Retinal Detachment, retaining exactly 1 representative eliminates any chaining risk.
- **Cross-Class Conflict Groups (63 groups, 390 images):** Image clusters where identical or burst-captured retinas carry conflicting disease labels across classes (e.g. Diabetic Retinopathy vs. Glaucoma vs. Healthy). All 63 groups are quarantined in [`outputs/audit/v4_quarantine.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_quarantine.csv). In accordance with scientific integrity guidelines:
  - No majority voting was applied.
  - No manual relabeling was performed.
  - All 390 images are permanently excluded from the benchmark.

---

## 3. Deterministic Representative Selection Rule

For every retained same-class source group, exactly **ONE representative image** was selected deterministically based on optical sharpness and valid field-of-view illumination:

$$\text{Quality Score} = \text{Var}\left(\nabla^2 I_{\text{gray}}\right) \times \frac{\sum (I_{\text{gray}} > 15)}{H \times W}$$

Where:
- $\text{Var}\left(\nabla^2 I_{\text{gray}}\right)$ is the Laplacian variance measuring high-frequency edge sharpness.
- $\frac{\sum (I_{\text{gray}} > 15)}{H \times W}$ is the ratio of valid, illuminated retinal tissue relative to total frame area.
- Ties are broken deterministically by sorted file path.

The selected image is assigned the role `REPRESENTATIVE` (retained in the benchmark pool), while all remaining images in the group are assigned `BURST_REDUNDANT` (excluded from the benchmark pool).

---

## 4. Full Mathematical Accounting Reconciliation

The complete raw archive of 5,335 images reconciles with 100% mathematical precision:

$$\begin{aligned}
\mathbf{5,335 \text{ Raw Files}} &= 942 \text{ (Raw cross-class exact duplicate files)} \\
&\quad + 6 \text{ (Redundant exact duplicate copies)} \\
&\quad + 390 \text{ (V4 quarantined cross-class conflict source group images)} \\
&\quad + 249 \text{ (V4 excluded same-class redundant burst frames)} \\
&\quad + 3,748 \text{ (Retained clean benchmark images)}
\end{aligned}$$

$$942 + 6 + 390 + 249 + 3,748 = \mathbf{5,335}$$

Every single image in the upstream archive is accounted for with an immutable role and lineage record in [`outputs/audit/v4_group_members.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_group_members.csv).

---

## 5. Candidate Group-Level Stratified Split (Seed 42)

Dataset partitioning was performed strictly at the **source-group level** across all 3,748 retained groups, stratified by the disease class of each group's canonical representative:
- **Train:** 2,622 images (**69.96%**)
- **Validation:** 563 images (**15.02%**)
- **Test:** 563 images (**15.02%**)

### Stratified Distribution Across Disease Classes:

| Disease Class | Train | Val | Test | Total Retained |
|---|:---:|:---:|:---:|:---:|
| **Diabetic Retinopathy** | 891 | 191 | 191 | 1,273 |
| **Glaucoma** | 579 | 124 | 124 | 827 |
| **Healthy** | 492 | 105 | 105 | 702 |
| **Macular Scar** | 217 | 47 | 47 | 311 |
| **Myopia** | 172 | 37 | 37 | 246 |
| **Retinitis Pigmentosa** | 81 | 17 | 17 | 115 |
| **Disc Edema** | 68 | 15 | 15 | 98 |
| **Retinal Detachment** | 63 | 14 | 14 | 91 |
| **Central Serous Chorioretinopathy [Color Fundus]** | 49 | 11 | 11 | 71 |
| **Pterygium** | 10 | 2 | 2 | 14 |
| **Total** | **2,622** | **563** | **563** | **3,748** |

---

## 6. Comprehensive Leakage Validation & Verification

Validation checks were executed across all split boundaries ([`outputs/audit/final_clean_split_validation_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_validation_v4.csv)):

| Verification Check | Observed Value | Expected Value | Status |
|---|:---:|:---:|:---:|
| **MD5 Cross-Split Overlap** | **0** | 0 | **PASS** |
| **Source Group ID Cross-Split Overlap** | **0** | 0 | **PASS** |
| **V3 High-Confidence Burst Pairs Crossing Split** | **0** | 0 | **PASS** |
| **Quarantined Images Present in Split** | **0** | 0 | **PASS** |
| **Unique Source Group ID per Retained Image** | **3,748 / 3,748** | 3,748 | **PASS** |
| **Mathematical Reconciliation to Raw 5,335** | **5,335** | 5,335 | **PASS** |
| **Expanded DINOv2 Top-50 Nearest Neighbor Audit** | **44,077 pairs** | Audited | **AUDITED** |

### Expanded DINOv2 Top-50 Nearest Neighbor Audit
For every one of the 3,748 retained images, its top-50 nearest neighbors in normalized DINOv2 feature space ($S_{ij} = u_i \cdot u_j$) were computed across the candidate split. Among the 44,077 cross-split candidate pairs examined:
- **Zero** confirmed same-source or high-confidence burst sequences cross split boundaries.
- The highest cosine similarities crossing boundaries reflect genuine anatomical features of the fundus (e.g. optic disc cup-to-disc ratio in glaucoma or general orange retinal background reflectance) rather than reused photograph sessions.

---

## 7. Crucial Scientific Disclaimers & Grounding

1. **No Claim of Patient-Level Independence:** The upstream Mendeley repository does not include patient or laterality identifiers. The V4 benchmark guarantees **image-level isolation and source-group separation**, but cannot guarantee patient-level independence.
2. **Independent Forensic Transformation Audit:** Pretrained DINOv2 representations provide an independent transformation-robust audit against perceptual hash blind spots. It is not presented as mathematical proof that zero unseen transformations exist.
3. **Severe Class Imbalance:** Pterygium contains only 14 retained images (10 Train, 2 Val, 2 Test). Macro-averaged F1 and balanced accuracy are mandatory evaluation metrics.
4. **Historical Benchmark Integrity:** All legacy files, checkpoints, and predictions under `data/` remain untouched for historical evaluation.

---

## 8. Final Decision & Status

# **`NOT_READY_FOR_CLEAN_DATA_BUILD`**

### Summary of Current State:
- Source-Group Resolution Audit V4 has established a unified source-group graph.
- All 86 V3 cross-split pairs and 63 cross-class conflict clusters have been cleanly resolved or quarantined.
- Zero duplicate or burst sequences cross the candidate split boundaries.
- **Physical directory creation of `data_clean/` remains paused awaiting user review.**

---

## 9. Produced Audit Artifacts

1. [`outputs/audit/v4_source_groups.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_source_groups.csv): Catalog of all 3,811 source groups, sizes, member lists, and coherence statuses.
2. [`outputs/audit/v4_group_members.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_group_members.csv): Full 4,387-image lineage, group IDs, roles (`SINGLETON`, `REPRESENTATIVE`, `BURST_REDUNDANT`, `CONFLICT_QUARANTINED`), and optical quality scores.
3. [`outputs/audit/v4_quarantine.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_quarantine.csv): Catalog of 390 quarantined images from 63 cross-class conflict groups.
4. [`outputs/audit/final_clean_split_manifest_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_manifest_v4.csv): Candidate clean split manifest containing 3,748 images (2,622 Train, 563 Val, 563 Test).
5. [`outputs/audit/final_clean_split_validation_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_validation_v4.csv): Validation results table confirming zero leakage across all metrics.
