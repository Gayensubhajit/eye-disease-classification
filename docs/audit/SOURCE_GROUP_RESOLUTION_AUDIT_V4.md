# Source-Group Resolution Audit & Split Manifest Report (V4)

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Source-Group Resolution Audit & Group-Level Stratified Split Rebuild  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `main`  
**Eligible Raw Pool:** 4,387 images (derived from 5,335 raw Mendeley archive images)  
**Retained Benchmark Pool:** 3,747 images  
**Final Split Scheme:** Group-Level Stratified Split (Train ≈ 70%, Val ≈ 15%, Test ≈ 15%)  
**Final Milestone Verdict:** **`NOT_READY_FOR_CLEAN_DATA_BUILD`** (Independent Artifact Verification Completed; Awaiting User Authorization for Clean Dataset Construction)  

---

## Executive Summary: The Architectural Transition to Source Groups

Prior audit phases (V1 to V3) revealed a fundamental methodological vulnerability in standard medical image splitting:
1. **The Hash Filter Blind Spot:** The initial dHash/pHash pre-filter ($d_{\text{dHash}} \le 2$ or $d_{\text{dHash}} \le 4, d_{\text{pHash}} \le 4$) missed subtle retinal burst sequences where minor camera angles, exposure shifts, or JPEG recompression elevated perceptual hash distances despite identical retinal vasculature.
2. **The V3 Discovery:** Transformation-robust DINOv2 feature embeddings uncovered **86 high-confidence capture sequence / burst relationships** crossing the V2 split boundaries (including **31 Train–Test pairs**, **44 Train–Val pairs**, and **8 cross-class label conflicts**).
3. **The Architectural Shift:** Because the public Mendeley archive (DOI `10.17632/s9bfhswzjb.1`) does not provide patient or laterality identifiers, creating a defensible benchmark requires transitioning from an **individual-image split** to a **source-group stratified split**.

In Audit V4, all discovered high-confidence source relationships across the entire eligible pool were unified into a **Source-Group Quarantine Graph**. Connected components were audited for internal coherence to prevent transitive chain over-merging. Ambiguous chains (specifically `SRC_GROUP_0007`) and cross-class conflicts were strictly quarantined with zero relabeling or majority voting. For coherent same-class groups, exactly one canonical representative was selected using a noise-robust optical quality score. Dataset partitioning was executed at the source-group level, ensuring that no source group spans split boundaries.

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
| **`CHAIN_REVIEW_QUARANTINED`** | 1 | 7 | `SRC_GROUP_0007` (7 Retinal Detachment frames, min DINO 0.8615); **100% QUARANTINED** |
| **`CROSS_CLASS_CONFLICT_GROUP`** | 63 | 390 | **100% QUARANTINED**; completely excluded from the clean benchmark |
| **Total** | **3,811** | **4,387** | **3,747 Retained; 640 Excluded/Quarantined** |

### Audit & Quarantine Decisions on Sensitive Groups:
1. **The 7-Image Retinal Detachment Chain (`SRC_GROUP_0007`):**
   - Members: `Retinal Detachment54, 55, 56, 57, 58, 59, 64`.
   - Investigation revealed that `Retinal Detachment64.jpg` had a weak pairwise link to `54` ($\text{NCC} = 0.678$, $\text{SSIM} = 0.7381$, DINOv2 cosine similarity drops to $0.8615$) triggered by circular border ORB keypoints, while images `54–59` formed a tight burst sequence.
   - **Resolution:** In accordance with the requirement not to gamble on ambiguous chains, **all 7 images in `SRC_GROUP_0007` are quarantined** in [`outputs/audit/v4_quarantine.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_quarantine.csv) under reason `AMBIGUOUS_TRANSITIVE_CHAIN`. Zero images from this chain enter the clean benchmark.
2. **Cross-Class Conflict Groups (63 groups, 390 images):**
   - Image clusters where visually identical or burst-captured retinas carry conflicting disease labels across classes (e.g. Diabetic Retinopathy vs. Glaucoma vs. Healthy).
   - **Resolution:** All 63 groups (390 images) are permanently quarantined in [`outputs/audit/v4_quarantine.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_quarantine.csv). Zero majority voting and zero manual relabeling were applied.

---

## 3. Improved Noise-Robust Representative Selection

To prevent high-frequency camera noise from inflating sharpness scores and to penalize over-exposed glare artifacts, canonical representatives were chosen using a refined, noise-robust formulation:

$$\text{Quality Score} = \text{Var}\left(\nabla^2 G_{\sigma=1}(I_{\text{gray}})\right) \times \frac{\sum (I_{\text{gray}} > 15)}{H \times W} \times \left(1.0 - \frac{\sum (I_{\text{gray}} \ge 250)}{H \times W}\right)$$

Where:
- $\text{Var}\left(\nabla^2 G_{\sigma=1}(I_{\text{gray}})\right)$ is the Laplacian variance computed **after Gaussian smoothing** ($\sigma = 1.0$), suppressing high-frequency sensor noise while measuring true anatomical retinal vessel edge clarity.
- $\frac{\sum (I_{\text{gray}} > 15)}{H \times W}$ is the valid illuminated field-of-view (FOV) area ratio.
- $\left(1.0 - \frac{\sum (I_{\text{gray}} \ge 250)}{H \times W}\right)$ explicitly penalizes camera flash glare, reflection washouts, and sensor saturation.
- Ties are broken deterministically by sorted file path.

Auditing this metric across all multi-image groups confirmed that:
- In small and sensitive classes (e.g. CSCR `SRC_GROUP_0020`, Pterygium `SRC_GROUP_0038`), the selected canonical representatives (`CSCR2.jpg`, `Pterygium13.jpg`) are clean, well-focused, properly illuminated, and free from cropping or flash-glare defects.
- In 9 burst groups where raw Laplacian variance had previously favored noisy border frames, the refined metric correctly selected the cleaner, well-exposed exposure.

---

## 4. Full Mathematical Accounting Reconciliation

The complete raw archive of 5,335 images reconciles with 100% mathematical precision:

$$\begin{aligned}
\mathbf{5,335 \text{ Raw Files}} &= 942 \text{ (Raw cross-class exact duplicate files)} \\
&\quad + 6 \text{ (Redundant exact duplicate copies)} \\
&\quad + 397 \text{ (V4 quarantined images: 390 cross-class conflicts + 7 ambiguous chain)} \\
&\quad + 243 \text{ (V4 excluded same-class redundant burst frames)} \\
&\quad + 3,747 \text{ (Retained clean benchmark images)}
\end{aligned}$$

$$942 + 6 + 397 + 243 + 3,747 = \mathbf{5,335}$$

Every single image in the upstream archive is accounted for with an immutable role and lineage record in [`outputs/audit/v4_group_members.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_group_members.csv).

---

## 5. Candidate Group-Level Stratified Split (Seed 42)

Dataset partitioning was performed strictly at the **source-group level** across all 3,747 retained groups, stratified by the disease class of each group's canonical representative ([`outputs/audit/final_clean_split_manifest_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_manifest_v4.csv)):
- **Train:** 2,622 images (**69.98%**)
- **Validation:** 562 images (**15.00%**)
- **Test:** 563 images (**15.03%**)

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
| **Retinal Detachment** | 63 | 13 | 14 | 90 |
| **Central Serous Chorioretinopathy [Color Fundus]** | 49 | 11 | 11 | 71 |
| **Pterygium** | 10 | 2 | 2 | 14 |
| **Total** | **2,622** | **562** | **563** | **3,747** |

---

## 6. Comprehensive Traceability: 86 V3 Relationships Reconciled

To guarantee full traceability, all **86 high-confidence cross-split burst relationships** discovered in V3 were audited individually against V4 group assignments ([`outputs/audit/v4_v3_reconciliation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_v3_reconciliation.csv)):

| V4 Pair Resolution Status | Pair Count | Description |
|---|:---:|---|
| **`CONFLICT_QUARANTINED` $\leftrightarrow$ `CONFLICT_QUARANTINED`** | 13 | Cross-class conflict pairs; both images quarantined, 0 in split |
| **`CHAIN_QUARANTINED` $\leftrightarrow$ `CHAIN_QUARANTINED`** | 5 | Ambiguous chain pairs from `SRC_GROUP_0007`; both images quarantined, 0 in split |
| **`BURST_REDUNDANT` $\leftrightarrow$ `BURST_REDUNDANT`** | 7 | Secondary burst frames within larger groups; both excluded, 0 in split |
| **`REPRESENTATIVE` $\leftrightarrow$ `BURST_REDUNDANT`** | 61 | Exactly 1 canonical representative retained in exactly 1 split; redundant copy excluded |
| **Total V3 Pairs Reconciled** | **86** | **Zero Cross-Split Leakage (`RESOLVED_NO_LEAKAGE`: 86/86)** |

---

## 7. Independent Leakage Validation & Top-50 Nearest Neighbor Audit

Validation checks were executed across all split boundaries ([`outputs/audit/final_clean_split_validation_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_validation_v4.csv)):

| Verification Check | Observed Value | Expected Value | Status |
|---|:---:|:---:|:---:|
| **MD5 Cross-Split Overlap** | **0** | 0 | **PASS** |
| **Source Group ID Cross-Split Overlap** | **0** | 0 | **PASS** |
| **V3 High-Confidence Burst Pairs Crossing Split** | **0** | 0 | **PASS** |
| **Quarantined Images Present in Split** | **0** | 0 | **PASS** |
| **Unique Source Group ID per Retained Image** | **3,747 / 3,747** | 3,747 | **PASS** |
| **Mathematical Reconciliation to Raw 5,335** | **5,335** | 5,335 | **PASS** |
| **Expanded DINOv2 Top-50 Nearest Neighbor Audit** | **44,275 pairs** | Audited | **AUDITED** |

### Expanded DINOv2 Top-50 Nearest Neighbor Audit:
For every one of the 3,747 retained images, its top-50 nearest neighbors in normalized DINOv2 feature space ($S_{ij} = u_i \cdot u_j$) were computed across the candidate split. Among the 44,275 cross-split candidate pairs examined:
- **Zero** confirmed same-source or high-confidence burst sequences cross split boundaries.
- The highest cosine similarities crossing boundaries reflect genuine anatomical features of the fundus (e.g. optic disc cup-to-disc ratio in glaucoma or general orange retinal background reflectance) rather than reused photograph sessions.

---

## 8. Crucial Scientific Disclaimers & Grounding

1. **No Claim of Patient-Level Independence:** The upstream Mendeley repository does not include patient or laterality identifiers. The V4 benchmark guarantees **image-level isolation and source-group separation**, but cannot guarantee patient-level independence.
2. **Independent Forensic Transformation Audit:** Pretrained DINOv2 representations provide an independent transformation-robust audit against perceptual hash blind spots. It is not presented as mathematical proof that zero unseen transformations exist.
3. **Severe Class Imbalance:** Pterygium contains only 14 retained images (10 Train, 2 Val, 2 Test). Macro-averaged F1 and balanced accuracy are mandatory evaluation metrics.
4. **Historical Benchmark Integrity:** All legacy files, checkpoints, and predictions under `data/` remain untouched for historical evaluation.

---

## 9. Final Decision & Status

# **`NOT_READY_FOR_CLEAN_DATA_BUILD`**

### Summary of Current State:
- All 86 V3 relationships have been individually reconciled with zero cross-split leakage.
- Ambiguous chain `SRC_GROUP_0007` (7 images) has been completely quarantined.
- Robust optical quality scoring (denoised Laplacian $\times$ FOV $\times$ saturation penalty) has been verified.
- Candidate clean split manifest contains 3,747 images across 10 classes with 100% mathematical reconciliation.
- **Physical directory creation of `data_clean/` remains paused awaiting user authorization.**

---

## 10. Produced Audit Artifacts

1. [`docs/audit/SOURCE_GROUP_RESOLUTION_AUDIT_V4.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/audit/SOURCE_GROUP_RESOLUTION_AUDIT_V4.md): Full markdown audit report.
2. [`outputs/audit/v4_source_groups.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_source_groups.csv): Catalog of all 3,811 source groups, sizes, member lists, and coherence statuses.
3. [`outputs/audit/v4_group_members.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_group_members.csv): Complete 4,387-image lineage, group IDs, roles, and denoised optical quality scores.
4. [`outputs/audit/v4_quarantine.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_quarantine.csv): Catalog of 397 quarantined images (390 cross-class conflicts + 7 ambiguous chain frames).
5. [`outputs/audit/v4_v3_reconciliation.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/v4_v3_reconciliation.csv): Individual pair-by-pair reconciliation table for all 86 V3 burst pairs.
6. [`outputs/audit/final_clean_split_manifest_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_manifest_v4.csv): Candidate clean split manifest containing 3,747 images (2,622 Train, 562 Val, 563 Test).
7. [`outputs/audit/final_clean_split_validation_v4.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/final_clean_split_validation_v4.csv): Automated test table confirming zero leakage.
