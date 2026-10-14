# RFMiD Cross-Dataset Leakage Audit: Flagged Candidate Pairs Adjudication Protocol & Review Package

**Audit Package Directory:** `outputs/audit/rfmid_flagged_pairs_review_package/`  
**Adjudication Ledger:** `flagged_pairs_adjudication_ledger.csv` ($N=78$ rows)  
**High-Resolution Contact Sheets:** `outputs/audit/flagged_pair_contact_sheets/` ($78$ PNG images, $1536 \times 592$ px)  
**Creation Date:** 2026-10-12  
**Protocol Version:** 1.0 (Manual Adjudication Phase)  
**Operational Status:** **Computational screening complete; manual adjudication pending (71 cases).**

---

## 1. Executive Summary & Review Scope

During the three-pillar external data leakage audit between the external **RFMiD v1.0 dataset** ($N=3,200$) and the **internal reference cohorts** (Canonical Clean Benchmark $N=3,747$; Pre-Filtering Eligible Pool $N=4,387$), an exhaustive computational screening was executed across all $14,038,400$ possible cross-dataset image pairs.

* **High-Similarity Candidate Population ($N=8,266$):** 8,266 candidate pairs ($0.0589\%$ of all cross-dataset pairs) crossed the DINOv2 latent cosine similarity threshold of $\ge 0.950$. Every one of these 8,266 pairs was computationally evaluated using circular FOV-masked float32 SSIM, masked NCC, and ORB RANSAC homography.
* **Flagged Sensitivity Subset ($N=78$):** Exactly 78 unique candidate pairs triggered one or more pre-specified sensitivity screening flags (63 high SSIM, 7 top DINOv2 cosine, 4 low pHash, 4 ORB homography; exact disjoint union).
* **Completed Boundary Reviews ($N=7$):** Seven boundary and anchor pairs were visually inspected and verified to represent distinct retinas (`NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`).
* **Pending Manual Reviews ($N=71$):** Exactly 71 candidate pairs triggered screening flags (62 high-SSIM, 6 top-DINOv2, 3 low-pHash) and met automated safety bounds (ORB inliers $< 20$, masked NCC $< 0.85$). They are formally logged as `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` and are compiled in this review package for clinical sign-off.

### Image-Level Accounting of the 71 Pending Cases
The 71 pending candidate pairs do **not** represent 71 distinct images:
* **Unique External RFMiD Images:** **48 unique images** (38 Training, 28 Evaluation, 5 Test across multiple pairings).
* **Unique Internal Reference Images:** **43 unique images** (32 Glaucoma, 22 DR, 14 Healthy, 2 Disc Edema, 1 Myopia across multiple pairings).
* Multiple pairings arise because certain untextured normal fundus images with average retinal pigmentation share high SSIM across multiple reference counterparts.

---

## 2. Mutually Exclusive Reconciliation of the 8,266 Candidate Pairs

To ensure strict scientific accounting where no candidate pair is double-counted, all 8,266 pairs are partitioned into three mutually exclusive categories:

| Category | Disposition Code | Pair Count | Percentage | Modality & Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Completed Visual Reviews** | `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW` | **7** | 0.08% | Direct manual inspection of contact sheets, heatmaps, and keypoint geometry. Confirmed non-duplicates. |
| **2. Flagged Candidates Awaiting Review** | `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` | **71** | 0.86% | Triggered sensitivity screening flags; met automated screening bounds; contact sheets rendered; **pending manual adjudication**. |
| **3. Unflagged Screening Candidates** | `AUTOMATED_SCREEN_NO_CONFIGURED_FLAG` | **8,188** | 99.06% | Did not trigger any configured screening flags; median masked SSIM 0.7186, median NCC 0.4450, median ORB inliers 0. **Individual visual review not performed.** |
| **Total Candidate Population** | | **8,266** | **100.00%** | Comprehensive computer-vision evaluation completed for all pairs. |

---

## 3. Biophysical Rationale for False-Positive Similarity in Fundus Photography

When screening color retinal fundus photographs using deep representation models and image comparison metrics, elevated similarity scores frequently occur between biologically independent retinas due to imaging standardization rather than data leakage:

1. **Global Structural Standardization:** All standard fundus cameras enforce a circular field-of-view aperture (typically 45° to 50°) surrounded by a solid black mask. The optic disc, macula, and major vascular arcades occupy fixed canonical geometric positions across human subjects.
2. **Spectral Homogeneity:** The neuroretinal background across healthy and mildly diseased retinas is uniformly composed of reddish-orange pigmentary epithelium.
3. **SSIM Elevation on Low-Texture Fields:** SSIM evaluates local luminance, contrast, and structural correlation. In fundus photographs with smooth, evenly illuminated backgrounds, SSIM values of $0.85 - 0.90$ are common even when fine vascular branching, optic cup morphology, and focal lesions are entirely non-concordant.
4. **Distinction Between Anatomic Proximity and True Duplication:**
   * **True Image Duplicate / Burst Capture:** Requires concordant vascular branching ($\text{NCC} \ge 0.85$), dense keypoint homography ($\text{ORB inliers} \ge 20$), and near-zero perceptual hash distance ($\text{pHash} \le 2$).
   * **Population Anatomic Similarity:** Characterized by elevated DINOv2 cosine ($\ge 0.950$) and moderate SSIM ($\ge 0.85$), but low NCC ($< 0.60$), near-zero ORB inliers ($0 - 4$), and high perceptual hash distance ($\ge 16$ bits).

---

## 4. Standardized Four-Pillar Clinical Adjudication Checklist

For each of the 71 pending pairs, the reviewer inspects the pre-rendered contact sheet in `outputs/audit/flagged_pair_contact_sheets/pair_<pair_id>_*.png` and records evaluations across four anatomical landmarks:

1. **Retinal Vascular Tree Architecture (`checklist_retinal_vasculature_diverges`):**
   * Inspect the primary superior and inferior temporal vascular arcades branching from the optic nerve head.
   * Verify whether vessel bifurcations, caliber variations, tortuosity, and arteriolar-venular crossings diverge.
   * *Criterion:* If vascular branch patterns do not align, the images represent distinct eyes.
2. **Optic Nerve Head Morphology (`checklist_optic_disc_diverges`):**
   * Inspect the optic disc contour, cup-to-disc ratio (CDR), neuroretinal rim width, and lamina cribrosa exposure.
   * Check for peripapillary atrophy (PPA) halos and crescent geometry.
   * *Criterion:* Differences in cup ovality, rim notches, or PPA distribution confirm distinct subjects.
3. **Macular & Foveal Characteristics (`checklist_macula_fovea_diverges`):**
   * Inspect the foveal avascular zone (FAZ), foveal light reflex, and luteal pigment density.
   * *Criterion:* Non-concordant foveal distance from the disc or distinct macular architecture confirms distinct retinas.
4. **Pathological Lesions & Focal Biomarkers (`checklist_pathology_lesions_diverges`):**
   * Inspect the presence, quantity, and spatial distribution of microaneurysms, dot-and-blot hemorrhages, hard circinate exudates, cotton wool spots, drusen, or choroidal neovascularization.
   * *Criterion:* Lesions present in one image but absent in the other (at corresponding anatomical coordinates) confirm independence.

---

## 5. Decision Framework & Permissible Adjudication Verdicts

For each row in `flagged_pairs_adjudication_ledger.csv`, the reviewer records one of the following decisions:

* **`DISTINCT_ANATOMY`**: Direct visual review confirms clear anatomical divergence across vascular, disc, macular, or lesion features. The candidate pair is cleared as an independent retina.
* **`CONFIRMED_DUPLICATE`**: Visual inspection confirms matching vascular branching, identical focal lesions, and identical anatomical geometry, indicating image reuse or identical patient re-imaging. *(Would require immediate quarantine of the external image).*
* **`INDETERMINATE`**: Extreme image degradation, severe media opacity, or severe underexposure prevents anatomical verification.

---

## 6. Formal Limitations & CLAIM 2024 Compliance

1. **Patient-Level Linkage Limitation:**  
   Public datasets (RFMiD and the Kaggle source pool) do not provide cross-institutional patient master identifiers. Direct visual inspection can confirm that two photographs are not identical captures or re-compressed duplicates, but **cannot definitively verify that two eyes do not belong to the same human patient** (e.g. contralateral eye or longitudinal visit under altered pathology).
2. **Scope of Claim:**  
   The audit supports: **No image-level duplication confirmed in recorded checks**. It does **not** claim complete independent patient-level validation.
