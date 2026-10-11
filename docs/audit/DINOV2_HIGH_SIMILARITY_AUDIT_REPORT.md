# Comprehensive Forensic Audit Report: Full-Pair Resolution of All 8,266 DINOv2 Pairs (Cosine >= 0.950)

**Evaluation Scope:** Complete pair-level evaluation of all $N=8,266$ pairs crossing DINOv2 Cosine Similarity $\ge 0.950$  
**External Cohort:** RFMiD v1.0 ($N=3,200$ images across Training, Validation, and Test sets)  
**Internal Reference Cohort:** Eligible Reference Pool ($N=4,387$ images)  
**Audit Execution Date:** 2026-10-11  
**Total Pairs Individually Audited:** **8,266**  
**Final Audit Verdict:** **`NO_IMAGE_DUPLICATION_CONFIRMED_IN_RECORDED_CHECKS`**  
**Confirmed Image Duplicates:** **0**  
**Visually Inspected Flagged Pairs:** **7** (`NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`)  
**Automated-Screening-Only Flagged Pairs:** **71** (`AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`)  
**Unflagged Candidate Pairs:** **8,188** (`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`; individual visual review not performed)  
**Formal Patient Overlap Disclosure (CLAIM 2024 Alignment):**  
> *Among the 14,038,400 possible cross-dataset image pairs (3,200 RFMiD images × 4,387 internal reference images), 8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of $\ge 0.950$ and were computationally evaluated with circular FOV-masked float32 SSIM, NCC, and ORB RANSAC homography. No exact image duplicates or near-duplicate source pairs were confirmed in the recorded checks. No duplicate was observed during direct visual review of seven selected boundary pairs. The remaining 71 flagged pairs are pending individual manual review. The other 8,188 candidates did not trigger the configured screening flags; they were not individually visually reviewed. Patient-level overlap remains unverifiable from the available metadata.*

**Internal Test Set Historical Exposure Disclosure:**  
> *The internal benchmark test partition (`data_clean/test`, $N=561$ nine-class posterior-segment fundus images) was not evaluated during the 9-class baseline training or during this RFMiD cross-dataset audit run. However, the underlying images were previously evaluated in earlier 10-class baseline experiments in the project's development history. Therefore, the internal test partition cannot be described as a pristine, never-before-accessed confirmatory test cohort. Any future confirmatory validation will require an independent external cohort.*

**Inference History & Correction Safeguards:**  
> *No model retraining, new model inference, or threshold changes were performed during the final audit-correction pass. An earlier exploratory inference run evaluated the frozen checkpoints on all 3,200 RFMiD images; those external-evaluation results remain exploratory. The internal test cohort was also evaluated during earlier ten-class baseline experiments and is not a pristine confirmatory benchmark.*

---

## 1. Executive Summary & Audit Methodology

In response to methodological feedback that nearest-neighbour ranking does not logically guarantee that lower-ranked pairs are free of structural duplication, we executed an **exhaustive pair-level computer vision forensic analysis across every one of the 8,266 candidate pairs**.

### Audit Methodology
1. **Full Pair Set Enumeration:**
   - Among the 14,038,400 possible cross-dataset image pairs (3,200 RFMiD images × 4,387 internal reference images), 8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of $\ge 0.950$ and underwent pair-level computer-vision screening.
   - The pairs involve **805 unique external images** and **1010 unique internal images**.
2. **Individual Pair-Level Computation:**
   - Every single pair was individually loaded and tested with:
     - Circular FOV-masked `float32` SSIM (OpenCV Gaussian blurring with window 11x11, sigma 1.5, eroded mask)
     - Circular FOV-masked Normalized Cross-Correlation (NCC)
     - Mean absolute pixel difference
     - ORB keypoint matching with Lowe's ratio test ($< 0.75$) and RANSAC geometric homography inlier estimation
     - pHash and dHash Hamming distances
3. **Threshold Flags & Systematic Investigation:**
   - Pre-specified sensitivity review criteria were tracked for all 8,266 pairs:
     - `HIGH_SSIM`: Masked SSIM $\ge 0.85$
     - `HIGH_NCC`: Masked NCC $\ge 0.85$
     - `ORB_HOMOGRAPHY`: ORB RANSAC inliers $\ge 20$
     - `LOW_PHASH`: pHash distance $\le 4$ bits
     - `LOW_DHASH`: dHash distance $\le 2$ bits
     - `TOP_DINO`: DINOv2 Cosine Similarity $\ge 0.970$
     - `DINO_BURST`: DINOv2 Cosine Similarity $\ge 0.985$
   - All pairs triggering any sensitivity criterion were exported into high-resolution side-by-side contact sheets with difference heatmaps in `outputs/audit/flagged_pair_contact_sheets/` and evaluated via a structured review protocol.

---

## 2. Quantitative Forensic Summary Across All 8,266 Pairs

| Forensic Metric | Minimum | Median | Mean $\pm$ SD | 95th Percentile | Maximum | Duplication Threshold | Flagged Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DINOv2 Cosine Similarity** | 0.9500 | 0.9532 | 0.9540 $\pm$ 0.0035 | 0.9609 | **0.9741** | $\ge 0.985$ | **0** ($\ge 0.970$: **7**) |
| **FOV-Masked SSIM (float32)** | 0.4248 | 0.7186 | 0.7101 $\pm$ 0.0678 | 0.8066 | **0.9080** | $\ge 0.85$ | **63** |
| **Masked NCC** | -0.3748 | 0.4450 | 0.3731 $\pm$ 0.2517 | 0.6871 | **0.8168** | $\ge 0.85$ | **0** |
| **ORB RANSAC Homography Inliers** | 0 | 0 | 1.94 $\pm$ 3.19 | 8.0 | **43** | $\ge 20$ inliers | **4** |
| **pHash Hamming Distance (bits)** | 2 | 24 | 24.30 $\pm$ 6.27 | 34.0 | **42** | $\le 4$ bits | **4** |
| **dHash Hamming Distance (bits)** | 4 | 19 | 19.01 $\pm$ 4.93 | 27.0 | **34** | $\le 2$ bits | **0** |

---

## 3. Detailed Review and Disposition of Flagged Pairs ($N=78$)

Across all 8,266 pairs, exactly **78 pairs** triggered one or more sensitivity screening flags (63 high SSIM, 4 ORB homography, 4 low pHash, 7 top DINOv2; exact disjoint union: 78 pairs, zero overlapping flags).

A dedicated review record ([`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv)) explicitly distinguishes the pairs evaluated via direct visual review from those evaluated solely via automated metric screening:
- **Direct Manual Visual Inspection ($N=7$):**
  - The 4 pairs with ORB inliers $\ge 20$ (Pairs 7192, 1080, 1379, 5901) and 3 boundary metric anchors (Pair 52 for max SSIM, Pair 2630 for min pHash, Pair 5951 for max DINOv2) underwent direct manual visual inspection of high-resolution contact sheets and spatial keypoint distributions.
  - Disposition: **`NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`** (7 pairs; `visual_review_performed = True`).
- **Automated Metric Screening Only; Manual Review Pending ($N=71$):**
  - The remaining 71 candidate pairs (62 high-SSIM, 3 low-pHash, 6 top-DINOv2) were evaluated via quantitative threshold screening bounds (e.g. ORB inliers < 20, NCC < 0.85).
  - Disposition: **`AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`** (71 pairs; `visual_review_performed = False`). While comparative contact sheets and difference heatmaps were generated on disk, these 71 pairs are not represented as manually verified or confirmed independent retinas; they remain pending individual manual review under the existing protocol.

### A. Direct Visual Review of High-ORB Pairs ($N=4$)
Four candidate pairs exceeded the pre-specified threshold of 20 ORB inliers and were individually visually inspected:
1. **Pair 7192 (43 inliers):** External `395.png` (Training_Set) vs. Internal `Diabetic Retinopathy/DR688.jpg`
   - *Cosine:* 0.9540, *SSIM:* 0.7671, *NCC:* 0.4861.
   - *Visual Findings:* External `395.png` features dense circinate rings of yellow hard exudates surrounding the macula; Internal `DR688.jpg` has zero macular exudates and completely divergent vessel branchings.
   - *Aperture Boundary Analysis:* Spatial examination of inlier coordinates demonstrates that the keypoints concentrate tightly around the circular aperture mask boundary (specifically at the top-right cutout notch at $x \approx 366-392, y \approx 55-84$) rather than corresponding retinal structures. The available image and acquisition metadata do not establish the precise camera-specific cause. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
2. **Pair 1080 (30 inliers):** External `581.png` (Evaluation_Set) vs. Internal `Glaucoma/Glaucoma595.jpg`
   - *Cosine:* 0.9514, *SSIM:* 0.8265, *NCC:* 0.4902.
   - *Visual Findings:* Keypoint matches concentrate around the circular aperture boundary notch. Retinal vasculature is completely non-concordant (steeply curved superior arcade in `581.png` vs bifurcated arcade in `Glaucoma595.jpg`; cup depth and disc margins diverge). Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
3. **Pair 1379 (20 inliers):** External `600.png` (Evaluation_Set) vs. Internal `Glaucoma/Glaucoma595.jpg`
   - *Cosine:* 0.9516, *SSIM:* 0.8035, *NCC:* 0.3505.
   - *Visual Findings:* Inliers concentrate around the aperture boundary notch. Optic disc cup-to-disc ratio, vascular branches, and retinal pigmentation diverge. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
4. **Pair 5901 (23 inliers):** External `1774.png` (Training_Set) vs. Internal `Macular Scar/Macular Scar152.jpg`
   - *Cosine:* 0.9505, *SSIM:* 0.6227, *NCC:* 0.2761.
   - *Visual Findings:* Keypoints lie in low-contrast central fundus regions ($r \approx 70-186\text{ px}$) but exhibit low masked SSIM (0.6227) and low masked NCC (0.2761). Direct visual inspection confirms a large pigmented macular scar present internally that is completely absent externally, alongside non-matching vascular arborization. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.

### B. High Masked SSIM $\ge 0.85$ ($N=63$ Pairs)
- SSIM values ranged from 0.8501 to a maximum of **0.9080** (Pair 52: External `14.png` vs. Internal `DR749.jpg`).
- **Review Breakdown:**
  - **Pair 52 (Maximum SSIM 0.9080):** Individually visually inspected. High SSIM is driven by uniform, untextured peripheral fundus fields within the dilated FOV mask. Retinal vessel trees, microaneurysms, and disc cups completely diverge (ORB inliers=9). Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
  - **Remaining 62 Pairs:** Automated screening only. ORB inliers ranged from 0 to 12 (median: 0), masked NCC ranged from -0.1866 to 0.5654. Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=62$).

### C. Low Perceptual Hash Distance $\le 4$ Bits ($N=4$ Pairs)
- Four pairs had pHash distance $\le 4$ bits:
  - **Pair 2630 (Minimum pHash 2 bits):** Individually visually inspected. Optic disc tilt, neuroretinal rim contour, and arcade bifurcation diverge; SSIM is 0.6081 and ORB inliers=4. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
  - **Pairs 1533, 1543, 5025 (pHash 4 bits):** Automated screening only (dHash 8–11 bits, SSIM 0.6548–0.7263, ORB inliers 0–3). Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=3$).

### D. Top DINOv2 Cosine Similarity $\ge 0.970$ ($N=7$ Pairs)
- Seven pairs had DINOv2 cosine similarity $\ge 0.970$:
  - **Pair 5951 (Maximum DINOv2 Cosine 0.9741):** Individually visually inspected. Standard optic-nerve-head centered fundus; peripapillary atrophy halo and vessel layout diverge; ORB inliers=0. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
  - **Remaining 6 Pairs:** Automated screening only (DINOv2 cosine 0.9701–0.9723, SSIM 0.6924–0.7763, ORB inliers 0–17). Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=6$).

### E. Unflagged Candidate Pairs ($N=8188$)
- Exactly 8,188 candidate pairs did not trigger any configured screening flags and are classified as **`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`**.
- **Visual Review Status:** Individual visual review was not performed for these pairs (`visual_review_performed = False`).
- **Empirical Metric Distributions:** Observed metric distributions for this unflagged cohort: median FOV-masked SSIM: 0.7186 (mean: 0.7101 $\pm$ 0.0678), median masked NCC: 0.4450 (mean: 0.3731 $\pm$ 0.2517), median ORB inliers: 0 (mean: 1.94 $\pm$ 3.19).
- **Scope Limitation:** The label `AUTOMATED_SCREEN_NO_CONFIGURED_FLAG` denotes strictly that the pair did not meet any configured sensitivity threshold. It is not interpreted as positive proof of distinct retinas, biological independence, or patient-level independence.

---

## 4. Reconciled Pair-Level Artifacts

The complete forensic ledger and summary records are preserved:
1. **Exhaustive Pair-Level Ledger:** [`outputs/audit/external_leakage_audit_8266_pairs_complete.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete.csv) (8,266 rows with all CV metrics, flags, review modality tags, `visual_review_performed` indicator, notes, and individual dispositions: 7 `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`, 71 `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`, 8,188 `AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`).
2. **Dedicated Flagged Pairs Review Ledger:** [`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv) (78 rows detailing review modality, `visual_review_performed` indicator, forensic evidence, and specific disposition: 7 `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW` vs 71 `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`).
3. **Summary Metrics JSON:** [`outputs/audit/external_leakage_audit_8266_pairs_complete_summary.json`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete_summary.json).
4. **Visual Contact Sheets:** `outputs/audit/flagged_pair_contact_sheets/` (78 comparative PNGs with difference heatmaps; 56 MB preserved locally on disk, excluded from git tracking via `.gitignore` to prevent repository binary bloat, and fully reproducible via the audit script).
5. **Audit Execution Script:** [`scripts/run_full_8266_pairs_audit.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/run_full_8266_pairs_audit.py).

---

## 5. Final Audit Conclusion

1. **Computational Scope:** All 8,266 pairs with DINOv2 cosine similarity $\ge 0.950$ have been computationally evaluated using computer vision forensic tests (circular FOV-masked float32 SSIM, NCC, and ORB RANSAC). No pair was skipped, approximated, or inferred from nearest-neighbour rankings.
2. **Duplication Finding:** **No image-level duplication was confirmed in the recorded checks.** In the 7 boundary pairs directly visually inspected, no duplicate was observed under the stated review procedure. The remaining 71 flagged candidate pairs (including 62 high-SSIM pairs) met automated screening bounds and are preserved as pending individual manual review under the existing protocol. The 8,188 unflagged pairs did not trigger any configured screening flags (`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`; individual visual review was not performed).
3. **CLAIM 2024 Scope Limitation:** While no duplication was observed in the evaluated pairs, **patient-level overlap remains unverifiable** in the absence of longitudinal patient identifiers.
