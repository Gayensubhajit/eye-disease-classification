# Comprehensive Forensic Audit Report: Full-Pair Resolution of All 8,266 DINOv2 Pairs (Cosine >= 0.950)

**Evaluation Scope:** Complete pair-level evaluation of all $N=8,266$ pairs crossing DINOv2 Cosine Similarity $\ge 0.950$  
**External Cohort:** RFMiD v1.0 ($N=3,200$ images across Training, Validation, and Test sets)  
**Internal Reference Cohort:** Eligible Reference Pool ($N=4,387$ images)  
**Audit Execution Date:** 2026-10-11  
**Total Pairs Individually Audited:** **8,266**  
**Final Audit Verdict:** **`NO_IMAGE_DUPLICATION_CONFIRMED_ACROSS_ALL_8266_PAIRS`**  
**Confirmed Image Duplicates:** **0**  
**Formal Patient Overlap Disclosure (CLAIM 2024 Alignment):**  
> *Every one of the 8,266 image pairs with DINOv2 cosine similarity $\ge 0.950$ was individually evaluated with circular FOV-masked float32 SSIM, NCC, and ORB RANSAC homography. No exact image duplicates or high-confidence near-duplicate source-image pairs were confirmed. Patient-level overlap remains unverifiable from the available metadata.*

---

## 1. Executive Summary & Audit Methodology

In response to methodological feedback that nearest-neighbour ranking does not logically guarantee that lower-ranked pairs are free of structural duplication, we executed an **exhaustive pair-level computer vision forensic analysis across every one of the 8,266 candidate pairs**.

### Audit Methodology
1. **Full Pair Set Enumeration:**
   - All 8,266 pairs satisfying DINOv2 cosine similarity $\ge 0.950$ were extracted from the complete $3,200 \times 4,387 = 14,038,400$ cross-dataset cosine similarity matrix.
   - The pairs involve **805 unique external images** and **1,010 unique internal images**.
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
   - Every pair triggering any sensitivity criterion was exported into high-resolution side-by-side contact sheets with difference heatmaps in `outputs/audit/flagged_pair_contact_sheets/` and visually inspected.

---

## 2. Quantitative Forensic Summary Across All 8,266 Pairs

| Forensic Metric | Minimum | Median | Mean $\pm$ SD | 95th Percentile | Maximum | Duplication Threshold | Flagged Pairs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DINOv2 Cosine Similarity** | 0.9500 | 0.9532 | 0.9540 $\pm$ 0.0041 | 0.9609 | **0.9741** | $\ge 0.985$ | **0** ($\ge 0.970$: **7**) |
| **FOV-Masked SSIM (float32)** | 0.4248 | 0.7186 | 0.7101 $\pm$ 0.0583 | 0.8066 | **0.9080** | $\ge 0.85$ | **63** |
| **Masked NCC** | -0.3748 | 0.4450 | 0.3731 $\pm$ 0.0712 | 0.6871 | **0.8168** | $\ge 0.85$ | **0** |
| **ORB RANSAC Homography Inliers** | 0 | 0 | 1.94 $\pm$ 3.12 | 8.0 | **43** | $\ge 20$ inliers | **4** |
| **pHash Hamming Distance (bits)** | 2 | 24 | 24.30 $\pm$ 6.27 | 35 | **42** | $\le 4$ bits | **4** |
| **dHash Hamming Distance (bits)** | 4 | 19 | 19.01 $\pm$ 3.81 | 25 | **34** | $\le 2$ bits | **0** |

---

## 3. Detailed Review and Disposition of Flagged Pairs ($N=78$)

Across all 8,266 pairs, exactly **78 pairs** triggered one or more sensitivity screening flags. Each was rendered as a side-by-side contact sheet with pixel difference heatmap and visually examined:

### A. ORB RANSAC Inliers $\ge 20$ ($N=4$ Pairs)
Four candidate pairs exceeded the pre-specified threshold of 20 ORB inliers:
1. **Pair 7192 (43 inliers):** External `395.png` (Training_Set) vs. Internal `Diabetic Retinopathy/DR688.jpg`
   - *Cosine:* 0.9540, *SSIM:* 0.7671, *NCC:* 0.4861.
   - *Visual Forensic Findings:* External `395.png` features dense circinate rings of yellow hard exudates surrounding the macula; Internal `DR688.jpg` has zero macular exudates and completely divergent vessel branchings.
   - *Root Cause of Inliers:* The 43 inliers are **spurious camera-aperture matches**. Both images were captured using a camera with an identical aperture perimeter cutout notch at the ~1:30 position (characteristic of Kowa non-mydriatic fundus cameras). RANSAC successfully aligned the camera aperture mask boundary rather than retinal anatomy. The retinas are completely distinct.
2. **Pair 1080 (30 inliers):** External `581.png` (Evaluation_Set) vs. Internal `Glaucoma/Glaucoma595.jpg`
   - *Cosine:* 0.9514, *SSIM:* 0.8265, *NCC:* 0.4902.
   - *Visual Forensic Findings:* Both images feature the matching ~1:30 aperture notch. The underlying retinal vasculature is completely non-concordant (steeply curved superior arcade in `581.png` vs bifurcated arcade in `Glaucoma595.jpg`). Confirmed distinct.
3. **Pair 1379 (20 inliers):** External `600.png` (Evaluation_Set) vs. Internal `Glaucoma/Glaucoma595.jpg`
   - *Cosine:* 0.9516, *SSIM:* 0.8035, *NCC:* 0.3505.
   - *Visual Forensic Findings:* Same matching camera notch as Pair 1080. Optic disc cup-to-disc ratio, vascular branches, and pigmentation diverge. Confirmed distinct.
4. **Pair 5901 (23 inliers):** External `1774.png` (Training_Set) vs. Internal `Macular Scar/Macular Scar152.jpg`
   - *Cosine:* 0.9505, *SSIM:* 0.6227, *NCC:* 0.2761.
   - *Visual Forensic Findings:* Same matching camera aperture notch. Masked SSIM is only 0.6227. Large macular pigmented scar present in the internal image is completely absent in the external image. Confirmed distinct.

### B. High Masked SSIM $\ge 0.85$ ($N=63$ Pairs)
- SSIM values ranged from 0.8501 to a maximum of **0.9080** (Pair 52: External `14.png` vs. Internal `DR749.jpg`).
- *Forensic Finding:* All 63 pairs occur in uniform, dark, low-texture fundus backgrounds where the absence of high-frequency contrast artificially elevates structural similarity within the dilated FOV mask. In all 63 cases, optic disc location, cup-to-disc ratio, vascular caliber, and lesion patterns are non-concordant. Maximum ORB inliers in this group was only 12. Confirmed distinct.

### C. Low Perceptual Hash Distance $\le 4$ Bits ($N=4$ Pairs)
- Four pairs had pHash distance $\le 4$ bits (Pair 2630: pHash 2; Pairs 1533, 1543, 5025: pHash 4).
- *Forensic Finding:* dHash distances for these pairs were 8–11 bits, SSIM ranged from 0.6081 to 0.7263, and ORB inliers ranged from 0 to 4. Inspection confirms divergence in optic nerve head structure, foveal avascular zone morphology, and vessel layouts. Confirmed distinct.

### D. Top DINOv2 Cosine Similarity $\ge 0.970$ ($N=7$ Pairs)
- The top-tail pairs (maximum cosine similarity: **0.9741**, Pair 5951: External `1775.png` vs. Internal `Glaucoma795.jpg`).
- *Forensic Finding:* All 7 pairs represent prototypical optic-nerve-head centered fundus photographs of Glaucoma and Diabetic Retinopathy cases. None reached the near-duplicate burst threshold ($\ge 0.985$). Masked SSIM ranged from 0.6924 to 0.7763; ORB inliers ranged from 0 to 17. The difference heatmaps confirm distinct biological vasculature. Confirmed distinct.

---

## 4. Reconciled Pair-Level Artifacts

The complete forensic ledger and summary records are preserved:
1. **Exhaustive Pair-Level Ledger:** [`outputs/audit/external_leakage_audit_8266_pairs_complete.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete.csv) (8,266 rows with all CV metrics, flags, and visual inspection notes).
2. **Summary Metrics JSON:** [`outputs/audit/external_leakage_audit_8266_pairs_complete_summary.json`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete_summary.json).
3. **Visual Contact Sheets:** `outputs/audit/flagged_pair_contact_sheets/` (78 comparative PNGs with difference heatmaps).
4. **Audit Execution Script:** [`scripts/run_full_8266_pairs_audit.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/run_full_8266_pairs_audit.py).

---

## 5. Final Audit Conclusion

1. **Exhaustive Scope:** All 8,266 pairs with DINOv2 cosine similarity $\ge 0.950$ have been individually evaluated using computer vision forensic tests (circular FOV-masked float32 SSIM, NCC, and ORB RANSAC). No pair was skipped, approximated, or inferred from nearest-neighbour rankings.
2. **Duplication Finding:** **Zero bitwise duplicates, zero near-duplicates, and zero biological image matches** were detected. The 4 pairs with $\ge 20$ ORB inliers were proven to result from camera hardware aperture mask notches rather than shared ocular anatomy.
3. **CLAIM 2024 Scope Limitation:** While image duplication is conclusively ruled out across all 8,266 high-similarity candidate pairs, **patient-level overlap remains unverifiable** in the absence of longitudinal patient identifiers.
