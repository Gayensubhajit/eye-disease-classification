# Three-Pillar Pre-Inference External Data Leakage Audit Report

**Evaluation Dataset:** Retinal Fundus Multi-Disease Image Dataset (RFMiD v1.0; $N=3,200$)  
**Internal Reference Cohorts:**
- Active Clean Benchmark V4: `data_clean/` ($N=3,747$; Train 2,626, Val 560, Test 561)
- Pre-Filtering Eligible Pool: `clean_split_manifest.csv` ($N=4,387$; includes retained cohort prior to partitioning)  
**Audit Execution Date:** 2026-10-11  
**Final Audit Verdict:** **`NO_IMAGE_DUPLICATION_CONFIRMED_IN_RECORDED_CHECKS`**  
**Visually Inspected Flagged Pairs:** **7** (`NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`)  
**Automated-Screening-Only Flagged Pairs:** **71** (`AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`)  
**Unflagged Candidate Pairs:** **8,188** (`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`; individual visual review not performed)  
**Formal Patient Overlap Statement (CLAIM 2024 Alignment):**  
> *Among the 14,038,400 possible cross-dataset image pairs (3,200 RFMiD images × 4,387 internal reference images), 8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of $\ge 0.950$ and were computationally evaluated with circular FOV-masked float32 SSIM, NCC, and ORB RANSAC homography. No exact image duplicates or near-duplicate source pairs were confirmed in the recorded checks. No duplicate was observed during direct visual review of seven selected boundary pairs. The remaining 71 flagged pairs are pending individual manual review. The other 8,188 candidates did not trigger the configured screening flags; they were not individually visually reviewed. Patient-level overlap remains unverifiable from the available metadata.*

**Internal Test Set Historical Exposure Disclosure:**  
> *The internal benchmark test partition (`data_clean/test`, $N=561$ nine-class posterior-segment fundus images) was not evaluated during the 9-class baseline training or during this RFMiD cross-dataset audit run. However, the underlying images were previously evaluated in earlier 10-class baseline experiments in the project's development history. Therefore, the internal test partition cannot be claimed as a pristine, never-before-accessed confirmatory test cohort. Any future confirmatory validation will require an independent external cohort.*

**Inference History & Correction Safeguards:**  
> *No model retraining, new model inference, or threshold changes were performed during the final audit-correction pass. An earlier exploratory inference run evaluated the frozen checkpoints on all 3,200 RFMiD images; those external-evaluation results remain exploratory. The internal test cohort was also evaluated during earlier ten-class baseline experiments and is not a pristine confirmatory benchmark.*

---

## 1. Audit Denominators & Screening Summary by Pillar

To ensure conservative screening against potential cross-dataset contamination, the audit utilized both the active 3,747-image cleaned benchmark and the broader 4,387-image eligible pool. The exact comparison denominator and screening threshold for each pillar are detailed below:

| Audit Pillar | Internal Reference Cohort | External Cohort | Pair Comparisons Evaluated | Screening Criteria / Threshold | Candidates Triggered | Confirmed Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pillar 1A: SHA-256 Bitwise** | Active Clean (`data_clean`, $N=3,747$) | RFMiD ($N=3,200$) | $3,200 \times 3,747 = \mathbf{11,990,400}$ | Exact 256-bit cryptographic digest match | 0 | 0 (**PASS**) |
| **Pillar 1B: MD5 Bitwise** | Eligible Pool ($N=4,387$) | RFMiD ($N=3,200$) | $3,200 \times 4,387 = \mathbf{14,038,400}$ | Exact 128-bit MD5 digest match | 0 | 0 (**PASS**) |
| **Pillar 2: Perceptual Hashes** | Eligible Pool ($N=4,387$) | RFMiD ($N=3,200$) | $3,200 \times 4,387 = \mathbf{14,038,400}$ | $d_{\text{dHash}} \le 2$, $d_{\text{pHash}} \le 3$, or ($d_{\text{dHash}} \le 4 \land d_{\text{pHash}} \le 4$) | 11 | 0 (**PASS**) |
| **Pillar 3: DINOv2 Latent Cosine** | Eligible Pool ($N=4,387$) | RFMiD ($N=3,200$) | $3,200 \times 4,387 = \mathbf{14,038,400}$ | Cosine Sim $\ge 0.950$ (exhaustive pair evaluation trigger) | 8,266 | 0 (**PASS**) |

*Note on Denominators:* Screening against the 4,387-image eligible pool (rather than solely the 3,747 retained images) provides a strictly conservative guarantee: if any RFMiD image had entered any part of the source repository pool—even images held in reserve or later quarantined—it would be detected.

---

## 2. DINOv2 Latent Space Distribution & Threshold Reconciliation

Among the 14,038,400 possible cross-dataset image pairs (3,200 RFMiD images × 4,387 internal reference images), 8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of $\ge 0.950$ and underwent pair-level computer-vision screening:
- **Pairs with Cosine Similarity $\ge 0.950$:** **8,266** pairs ($0.059\%$ of all pairs).
- **Pairs with Cosine Similarity $\ge 0.960$:** **550** pairs ($0.0039\%$ of all pairs).
- **Pairs with Cosine Similarity $\ge 0.970$:** **7** pairs.
- **Pairs with Cosine Similarity $\ge 0.980$:** **0** pairs.
- **Pairs with Cosine Similarity $\ge 0.985$ (Automatic near-duplicate trigger):** **0** pairs.
- **Maximum Cosine Similarity Observed:** **0.9741**.

```
Cosine Similarity Distribution Across Top-1 Nearest Neighbors:
  Minimum:            0.7608
  25th Percentile:    0.9170
  Median:             0.9369
  75th Percentile:    0.9500
  95.0th Percentile:  0.9606
  99.0th Percentile:  0.9660
  Maximum:            0.9741
```

### Anatomic Proximity vs. Source-Image Duplication

In self-supervised vision models such as Meta AI's DINOv2 (`vit_small_patch14_dinov2.lvd142m`), standard retinal fundus photographs naturally concentrate in a narrow latent region. All fundus photographs share global morphological features: circular aperture masks, yellowish optic discs at the nasal or temporal margins, darker central maculae, and prominent major temporal vascular arcades.

Consequently, cosine similarity values between $0.92$ and $0.96$ reflect standard population-level anatomical similarity across well-centered normal or mildly diseased fundi, rather than duplicate captures or image reuse. High cosine similarity alone cannot be interpreted as proof of duplication.

---

## 3. Deep Computer Vision Forensic Verification: Exhaustive Full-Pair Resolution ($N=8,266$)

### A. Full Pair-Level Evaluation Across All 8,266 Candidate Pairs
To ensure complete forensic closure and eliminate reliance on nearest-neighbour rank-preservation assumptions, **every one of the 8,266 image pairs with DINOv2 cosine similarity $\ge 0.950$ was individually evaluated** with circular FOV-masked `float32` SSIM, NCC, pixel difference, and ORB RANSAC homography.

- **Candidate Space:** Among the 14,038,400 possible pairs, 8,266 pairs (0.0589%) met the DINOv2 cosine-similarity threshold of $\ge 0.950$ and underwent pair-level computer-vision screening, involving **805 unique external images** and **1,010 unique internal images**.
- **Overall Forensic Distribution Across all 8,266 Pairs:**
  - **DINOv2 Cosine Similarity:** Median: 0.9532, Mean: $0.9540 \pm 0.0041$, Max: **0.9741** (Count $\ge 0.970$: 7, Count $\ge 0.985$: 0).
  - **Circular FOV-Masked SSIM (float32):** Median: 0.7186, Mean: $0.7101 \pm 0.0583$, Max: **0.9080** (Count $\ge 0.85$: 63).
  - **Circular FOV-Masked NCC:** Median: 0.4450, Mean: $0.3731 \pm 0.0712$, Max: **0.8168** (Count $\ge 0.85$: 0).
  - **ORB RANSAC Homography Inliers:** Median: 0, Mean: $1.94 \pm 3.12$, Max: **43** (Count $\ge 20$: 4).
  - **pHash Hamming Distance:** Median: 24 bits, Mean: $24.30 \pm 6.27$ bits, Min: 2 bits (Count $\le 4$: 4).
  - **dHash Hamming Distance:** Median: 19 bits, Mean: $19.01 \pm 3.81$ bits, Min: 4 bits (Count $\le 2$: 0).

### B. Detailed Review and Disposition of Flagged Candidate Pairs ($N=78$)
Across all 8,266 pairs, exactly **78 unique pairs** triggered one or more pre-specified sensitivity screening flags (63 high SSIM, 4 ORB homography, 4 low pHash, 7 top DINOv2; exact disjoint union: 78 pairs, zero overlapping flags).

A dedicated review record ([`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv)) explicitly documents the disposition and review modality for each flagged pair:
- **Direct Manual Visual Inspection ($N=7$):** The 4 pairs with ORB inliers $\ge 20$ (Pairs 7192, 1080, 1379, 5901) alongside key metric anchors (Pair 52 for maximum SSIM, Pair 2630 for minimum pHash, and Pair 5951 for maximum DINOv2 cosine) were individually reviewed via high-resolution contact sheets, difference heatmaps, and keypoint spatial distributions. Disposition: **`NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`** (7 pairs).
- **Automated Metric Screening Only; Manual Review Pending ($N=71$):** The remaining 71 candidate pairs (62 high-SSIM, 3 low-pHash, 6 top-DINOv2) were evaluated via quantitative threshold screening bounds (e.g. ORB inliers < 20, NCC < 0.85), with individual comparative contact sheets and difference heatmaps generated and preserved on disk. Disposition: **`AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING`** (71 pairs). These pairs are not represented as manually verified or confirmed independent retinas; they remain pending individual manual review under the existing protocol.
- **Unflagged Candidate Pairs ($N=8,188$):** The remaining 8,188 candidate pairs did not trigger any configured screening flags and are classified as **`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`**. Individual visual review was not performed for these pairs. This label indicates strictly that the pair did not meet any configured sensitivity screening flags (observed metric distributions: median FOV-masked SSIM 0.7186, median masked NCC 0.4450, median ORB inliers 0). It is not interpreted as proof of distinct retinas or patient independence.

1. **ORB RANSAC Inliers $\ge 20$ ($N=4$ Pairs):**
   - **Pair 7192 (43 inliers):** External `395.png` (Training_Set) vs. Internal `DR688.jpg`. External image exhibits dense circinate rings of yellow hard exudates around the macula; internal image has zero exudates. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
   - **Pair 1080 (30 inliers):** External `581.png` (Evaluation_Set) vs. Internal `Glaucoma595.jpg`. Retinal vascular arcades and optic disc cup diverge. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
   - **Pair 1379 (20 inliers):** External `600.png` (Evaluation_Set) vs. Internal `Glaucoma595.jpg`. Optic cup geometry and vessel caliber diverge. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
   - **Pair 5901 (23 inliers):** External `1774.png` (Training_Set) vs. Internal `Macular Scar152.jpg`. Dense pigmented scar present internally, absent externally; masked SSIM is only 0.6227 and NCC is 0.2761. Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
   - *Aperture Boundary & Geometry Findings:* In Pairs 7192, 1080, and 1379, keypoint coordinates cluster tightly at the circular aperture mask boundary notch ($x \approx 366-392, y \approx 55-84$) rather than retinal structures. In Pair 5901, matches fall in central low-contrast fundus without geometric or vascular concordance. The available image and acquisition metadata do not establish the precise camera-specific cause. The underlying biological retinal vasculature, optic disc, and lesions are completely divergent.
2. **High Masked SSIM $\ge 0.85$ ($N=63$ Pairs):**
   - SSIM ranged from 0.8501 to 0.9080 (max: Pair 52, External `14.png` vs. Internal `DR749.jpg`).
   - Pair 52 was individually visually inspected: high SSIM is driven by large uniform untextured background fundus fields; retinal vessel trees, microaneurysms, and color hue completely diverge (ORB inliers=9; Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`).
   - The remaining 62 pairs met automated screening criteria (ORB inliers 0–12, NCC < 0.85) with contact sheets rendered on disk; Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=62$).
3. **Low Perceptual Hash Distance $\le 4$ Bits ($N=4$ Pairs):**
   - Pair 2630 (pHash 2 bits) was individually visually inspected: distinct optic disc angle, neuroretinal rim thinning, and superior arcade bifurcation (SSIM 0.6081, ORB inliers 4; Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`).
   - Pairs 1533, 1543, 5025 (pHash 4 bits) met automated screening bounds (dHash 8–11 bits, SSIM 0.6548–0.7263, ORB inliers 0–3); Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=3$).
4. **Top DINOv2 Cosine Similarity $\ge 0.970$ ($N=7$ Pairs):**
   - Pair 5951 (maximum cosine similarity 0.9741) was individually visually inspected: standard optic-nerve-head centered fundus; peripapillary atrophy halo and vessel layout diverge; ORB inliers=0; Disposition: `NO_DUPLICATE_OBSERVED_ON_VISUAL_REVIEW`.
   - The remaining 6 pairs met automated screening bounds (DINOv2 cosine 0.9701–0.9723, SSIM 0.6924–0.7763, ORB inliers 0–17); Disposition: `AUTOMATED_SCREEN_ONLY_MANUAL_REVIEW_PENDING` ($N=6$).

Complete ledgers are preserved in [`outputs/audit/external_leakage_audit_8266_pairs_complete.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete.csv) (8,266 rows), [`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv) (78 rows), and detailed in [`docs/audit/DINOV2_HIGH_SIMILARITY_AUDIT_REPORT.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/audit/DINOV2_HIGH_SIMILARITY_AUDIT_REPORT.md). Contact sheets for all 78 flagged pairs are saved on disk in `outputs/audit/flagged_pair_contact_sheets/` (56 MB; excluded from git tracking via `.gitignore` to prevent repository binary bloat, and fully reproducible via the audit script).

---

## 4. Methodological Conclusion & CLAIM 2024 Alignment

1. **Image-Level Duplication:**  
   The audit confirms **zero bitwise duplicates** across 11,990,400 SHA-256 and 14,038,400 MD5 comparisons, and **zero near-duplicate source-image pairs** confirmed across perceptual hashing and deep CV feature matching.
2. **Review Status of Candidate Pairs:**  
   Among the 8,266 pairs crossing DINOv2 cosine $\ge 0.950$, 7 boundary pairs were visually inspected with no duplicate observed. The remaining 71 flagged candidate pairs (including 62 high-SSIM cases) met automated screening bounds and are preserved as pending individual manual review under the existing protocol. The 8,188 unflagged pairs did not trigger any configured screening flags (`AUTOMATED_SCREEN_NO_CONFIGURED_FLAG`; individual visual review not performed).
3. **Patient-Level Independence Limitation:**  
   In compliance with the CLAIM 2024 reporting guideline (Checklist for Artificial Intelligence in Medical Imaging), we explicitly report that because neither dataset provides de-identified master patient indexes or longitudinal clinical records, patient-level independence cannot be mathematically proven.
4. **Audit Verdict:**  
   **`NO_IMAGE_DUPLICATION_CONFIRMED_IN_RECORDED_CHECKS`**. The RFMiD cohort is verified to be free of confirmed image-level duplication in the recorded checks, with 71 flagged pairs pending manual review.
