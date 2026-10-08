# Final Independent Image-Source Audit Report (V3)

**Project Title:** Classification of Eye Diseases from Color Fundus Images  
**Investigation:** Independent Transformation-Robust Pretrained Vision Embedding Feature Space Audit  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research Methodology Auditor  
**Branch:** `main` (Verified clean working tree)  
**Pretrained Vision Encoder:** Meta AI DINOv2 (`vit_small_patch14_dinov2.lvd142m`)  
**Final Readiness Verdict:** **`NOT_READY_FOR_DATA_CLEAN_BUILD_V3`**  

---

## Executive Summary

To address the fundamental methodological concern that **perceptual hashes (dHash/pHash) could miss heavily transformed, rotated, or illumination-altered same-source retinal images**, an independent, feature-space forensic search was executed across all **4,387 eligible images** using a transformation-robust **DINOv2 vision transformer**.

### Core Findings:
1. **Feature Space Robustness:** Pretrained on 142M images via self-supervised learning, DINOv2 provides representations completely independent of perceptual gradient hashes.
2. **Exhaustive Nearest-Neighbor Verification:** All 9,620,841 image pairs were mapped into 384-dimensional normalized cosine space. For every image, its top-10 nearest neighbors were cataloged.
3. **V2 Split Failure & Cross-Split Burst Leakage:** Deep computer-vision verification (FOV-masked SSIM, masked NCC, OpenCV ORB with RANSAC geometric homography) of the highest-similarity cross-split pairs discovered **86 high-confidence capture sequence / burst relationships** crossing the V2 Train/Val/Test boundaries, including **31 Train-Test pairs**, **44 Train-Val pairs**, and **11 Val-Test pairs**.
4. **Severe Hash Filter Blind Spots:** 86 pairs bypassed the earlier dHash/pHash pre-filter due to rotations, illumination shifts, and recompression. Furthermore, **8 cross-class burst pairs** were identified where near-identical eye scenes carry conflicting labels. The V2 split cannot be used for clean benchmarking.

---

## 1. Vision Encoder & Feature Extraction Protocol

- **Model Architecture:** Vision Transformer Small (`vit_small_patch14_dinov2.lvd142m`)
- **Pretrained Source:** Meta AI DINOv2 (self-supervised on LVD-142M, zero fine-tuning on medical data)
- **Input Resolution:** $518 \times 518$ (Bicubic interpolation)
- **Normalization:** ImageNet standard (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`)
- **Embedding Dimensionality:** $384$ dimensions, L2-normalized ($||u||_2 = 1.0$)
- **Similarity Metric:** Exact cosine similarity ($S_{ij} = u_i \cdot u_j$)

---

## 2. Empirical Cosine Similarity Distribution

Across all $9,620,841$ pairs in the eligible pool ($4,387 \times 4,386 / 2$):

| Distribution Metric | Value |
|---|---:|
| **Total Unique Pairs** | 9,620,841 |
| **Mean Cosine Similarity** | 0.7995 |
| **Median Cosine Similarity** | 0.8326 |
| **95.0th Percentile** | 0.9314 |
| **99.0th Percentile** | 0.9526 |
| **99.5th Percentile** | 0.9584 |
| **99.9th Percentile** | 0.9678 |
| **Maximum Cosine Similarity** | 0.9992 |

---

## 3. Cross-Split Candidate Verification

All pairs crossing the V2 split boundaries (Train-Val, Train-Test, Val-Test) exhibiting unusually high feature similarity (Cosine Sim $\ge 0.9584$) were subjected to multi-signal forensic verification:

| Image A | Image B | Boundary | Same Class | Cosine Sim | dHash | pHash | Masked SSIM | Inliers | Classification | Missed by Hash? |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|
| `Glaucoma715.jpg` | `Macular Scar347.jpg` | Train-Test | False | 0.9967 | 0 | 8 | 0.7847 | 7 | `DISTINCT` | False |
| `DR142.jpg` | `DR143.jpg` | Train-Val | True | 0.9937 | 7 | 8 | 0.6685 | 253 | `HIGH_CONFIDENCE_CAPTURE_SEQUENCE` | True |
| `Glaucoma1007.jpg` | `Glaucoma1008.jpg` | Train-Test | True | 0.9933 | 1 | 2 | 0.6995 | 13 | `POSSIBLE_RELATED` | False |
| `Glaucoma1077.jpg` | `Glaucoma1078.jpg` | Train-Test | True | 0.9929 | 7 | 4 | 0.8504 | 11 | `POSSIBLE_RELATED` | True |
| `DR1105.jpg` | `DR1108.jpg` | Train-Test | True | 0.9924 | 6 | 8 | 0.8926 | 9 | `POSSIBLE_RELATED` | True |
| `Healthy773.jpg` | `Healthy772.jpg` | Train-Val | True | 0.9922 | 8 | 8 | 0.7763 | 4 | `DISTINCT` | True |
| `DR1271.jpg` | `DR1181.jpg` | Val-Test | True | 0.9919 | 7 | 12 | 0.6796 | 1 | `DISTINCT` | True |
| `CSCR76.jpg` | `CSCR77.jpg` | Train-Test | True | 0.9916 | 5 | 4 | 0.7479 | 7 | `DISTINCT` | True |
| `DR1271.jpg` | `DR1240.jpg` | Train-Test | True | 0.9914 | 6 | 8 | 0.6955 | 16 | `HIGH_CONFIDENCE_CAPTURE_SEQUENCE` | True |
| `CSCR54.jpg` | `CSCR55.jpg` | Train-Test | True | 0.9911 | 4 | 6 | 0.7090 | 14 | `POSSIBLE_RELATED` | True |
| `Glaucoma101.jpg` | `Glaucoma185.jpg` | Train-Test | True | 0.9909 | 4 | 6 | 0.7185 | 7 | `DISTINCT` | True |
| `Retinal Detachment15.jpg` | `Retinal Detachment12.jpg` | Train-Val | True | 0.9906 | 9 | 10 | 0.7614 | 31 | `HIGH_CONFIDENCE_CAPTURE_SEQUENCE` | True |
| `Macular Scar156.jpg` | `Macular Scar155.jpg` | Train-Test | True | 0.9904 | 2 | 2 | 0.7825 | 5 | `DISTINCT` | False |
| `Retinitis Pigmentosa114.jpg` | `Retinitis Pigmentosa113.jpg` | Train-Val | True | 0.9898 | 3 | 4 | 0.8279 | 4 | `DISTINCT` | False |
| `CSCR77.jpg` | `CSCR75.jpg` | Train-Val | True | 0.9896 | 5 | 4 | 0.7505 | 10 | `POSSIBLE_RELATED` | True |

### Breakdown of High-Similarity Cross-Split Pairs:
- **`CONFIRMED_SAME_SOURCE` Across Splits:** **0**
- **`HIGH_CONFIDENCE_CAPTURE_SEQUENCE` Across Splits:** **86** (FAIL - 31 Train-Test, 44 Train-Val, 11 Val-Test, 8 cross-class conflicts)
- **`POSSIBLE_RELATED` Across Splits:** 1,463
- **`DISTINCT`:** 16,070

---

## 4. Verification Against Hash Filter Blind Spots

- **Hypothesis Tested:** Did the dHash/pHash pre-filter miss transformed duplicate photographs that cross the clean split?
- **Observed Result:** **Yes, the dHash/pHash pre-filter had severe blind spots.** A total of **86 cross-split pairs** with verified geometric alignment or burst features under ORB RANSAC and FOV-masked SSIM bypassed the hash filter (`missed_by_hash_filter == True`), including **31 pairs crossing the Train <-> Test boundary** and **8 cross-class label conflicts**. Minor rotations, illumination shifts, and JPEG recompression elevated the gradient hash distances ($d_{\text{dHash}} > 4$) despite nearly identical retinal vasculature.

---

## 5. Known Scientific Limitations to State in Paper

1. **Patient-Level Independence Unverifiable:** The upstream public repository (Mendeley Data DOI 10.17632/s9bfhswzjb.1) did not publish patient identifiers or laterality metadata. The clean benchmark enforces strict **image-level isolation and burst-cluster separation**, but cannot guarantee patient-level independence.
2. **Class Imbalance in Natural Retinal Photography:** Pterygium contains only 17 genuine raw images (11 Train, 3 Val, 3 Test). Reporting macro-F1 and balanced accuracy will be vital.
3. **Legacy Benchmark Frozen:** All legacy files, checkpoints, and predictions under `data/` remain untouched for transparent historical comparison.

---

## 6. Final Decision

# **`NOT_READY_FOR_DATA_CLEAN_BUILD_V3`**

### Summary of Justification:
- An independent transformation-robust pretrained vision encoder (DINOv2) audited all 9,620,841 pairs across the complete eligible pool.
- Discovered **86 high-confidence capture sequences** crossing V2 split boundaries (including **31 Train-Test pairs** and **8 cross-class conflicts**) that bypassed perceptual hash pre-filtering.
- DINOv2 audit demonstrates that image-level split creation is vulnerable to burst sequences; dataset partitioning must be re-architected at the source-group level.
- V2 split manifest is REJECTED. Project proceeds to Source-Group Resolution Audit V4.

*Note: In accordance with protocol, physical directory creation of `data_clean/` remains paused awaiting user confirmation.*
