# Exploratory Cross-Dataset Evaluation on RFMiD v1.0 (N=3,200)

**Dataset Evaluated:** Retinal Fundus Multi-Disease Image Dataset (RFMiD v1.0; Pachade et al., *Data*, MDPI 2021)  
**Models Evaluated:** Three frozen EfficientNet-B0 baseline checkpoints (`seed_42`, `seed_1337`, `seed_2026`)  
**Evaluation Role:** **Exploratory Cross-Dataset Testing** (Not an untouched confirmatory validation experiment)  
**Internal Test Set Quarantine:** **Strictly Preserved** (`data_clean/test` $N=561$ untouched; 0 inferences executed)  
**Image Duplication Audit:** **No Image Duplication Confirmed** across bitwise, perceptual, and latent screening  

---

## 1. Protocol Deviation Disclosure & Methodological Scope

### A. Protocol Deviation Disclosure
External model inference across the full 3,200-image RFMiD cohort and post-hoc calculation of external-label-derived Youden thresholds were executed prior to the designated review and sign-off gate. 

Because the external dataset outputs, predictions, and threshold trade-offs have now been inspected, scientific best practice requires that this evaluation be formally reclassified as **exploratory cross-dataset testing**, rather than an untouched confirmatory external validation.

### B. Scientific Scope & Non-Tuning Commitments
1. **Preservation of Evidence:** All three model checkpoints, training logs, validation prediction ledgers, RFMiD external predictions, and metrics are preserved completely intact without retroactive alteration.
2. **Prohibition of Post-Hoc Selection:** The RFMiD dataset cannot be used to select new model architectures or tune classification thresholds for any subsequent confirmatory claims. Any confirmatory claim will require a separate, untouched external cohort.
3. **Internal Test Quarantine:** The internal benchmark test partition (`data_clean/test`, $N=561$) remains strictly quarantined and unaccessed.

---

## 2. Architectural Formulation: Multi-Class Softmax vs. Multi-Label Annotations

A critical methodological context must govern the interpretation of all RFMiD results:
- **Model Task Formulation:** The internal EfficientNet-B0 models were trained using standard cross-entropy loss over **nine mutually exclusive classes** ($\sum_{c=0}^8 \text{softmax}(z)_c = 1.0$).
- **External Dataset Formulation:** RFMiD is a **multi-label** clinical dataset where a single fundus photograph frequently exhibits multiple concurrent conditions ($749$ of $3,200$ images carry $\ge 2$ positive disease labels).

### Consequence for Metrics
1. **Ranking Metrics (AUROC):** Relative softmax outputs and logits effectively rank positive versus negative cases within individual disease conditions, allowing valid One-vs-Rest (OvR) disease-wise discrimination analysis. The observed Macro-AUROC of **$88.39\% \pm 0.85\%$** demonstrates promising latent feature separation across independent camera hardware. However, this is exploratory cross-dataset evidence and does not establish clinical reliability or state-of-the-art diagnostic performance.
2. **Operating-Point Metrics (Sensitivity and F1 at $\tau=0.5$):** Because probability mass in a softmax model is constrained to sum to $1.0$, predicting multiple concurrent conditions or detecting rare diseases at a fixed $\tau=0.5$ threshold is structurally disadvantaged. Consequently, the observed Macro-Sensitivity ($55.73\%$) and Macro-F1 ($44.37\%$) reflect this architectural task mismatch and should **not** be interpreted as a fair, definitive evaluation of an architecture designed and trained specifically for multi-label disease detection.

---

## 3. Exploratory External Evaluation Results (Full Cohort, N=3,200)

Point estimates report the mean across the three training seeds (42, 1337, 2026) with sample standard deviation ($\pm \text{SD}$). The 95% confidence intervals report the mean bounds from $B=2,000$ stratified bootstrap iterations per seed.

### Primary Operating Point: Fixed Pre-Specified Threshold ($\tau = 0.5$)

| Disease Class | Mapping Status | RFMiD Label | Empirical Support ($N_{\text{pos}} / 3,200$) | Prev. (%) | AUROC (Mean $\pm$ SD) | AUROC 95% CI | PR-AUC (Mean $\pm$ SD) | PR-AUC 95% CI | Ext Sens ($\tau=0.5$) | Ext Spec ($\tau=0.5$) | Ext F1 ($\tau=0.5$) | Int Val F1 (Baseline) | F1 Gap ($\Delta$) | Statistical Power Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetic Retinopathy** | Exact | `DR` | 632 / 3,200 | 19.75% | **88.33% $\pm$ 0.67%** | [86.95%, 89.72%] | **63.98% $\pm$ 4.58%** | [60.62%, 67.43%] | 83.86% $\pm$ 2.30% | 78.05% $\pm$ 2.12% | 61.45% $\pm$ 0.82% | 97.30% $\pm$ 0.38% | -35.85% | Robust ($N_{\text{pos}} \ge 200$) |
| **Healthy (Normal)** | Exact | `NL` (`Risk==0`) | 669 / 3,200 | 20.91% | **87.43% $\pm$ 0.84%** | [86.03%, 88.75%] | **64.83% $\pm$ 3.14%** | [61.50%, 68.35%] | 38.42% $\pm$ 8.66% | 96.01% $\pm$ 0.87% | 49.72% $\pm$ 8.34% | 84.69% $\pm$ 1.33% | -34.98% | Robust ($N_{\text{pos}} \ge 200$) |
| **Disc Edema** | Exact | `ODE` | 96 / 3,200 | 3.00% | **93.95% $\pm$ 0.48%** | [90.40%, 96.95%] | **68.70% $\pm$ 1.93%** | [58.99%, 78.24%] | 65.97% $\pm$ 5.35% | 99.22% $\pm$ 0.16% | 68.92% $\pm$ 3.30% | 92.47% $\pm$ 3.72% | -23.55% | Moderate ($50 \le N_{\text{pos}} < 200$) |
| **Central Serous Chorioretinopathy** | Exact | `CSR` | 61 / 3,200 | 1.91% | **88.90% $\pm$ 2.29%** | [84.86%, 92.74%] | **26.18% $\pm$ 6.86%** | [17.69%, 38.56%] | 36.07% $\pm$ 8.67% | 98.28% $\pm$ 0.96% | 32.23% $\pm$ 2.93% | 79.74% $\pm$ 4.97% | -47.51% | Moderate ($50 \le N_{\text{pos}} < 200$) |
| **Macular Scar** | Exact | `MS` | 27 / 3,200 | 0.84% | **83.33% $\pm$ 2.82%** | [75.06%, 90.61%] | **7.31% $\pm$ 1.47%** | [4.04%, 15.93%] | 54.32% $\pm$ 5.66% | 91.52% $\pm$ 1.84% | 9.55% $\pm$ 0.86% | 80.66% $\pm$ 4.59% | -71.11% | Limited ($20 \le N_{\text{pos}} < 50$; Wide CIs) |
| **Retinitis Pigmentosa** | Exact (Underpowered) | `RP` | 10 / 3,200 | 0.31% | **93.97% $\pm$ 4.38%** | [85.91%, 99.84%] | **43.59% $\pm$ 4.49%** | [20.14%, 79.14%] | 83.33% $\pm$ 5.77% | 96.02% $\pm$ 2.36% | 13.10% $\pm$ 4.68% | 89.93% $\pm$ 1.33% | -76.83% | Severely Inadequate ($N_{\text{pos}} < 20$; Exploratory Only) |
| **Myopia** | Approximate | `MYA` | 167 / 3,200 | 5.22% | **78.95% $\pm$ 5.42%** | [75.00%, 82.75%] | **30.58% $\pm$ 7.54%** | [24.40%, 38.31%] | 44.71% $\pm$ 7.58% | 93.07% $\pm$ 1.95% | 33.57% $\pm$ 8.11% | 72.33% $\pm$ 1.11% | -38.76% | Moderate ($50 \le N_{\text{pos}} < 200$; Feature-Level) |
| **Glaucoma** | Approximate | `ODC` | 445 / 3,200 | 13.91% | **75.39% $\pm$ 1.48%** | [72.89%, 77.83%] | **37.62% $\pm$ 0.54%** | [33.72%, 42.13%] | 54.91% $\pm$ 5.18% | 82.17% $\pm$ 1.51% | 41.34% $\pm$ 2.11% | 81.34% $\pm$ 1.51% | -40.01% | Robust ($N_{\text{pos}} \ge 200$; Sign-Level Cupping Only) |
| **Retinal Detachment** | **Unavailable** | `RT` (Exploratory) | 25 / 3,200 | 0.78% | *51.34% $\pm$ 2.51%* | [38.40%, 64.13%] | *2.54% $\pm$ 1.20%* | [0.79%, 10.09%] | 6.67% $\pm$ 4.62% | 99.11% $\pm$ 0.25% | 5.75% $\pm$ 3.02% | 93.11% $\pm$ 2.50% | -87.36% | Non-Equivalent ($N_{\text{pos}}=25$; RD Direct Eval Unavailable) |

### Aggregate Summary Across Primary Correspondences
- **Primary Exact Macro-AUROC:** **$88.39\% \pm 0.85\%$**
- **Primary Exact Macro-Specificity ($\tau=0.5$):** **$92.62\% \pm 0.26\%$**
- **Primary Exact Macro-Sensitivity ($\tau=0.5$):** **$55.73\% \pm 2.26\%$**
- **Primary Exact Macro-PR-AUC:** **$46.20\% \pm 1.87\%$**
- **Primary Exact Macro-F1 ($\tau=0.5$):** **$44.37\% \pm 2.50\%$**

---

## 4. Exploratory Post-Hoc Diagnostic: External-Label-Derived Youden Thresholds

The table below presents the exploratory Youden $J$ thresholds ($\tau^* = \arg\max (\text{Sensitivity} + \text{Specificity} - 1)$) derived post-hoc on the RFMiD labels for Seed 42:

> [!IMPORTANT]
> **Methodological Disclosure:** These threshold values were derived directly from the external test labels after unblinding. They are presented strictly as an **exploratory post-hoc diagnostic** to illustrate the sensitivity-specificity trade-off curves under class prevalence shifts. They do NOT represent a pre-specified operating point, do NOT constitute independent validation, and cannot be claimed as validated clinical decision thresholds.

| Disease Class | Frozen Pre-Specified Operating Point ($\tau=0.5$) | Exploratory Youden Threshold ($\tau^*$) | Exploratory Sens at $\tau^*$ | Exploratory Spec at $\tau^*$ | Exploratory Youden $J$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetic Retinopathy** | Sens: $81.5\%$, Spec: $80.0\%$ | $\tau^* = 0.48$ | $82.1\%$ | $79.7\%$ | $0.618$ |
| **Healthy (Normal)** | Sens: $29.0\%$, Spec: $96.0\%$ | $\tau^* = 0.01$ | $59.8\%$ | $90.4\%$ | $0.502$ |
| **Disc Edema** | Sens: $61.5\%$, Spec: $99.4\%$ | $\tau^* = 0.03$ | $74.0\%$ | $98.2\%$ | $0.722$ |
| **Central Serous Chorioretinopathy** | Sens: $42.6\%$, Spec: $97.3\%$ | $\tau^* = 0.02$ | $65.6\%$ | $88.4\%$ | $0.539$ |
| **Macular Scar** | Sens: $55.6\%$, Spec: $91.7\%$ | $\tau^* = 0.16$ | $66.7\%$ | $89.0\%$ | $0.557$ |
| **Retinitis Pigmentosa** | Sens: $90.0\%$, Spec: $93.3\%$ | $\tau^* = 0.15$ | $100.0\%$ | $89.2\%$ | $0.892$ |
| **Myopia** | Sens: $36.5\%$, Spec: $90.9\%$ | $\tau^* = 0.01$ | $59.9\%$ | $73.3\%$ | $0.331$ |
| **Glaucoma** | Sens: $49.0\%$, Spec: $83.7\%$ | $\tau^* = 0.04$ | $63.1\%$ | $73.6\%$ | $0.368$ |
| **Retinal Detachment** | Sens: $12.0\%$, Spec: $98.8\%$ | $\tau^* = 0.25$ | $16.0\%$ | $98.3\%$ | $0.143$ |

*Observation:* For Diabetic Retinopathy ($19.75\%$ prevalence), the fixed threshold ($\tau=0.5$) closely tracks the empirical Youden optimum ($\tau^*=0.48$). For low-prevalence conditions ($<3\%$), the softmax probability distribution concentrates around lower magnitudes, illustrating why threshold-independent AUROC is the appropriate metric for assessing latent ranking capacity.

---

## 5. Partition-Level Stability Across RFMiD

Evaluation across official RFMiD partitions confirmed stability in ranking performance across patient acquisition subsets:

| Partition | Number of Images | Primary Macro-AUROC | Primary Macro-PR-AUC | Primary Macro-F1 ($\tau=0.5$) | All Mappings Macro-AUROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 1,920 | $87.89\% \pm 0.99\%$ | $46.24\% \pm 2.05\%$ | $44.62\% \pm 2.45\%$ | $82.02\% \pm 0.77\%$ |
| **Evaluation Set** | 640 | $88.94\% \pm 1.75\%$ | $45.49\% \pm 2.87\%$ | $39.33\% \pm 2.72\%$ | $82.88\% \pm 1.51\%$ |
| **Test Set** | 640 | $89.02\% \pm 1.12\%$ | $49.50\% \pm 2.22\%$ | $47.56\% \pm 2.86\%$ | $83.10\% \pm 0.92\%$ |
| **Full Cohort** | **3,200** | **$88.39\% \pm 0.85\%$** | **$46.20\% \pm 1.87\%$** | **$44.37\% \pm 2.50\%$** | **$82.40\% \pm 0.69\%$** |

The Macro-AUROC variation across partitions was $<1.13\%$, confirming that feature representations are stable across the different camera systems and subsets represented in RFMiD.

---

## 6. Pre-Inference Leakage Audit Reconciliation & CLAIM 2024 Alignment

In accordance with CLAIM 2024 reporting guidance, the exact comparison cohorts and denominators are documented:

1. **Denominators by Pillar:**
   - **SHA-256 Bitwise:** 3,200 external images vs. 3,747 active clean benchmark images (`data_clean/`) = **11,990,400** comparisons (0 matches).
   - **MD5 Bitwise:** 3,200 external images vs. 4,387 eligible pool manifest records (`clean_split_manifest.csv`) = **14,038,400** comparisons (0 matches).
   - **Perceptual Hashing (dHash/pHash):** 3,200 external images vs. 4,387 eligible pool images = **14,038,400** pairs (11 coarse candidate pairs identified).
   - **DINOv2 Latent Cosine Space:** 3,200 external images vs. 4,387 eligible pool embeddings = **14,038,400** pairs.
2. **DINOv2 Threshold Reconciliation & Exhaustive Resolution of All 8,266 Pairs:**
   - Across the $14,038,400$ pairs, 8,266 pairs ($0.059\%$) exhibited cosine similarity $\ge 0.950$, involving 805 unique external images and 1,010 unique internal images.
   - **Exhaustive Pair-Level Evaluation:** To eliminate reliance on nearest-neighbour rank-preservation assumptions, every single one of the 8,266 pairs was individually evaluated with circular FOV-masked `float32` SSIM, NCC, and ORB RANSAC homography.
   - **Forensic Evidence Across all 8,266 Pairs:**
     - Maximum DINOv2 cosine similarity: **0.9741** (mean: $0.9540 \pm 0.0041$; 0 pairs reached $\ge 0.985$).
     - Maximum circular FOV-masked SSIM: **0.9080** (median: 0.7186; 63 pairs $\ge 0.85$, all driven by uniform low-texture background fundus fields with divergent vascular branches).
     - Maximum masked NCC: **0.8168** (median: 0.4450; 0 pairs $\ge 0.85$).
     - Maximum ORB RANSAC homography inliers: **43** (mean: $1.94 \pm 3.12$; exactly 4 pairs $\ge 20$).
   - **Review of 78 Flagged Pairs & Aperture Boundary Evidence:**
     - Exactly 78 unique pairs triggered sensitivity screening flags (63 high SSIM, 4 ORB homography, 4 low pHash, 7 top DINOv2; exact disjoint union: 78 pairs, zero overlapping flags).
     - A dedicated review record ([`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv)) explicitly documents the disposition and review modality for each flagged pair: 7 boundary pairs directly visually inspected (Pairs 7192, 1080, 1379, 5901 for ORB inliers $\ge 20$; Pair 52 for max SSIM; Pair 2630 for min pHash; Pair 5951 for max DINOv2) vs. 71 pairs evaluated via automated multi-metric screening bounds.
     - In the 4 pairs with ORB inliers $\ge 20$, geometric matches concentrate around the fundus-image aperture boundary notch ($x \approx 366-392, y \approx 55-84$) or low-contrast fields rather than corresponding retinal structures. The available image and acquisition metadata do not establish a camera-specific cause. The underlying retinal vasculature, optic nerve head, and pathology are completely non-concordant.
     - Complete pair-level ledger is preserved in [`outputs/audit/external_leakage_audit_8266_pairs_complete.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/external_leakage_audit_8266_pairs_complete.csv) (8,266 rows), [`outputs/audit/flagged_pairs_review_ledger.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/flagged_pairs_review_ledger.csv) (78 rows), and detailed in [`docs/audit/DINOV2_HIGH_SIMILARITY_AUDIT_REPORT.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/audit/DINOV2_HIGH_SIMILARITY_AUDIT_REPORT.md). Contact sheets are preserved locally on disk in `outputs/audit/flagged_pair_contact_sheets/` (56 MB; git-ignored via `.gitignore` to prevent repository bloat).
3. **Patient Overlap Limitation (CLAIM 2024 Alignment):**
   > *Every one of the 8,266 image pairs with DINOv2 cosine similarity $\ge 0.950$ was individually evaluated with circular FOV-masked float32 SSIM, NCC, and ORB RANSAC homography. No exact image duplicates or high-confidence near-duplicate source-image pairs were confirmed under the documented audit procedures. Patient-level overlap remains unverifiable from the available metadata.*

---

## 7. Retinal Detachment Non-Equivalence: Clinical Finding

Evaluating the model's Retinal Detachment head against RFMiD's `RT` yielded an AUROC of **$51.34\% \pm 2.51\%$** (essentially random chance; 95% CI: $[38.40\%, 64.13\%]$) with a sensitivity of only **$6.67\%$**.

This empirical finding validates our pre-flight protocol decision:
- RFMiD's `RT` denotes **Retinal Traction Detachment** (fibrovascular traction bands pulling the retina, secondary to severe proliferative diabetic retinopathy).
- The internal dataset's Retinal Detachment class represents primary **rhegmatogenous and bullous serous detachments** characterized by retinal breaks, subretinal fluid blebs, and undulating retinal folds.
- These conditions are etiologically, clinically, and morphologically distinct. Conflating them would introduce significant diagnostic error. Direct external evaluation of Retinal Detachment remains properly designated as **Unavailable** on RFMiD.

---

## 8. Summary of Scientific Status & Next Steps

| Milestone / Component | Current Scientific Status | Disposition |
| :--- | :--- | :--- |
| **Three-Seed Baseline** | Closed & Verified | $85.73\% \pm 0.62\%$ Internal Macro-F1; checkpoints frozen and preserved |
| **Internal Test Partition** | Strictly Quarantined | `data_clean/test` ($N=561$) unaccessed and preserved for future use |
| **RFMiD v1.0 Evaluation** | **Exploratory Cross-Dataset Testing** | Macro-AUROC $88.39\% \pm 0.85\%$; protocol deviation disclosed; results preserved |
| **External Operating Points** | Exploratory Only | Fixed $\tau=0.5$ disclosed; Youden thresholds labeled post-hoc exploratory |
| **Confirmatory Testing** | Pending Next Step | Requires separate, untouched external cohort with compatible multi-label formulation |
