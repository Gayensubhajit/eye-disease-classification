# External Validation Preflight Protocol & Label Support Specification (v1.1)

**Date:** 2026-10-11  
**Project:** 9-Class Posterior-Segment Fundus Classification Benchmark  
**Document Status:** Pre-Registered Governance Contract (Pre-Inference)  
**Parent Protocols:** [Research Protocol v1.1](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/RESEARCH_PROTOCOL_V1_1.md), [Baseline Provenance Manifest](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_9class_baseline_effnet/PROVENANCE.json)

---

## 1. Internal Baseline Provenance & Freezing Summary

The internal 9-class baseline evaluation on the development split (2,612 train, 560 validation, 561 quarantined test images) is officially **frozen**. No additional training runs, hyperparameter sweeps, architecture modifications, or internal test split evaluations will be conducted.

### 1.1 Code Provenance Across Training Runs
- **Git Base Commit Reference:** `cc2fc8b8f557f1b8e81922fa6a129afdcd107b4a` (branch: `main`)
- **Core Training Code Base:** The core library files in `src/` (`src/train.py`, `src/data/dataset.py`, `src/data/preprocessing.py`, `src/models/backbone.py`, `src/losses/focal_loss.py`, `src/metrics.py`) were committed at `cc2fc8b8f557f1b8e81922fa6a129afdcd107b4a` and remained **100% untouched and identical** across all three training runs.
- **Runner Script History (`scripts/run_9class_3seed_baseline.py`):**
  - **Seed 42 & Seed 1337 Execution:** Executed sequentially via `src/train.py` using the initial runner implementation.
  - **Seed 2026 Resumption:** Following an environment restart, the runner was updated with wrapper-level resilience improvements:
    1. Resumption logic checking for existing `val_metrics.json` and `best_model.pth` to bypass re-training already-completed seeds.
    2. Path normalization (`.resolve()`) to ensure reliable relative path resolution across filesystem mounts.
    3. Setting `NO_ALBUMENTATIONS_UPDATE=1` in the subprocess environment to eliminate network timeout delays on worker startup.
    4. Populating `per_class` as a named dictionary rather than a list of records in `val_metrics.json`, writing `best_model.pth.sha256`, and asserting sample-level label indexing (`assert int(y_true[idx]) == int(true_sample_label)`).
  - **Scientific Consequence:** No training hyperparameters, architectures, loss functions, optimizers, learning rate schedules, seeds, or data splits were altered. The underlying model training execution in `src/train.py` was identical across all three seeds.
- **Current Runner Script SHA-256:** `9439fa379df8b750df6883227f0f475d2753e5bb5cb7a55047a58f9edf967906`

### 1.2 Checkpoint Cryptographic Ledger & File Inventory
Every seed contains its complete artifact suite on disk:

| Seed | Checkpoint Path | Binary SHA-256 Digest | Val Macro-F1 | Val Balanced Acc | Val Overall Acc | Correct / Total |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **42** | [`outputs/clean_9class_baseline_effnet/seed_42/best_model.pth`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_9class_baseline_effnet/seed_42/best_model.pth) | `e54f0f86fafaa95f7921a83b383f2b9928111383a68761e7f915b8b63abda38d` | 85.14% | 85.42% | 87.32% | 489 / 560 |
| **1337** | [`outputs/clean_9class_baseline_effnet/seed_1337/best_model.pth`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_9class_baseline_effnet/seed_1337/best_model.pth) | `887f3a16aeb029054e9a31f75837c409fd73b76c1b8ad2677d745af9928cbe41` | 85.69% | 87.46% | 87.50% | 490 / 560 |
| **2026** | [`outputs/clean_9class_baseline_effnet/seed_2026/best_model.pth`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_9class_baseline_effnet/seed_2026/best_model.pth) | `8d89c9e0d4ef73b3ae6a76d83c87c1eac28f5e891f11740bcc9515a1be58caa9` | 86.37% | 86.93% | 88.04% | 493 / 560 |

### 1.3 Expanded Mathematical & Confusion Matrix Verification
An automated audit script ([`scripts/audit_baseline_provenance_complete.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/audit_baseline_provenance_complete.py)) verified:
1. **Element-for-Element Confusion Matrix Matching:** Fresh $9 \times 9$ confusion matrices reconstructed from `val_predictions.csv` for each seed were asserted against the saved `val_confusion_matrix.csv` files on disk using `numpy.testing.assert_array_equal()`. Every cell was **identical**.
2. **Per-Class Metrics Recalculation:** All sensitivities, specificities, precisions, and F1 scores were recalculated directly from the confusion matrices and verified against `val_metrics.json`.
3. **Aggregate Summary Assertion:** All aggregate statistics in [`outputs/clean_9class_baseline_effnet/aggregate_results.json`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_9class_baseline_effnet/aggregate_results.json) were asserted against numpy calculations across the three seeds:
   $$\text{Macro-F1} = \mathbf{85.73\% \pm 0.62\%} \quad (\text{Range: } 85.14\% - 86.37\%)$$
   $$\text{Balanced Accuracy} = \mathbf{86.60\% \pm 1.06\%} \quad (\text{Range: } 85.42\% - 87.46\%)$$
   $$\text{Overall Accuracy} = \mathbf{87.62\% \pm 0.37\%} \quad (\text{Range: } 87.32\% - 88.04\%)$$
   $$\text{Peak Val Loss} = \mathbf{0.9771 \pm 0.1179}$$
4. **Internal Test Set Quarantine:** Confirmed strictly enforced (`evaluate_test: false`, 0 test inferences executed).

---

## 2. RFMiD Dataset Provenance & Release Integrity

### 2.1 Release Specification
* **Target External Dataset:** Retinal Fundus Multi-Disease Image Dataset (**RFMiD v1.0**).
* **Primary Literature Reference:** Pachade, S., Porwal, P., Thulkar, D., Kokare, M., Deshmukh, G., Sahasrabuddhe, V., Giancardo, L., Quellec, G., Mériaudeau, F. *Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research.* Data 2021, 6(2), 14. doi:[10.3390/data6020014](https://doi.org/10.3390/data6020014).
* **Licensing / Access Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Clinical Acquisition Hardware:** 3 clinical centers in India using 3 fundus cameras:
  - Topcon 3D OCT-2000 ($4288 \times 2848$)
  - Carl Zeiss VISUCAM 500 ($2048 \times 1536$)
  - Kowa VX-10$\alpha$ ($2144 \times 1424$)
* **Cohort Composition:** 3,200 total fundus images divided into Training ($N=1,920$, 60%), Validation ($N=640$, 20%), and Test ($N=640$, 20%).

### 2.2 Published Header Inventory & Column Verification
The published RFMiD v1 label inventory defines 46 columns:
`ID`, `Disease_Risk`, `DR`, `ARMD`, `MH`, `DN`, `MYA`, `BRVO`, `TSLN`, `ERM`, `LS`, `MS`, `CSR`, `ODC`, `CRVO`, `TV`, `AH`, `ODP`, `ODE`, `ST`, `AION`, `PT`, `RT`, `RS`, `CRS`, `EDN`, `RRN`, `RP`, `CD`, `HOD`, `WNL`, `OTHER` (plus remaining rare flags).

**Crucial Label Confirmations:**
1. **`MS` = Macular Scars:** In RFMiD v1, `MS` explicitly designates *Macular Scars* (fibrotic chorioretinal scarring in the macular area). Published total count is **27**.
2. **`RT` = Retinal Traction Detachment:** In RFMiD v1, `RT` designates *Retinal Traction Detachment*. Published total count is **25**. Standalone `RD` does NOT exist in the published v1 release headers.
3. **RFMiD v2 Discrepancies:** RFMiD v2 has an altered label inventory. All mappings defined herein apply strictly to the verified v1 column dictionary.

---

## 3. Updated Mapping, Support & Statistical Power Ledger

The table below reports the **exact published totals from Table 3 of Pachade et al. (2021)** across the full 3,200 cohort.

| Internal 9-Class Label | RFMiD v1 Column | Mapping Status | Published Table 3 Total ($N=3,200$) | Negative Support ($N=3,200$) | Statistical Power Status | Pre-Specified Metric Eligibility |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Diabetic Retinopathy** | `DR` | **Exact** | **632** ($19.8\%$) | 2,568 | **Robust (High)** | One-vs-Rest AUROC, PR-AUC, Sensitivity, Specificity (95% Bootstrap CI) |
| **Healthy** | `Disease_Risk == 0` / `NL` | **Exact** | **669** ($20.9\%$) | 2,531 | **Robust (High)** | Specificity, Normal-vs-Abnormal Discrimination, NPV (95% Bootstrap CI) |
| **Macular Scar** | `MS` | **Exact** | **27** ($0.8\%$) | 3,173 | **Limited (Small N)** | One-vs-Rest AUROC, PR-AUC, Sensitivity, Specificity (Flagged: wide bootstrap CIs) |
| **Central Serous Chorioretinopathy** | `CSR` | **Exact** | **61** ($1.9\%$) | 3,139 | **Moderate** | One-vs-Rest AUROC, Specificity, Sensitivity, PR-AUC (95% Bootstrap CI) |
| **Disc Edema** | `ODE` | **Exact** | **96** ($3.0\%$) | 3,104 | **Moderate** | One-vs-Rest AUROC, Specificity, Sensitivity (95% Bootstrap CI) |
| **Retinitis Pigmentosa** | `RP` | **Exact** | **10** ($0.3\%$) | 3,190 | **Severely Inadequate** ($N=10$) | **Exploratory Descriptive Only; Ineligible for definitive validation claims** |
| **Myopia** | `MYA` | **Approximate** | **167** ($5.2\%$) | 3,033 | **Adequate (Feature-level)** | One-vs-Rest AUROC (Conditional on Myopic Atrophy `MYA`), Specificity |
| **Glaucoma** | `ODC` | **Approximate** | **445** ($13.9\%$) | 2,755 | **Robust (Sign-level only)** | Ranking AUROC vs Optic Disc Cupping (`ODC`); **Ineligible for Clinical Glaucoma Diagnosis** |
| **Retinal Detachment** | None (`RT` exploratory only) | **Unavailable** | **25** ($0.8\%$) | 3,175 | **Non-Equivalent & Underpowered** | **Direct External Evaluation UNAVAILABLE; Exploratory RT AUROC Only** |

### 3.1 Crucial Distinction: Full-Dataset Totals vs. Actual Labelled Evaluation Cohort Support
1. **Challenge Benchmark Label Withholding:** In official public challenge distributions of RFMiD v1, the official test partition ($N=640$) was used for leaderboard benchmarking with ground truth labels withheld.
2. **Evaluation Cohort Verification Mandate:**
   - If the external release acquired contains ground truth for all 3,200 images, the evaluation cohort will comprise all 3,200 images.
   - If test partition labels are withheld in the acquired distribution, the labelled evaluation cohort will consist strictly of the **training + validation partitions ($N = 2,560$)**, and the test partition will NOT be treated as labelled evaluation data.
   - Under the $N = 2,560$ partition, available positive supports will be approximately $60\% - 80\%$ of Table 3 totals (e.g., $MS \approx 16-22$, $RP \approx 6-8$).
3. **Pre-Evaluation Check:** Prior to computing metrics or statistical power, the exact CSV headers, rows, and label availability must be verified on disk for the acquired release.

### 3.2 Rationale for Ineligibility & Power Warnings
1. **Retinal Detachment Direct Evaluation is UNAVAILABLE:**
   RFMiD v1 annotates `RT` (*Retinal Traction Detachment*; 25 total cases), which denotes tractional pulling forces from fibrovascular membranes (typically secondary to proliferative vitreoretinopathy or advanced PDR). Our internal Retinal Detachment class represents general/rhegmatogenous detachment (full neurosensory detachment from subretinal fluid). Direct external evaluation of Retinal Detachment is marked **Unavailable**. An exploratory analysis on `RT` may be reported as a lesion correlation, but must not be termed definitive external diagnostic accuracy.
2. **Severely Underpowered Classes (`RP`, `MS`):**
   With $N = 10$ positive cases for Retinitis Pigmentosa across all 3,200 images, a single classification error produces a $10.0\%$ swing in sensitivity. Therefore, `RP` is disqualified from confirmatory statistical validation claims and restricted to exploratory descriptive analysis. For Macular Scar (`MS`, $N = 27$), point estimates carry wide bootstrap confidence intervals that must be reported with explicit uncertainty bounds.

---

## 4. Multi-Label Semantics & Evaluation Mathematics

### 4.1 Strict Multi-Label Preservation
RFMiD fundus images frequently carry multiple co-occurring annotations (e.g., a patient presenting with both `DR` and `ODC`, or `DR` with a macular scar `MS`).
* **Rule:** RFMiD images will **NEVER** be coerced into an artificial mutually exclusive 9-class softmax prediction.
* **Methodology:** Every eligible class is evaluated as an **independent binary One-vs-Rest detection task**:
  $$\hat{s}_c(x) = P(Y = c \mid x)$$
  where $\hat{s}_c(x)$ is the model's softmax probability for class $c$.

### 4.2 Pre-Specified Metrics & Uncertainty Intervals
For each eligible disease class $c$:
1. **Threshold-Independent Discrimination:**
   - **Binary Area Under the ROC Curve (AUROC):** Assesses separation between positive cases ($y_c = 1$) and all negative cases ($y_c = 0$).
   - **Average Precision / PR-AUC:** Primary metric for highly imbalanced conditions (`MS`, `CSR`, `ODE`) where false positives heavily impact clinical utility.
2. **Operating Point Sensitivity & Specificity:**
   - Evaluated at the **pre-fixed internal validation threshold** $\tau_c = 0.5$ (or internal validation Youden-index optimal threshold).
   - **Zero Target-Domain Tuning:** Thresholds will NOT be adjusted, tuned, or re-optimized on the external RFMiD cohort.
3. **Uncertainty Intervals:**
   - All metrics will be reported with **95% non-parametric bootstrap confidence intervals** computed using 1,000 stratified resamples.

---

## 5. External Image Audit & Leakage Detection Strategy

Before calculating metrics on any newly ingested external dataset, a strict three-pillar leakage audit will be conducted:

1. **Pillar 1: Bit-Level Exact Duplication Scan:**
   - Compute SHA-256 digests for all external images.
   - Cross-reference against the internal dataset manifest (`data_clean/` train, val, and quarantined test).
2. **Pillar 2: Perceptual Hash (pHash) Near-Duplicate Scan:**
   - Compute 64-bit DCT perceptual hashes.
   - Screen for exact near-duplicates using Hamming distance $\le 3$.
3. **Pillar 3: Foundation Model Latent Embedding Similarity:**
   - Extract self-supervised DINOv2 embeddings ($768$-dim).
   - Flag any external sample with cosine similarity $> 0.95$ against any internal sample for visual review.

### 5.1 Documented Limitations of Automated Overlap Checks
- **Cross-Center Acquisition Artifacts:** Automated hash matching and embedding similarity can reliably detect identical images or re-compressed/re-scaled duplicates. However, they cannot definitively prove whether an eye photographed in an Indian hospital for RFMiD belongs to an individual who previously contributed to another public repository if the camera optics, cropping, and field of view differ significantly.
- **Independence Assumption:** True independence rests primarily on documented institutional provenance (multicenter Indian clinical acquisition vs. development benchmark sources).

---

## 6. Preflight Governance & Stop Protocol

- [x] **Internal Baseline Provenance:** Complete, verified, and frozen.
- [x] **Confusion Matrix & Aggregate Arithmetic:** Verified element-for-element ([`scripts/audit_baseline_provenance_complete.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/audit_baseline_provenance_complete.py)).
- [x] **RFMiD v1 Table 3 Counts:** Updated and reconciled with published Table 3 ($N=3,200$).
- [x] **Evaluation Partition Caution:** Documented that labelled support depends on test label release status.
- [x] **Retinal Detachment Direct Evaluation:** Formally marked **Unavailable**.
- [x] **Evaluation Mathematics:** Multi-label One-vs-Rest protocol pre-specified.
- [x] **Zero Contamination Guarantee:** No external inference executed; internal test split quarantined.

**Execution Status:** Ready for external dataset acquisition and label audit. No external inference will proceed prior to sign-off.
