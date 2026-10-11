# Research Protocol v1.1: Leakage-Aware, Externally Evaluated Retinal Disease Classification

**Project:** Classification of Eye Diseases from Color Fundus Photographs  
**Principal Investigator / Author:** Subhajit Gayen  
**Protocol Version:** 1.1 (Revised Scientific Protocol)  
**Date:** October 2026  
**Status:** FROZEN PENDING REVIEW — NO ACTIVE MODEL TRAINING  

---

## 1. Executive Summary & Research Motivation

The initial stage of this project evaluated deep learning models on the *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`, Rashid et al., *Data in Brief* 2024). Through extensive forensic audits, we eliminated synthetic near-duplicate leakage, quarantined cross-split burst frames, and established a cleaned benchmark (`data_clean/`, 3,747 images).

Subsequent methodological review identified critical issues in the task definition and scientific reporting:
1. **Imaging Modality Mismatch:** The source dataset publication combines **nine categories of posterior-segment retinal fundus photographs** with **one category of anterior-segment photography (Pterygium)**. Pterygium is a lesion of the bulbar conjunctiva and cornea on the ocular surface. Distinguishing Pterygium from retinal fundus images is not a retinal diagnostic task; it represents an anatomical and optical shortcut (anterior ocular surface vs. posterior retinal fundus).
2. **Severe Class Scarcity for Pterygium:** The cleaned benchmark contains only 14 unique Pterygium images across all splits (10 train, 2 val, 2 test). Reporting metrics such as 100% recall on 2 validation images provides an illusory signal of capability.
3. **Absence of Patient Identifiers & Laterality:** Patient identifiers, visit timestamps, and eye laterality tags (OD/OS) are unavailable in the distributed dataset files. As per CLAIM 2024 guidelines, the benchmark must be strictly reported as **image-level / source-group-isolated**, not patient-independent.
4. **Internal Test Exposure:** The internal test set (`data_clean/test`) was evaluated during the initial 10-class baseline experiments. While it remains locked against future model selection, it cannot be described as a pristine, never-before-accessed confirmatory cohort. Confirmatory generalization claims must rely on independent external evaluation.

This protocol redefines the primary task as a **homogeneous 9-class posterior-segment retinal fundus classification problem** and establishes an auditable research plan evaluating **shortcut mitigation, cross-dataset external generalization, probability calibration, and uncertainty-aware selective prediction**.

---

## 2. Research Questions and Hypotheses

* **RQ1 (Benchmark Validity & Shortcut Mitigation):**  
  Does eliminating the anterior-segment modality mismatch and near-duplicate leakage reveal the true baseline performance of deep neural networks on posterior-segment retinal disease classification?  
  *Hypothesis 1.1:* When evaluated on a pure 9-class fundus task, baseline classification metrics will reflect genuine retinal diagnostic difficulty, eliminating artificial 100% metrics driven by imaging modality shortcuts.

* **RQ2 (Cross-Dataset External Generalization):**  
  How effectively do retinal classifiers trained on the development fundus benchmark generalize to independent external datasets (e.g., RFMiD) acquired under different clinical protocols and populations?  
  *Hypothesis 2.1:* Models achieving >85% internal validation Macro-F1 will experience measurable performance degradation on external datasets due to domain shift, camera optical differences, and clinical comorbidity.

* **RQ3 (Uncertainty-Aware Prediction & Selective Abstention):**  
  Can calibrated predictive uncertainty reliably identify ambiguous, out-of-distribution, or erroneous classifications, enabling safe automated referral/deferral?  
  *Hypothesis 3.1:* Selective prediction with confidence-based abstention will significantly reduce high-risk diagnostic errors across a coverage-risk trade-off curve.

---

## 3. Final Task Definition & Class Dictionary

### Primary Task: 9-Class Posterior-Segment Retinal Disease Classification
The primary classification task operates strictly on color fundus photographs depicting the posterior segment (retina, optic disc, macula, and vascular arcade).

The training reference labels preserve the exact original dataset class names without speculative clinical reinterpretation:

| Class Index | Training Reference Label | Anatomical Target | Scope & Clinical Context |
|:---:|---|---|---|
| 0 | `Central Serous Chorioretinopathy [Color Fundus]` | Macula / RPE | Serous detachment of the neurosensory retina at the macula |
| 1 | `Diabetic Retinopathy` | Retinal Vasculature | Microvascular retinal complications of diabetes mellitus |
| 2 | `Disc Edema` | Optic Nerve Head | Swelling / edema of the optic disc margin |
| 3 | `Glaucoma` | Optic Nerve Head | Optic neuropathy characterized by structural disc changes |
| 4 | `Healthy` | Entire Posterior Pole | Normal fundus appearance without detected pathology |
| 5 | `Macular Scar` | Central Macula | Chorioretinal scarring in the macular region |
| 6 | `Myopia` | Posterior Pole | High / pathological myopic fundus changes |
| 7 | `Retinal Detachment` | Peripheral/Posterior Retina | Separation of neurosensory retina from the underlying RPE |
| 8 | `Retinitis Pigmentosa` | Mid-Peripheral Retina | Hereditary dystrophy presenting with pigmentary changes |

*Note on Clinical Codes:* Authoritative ICD-10 codes and diagnostic interpretations are intentionally omitted until formal review by a licensed ophthalmologist is obtained.

### Modality Separation: Pterygium
* **Modality:** Anterior-segment ocular surface photography (conjunctiva / cornea).
* **Protocol Action:** Permanently excluded from the primary retinal fundus classification task.
* **Accounting:** All 14 Pterygium images (10 train, 2 val, 2 test) are archived and documented in [`outputs/audit/excluded_pterygium_anterior_images.csv`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/audit/excluded_pterygium_anterior_images.csv). They remain in `data_clean/` to preserve benchmark immutability, but are masked out from all 9-class fundus dataloaders and evaluation matrices.

---

## 4. Dataset Inventory & Characterization

### Source Data Origin & Equipment
* **Source Dataset:** *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`, Rashid et al., *Data in Brief* 2024).
* **Clinical Sites:** Collected from **two hospitals** in Bangladesh: Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital.
* **Acquisition Equipment:** Acquired using **two Topcon camera systems** (Topcon TRC-50DX and Topcon NW400).
* **Metadata Constraints:** Per-image hospital and device assignments are not identified in the distributed files. Patient identifiers, visit timestamps, and eye laterality tags (OD/OS) are unavailable in the public release. Consequently, patient-level split independence cannot be asserted; partitions are strictly **image-level / source-group-isolated**.

### Cohort Accounting (`data_clean/`)

| Partition | Frozen 10-Class Benchmark | Excluded Pterygium | **9-Class Fundus Cohort** | Role in Research Protocol |
|---|:---:|:---:|:---:|---|
| `train` | 2,622 | 10 | **2,612** | Model parameter optimization |
| `val` | 562 | 2 | **560** | Checkpoint selection & temperature scaling |
| `test` | 563 | 2 | **561** | Locked internal evaluation (prior exposure noted) |
| **Total** | **3,747** | **14** | **3,733** | Pure Posterior-Segment Cohort |

### Per-Class Support in 9-Class Cohort

| Disease Class | Train | Val | Test | Total | Cohort % |
|---|:---:|:---:|:---:|:---:|:---:|
| Central Serous Chorioretinopathy | 49 | 11 | 11 | 71 | 1.90% |
| Diabetic Retinopathy | 891 | 191 | 191 | 1,273 | 34.10% |
| Disc Edema | 68 | 15 | 15 | 98 | 2.63% |
| Glaucoma | 579 | 124 | 124 | 827 | 22.15% |
| Healthy | 492 | 105 | 105 | 702 | 18.80% |
| Macular Scar | 217 | 47 | 47 | 311 | 8.33% |
| Myopia | 172 | 37 | 37 | 246 | 6.59% |
| Retinal Detachment | 63 | 13 | 14 | 90 | 2.41% |
| Retinitis Pigmentosa | 81 | 17 | 17 | 115 | 3.08% |
| **Total** | **2,612** | **560** | **561** | **3,733** | **100.0%** |

---

## 5. External Dataset Feasibility & Mapping Audit

Independent external evaluation is required to evaluate **RQ2** without threshold manipulation or fine-tuning on the target domain.

### 5.1 RFMiD Per-Class Mapping Specification & External Protocol
*Dataset Reference:* Retinal Fundus Multi-Disease Image Dataset (Pachade et al., *Scientific Data* 2021 & *Data*, MDPI 2021; 3,200 images across 3 clinical sites in India, CC BY 4.0).

| Source Class Label | RFMiD Label | Mapping Status | Published Table 3 Total ($N=3,200$) | Negative Support ($N=3,200$) | Statistical Power Status | Supporting Literature Source | Key Clinical Distinction & Caveats | Pre-Specified Metric Eligibility |
|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| **Diabetic Retinopathy** | `DR` | **Exact** | **632** | 2,568 | **Robust** | Pachade et al. (Data, MDPI 2021), Table 3 | Direct correspondence to non-proliferative/proliferative diabetic retinal vascular changes. | One-vs-Rest Binary AUROC, Specificity, Sensitivity, PR-AUC (95% Bootstrap CI) |
| **Healthy** | `Disease_Risk == 0` / `NL` | **Exact** | **669** | 2,531 | **Robust** | Pachade et al. (Data, MDPI 2021), Table 3 & Section 2.2 | Published Table 3 lists 669 Normal (`NL`) fundi without detectable retinal disease risk. | Specificity, Normal-vs-Abnormal Discrimination, NPV (95% Bootstrap CI) |
| **Macular Scar** | `MS` | **Exact** | **27** | 3,173 | **Limited** ($N=27$) | Pachade et al. (Data, MDPI 2021), Table 3 | In RFMiD v1, `MS` explicitly denotes Macular Scars (fibrotic chorioretinal scarring in macular zone). Published total is 27 across full cohort. Direct correspondence to internal Macular Scar class. Must verify partition support. | One-vs-Rest Binary AUROC, Specificity, Sensitivity, PR-AUC (Flagged: wide bootstrap CIs) |
| **Central Serous Chorioretinopathy** | `CSR` | **Exact** | **61** | 3,139 | **Moderate** | Pachade et al. (Data, MDPI 2021), Table 3 | Central Serous Retinopathy / Chorioretinopathy. Direct correspondence to localized neurosensory macular detachment. Published Table 3 lists 61 total positive cases. | One-vs-Rest Binary AUROC, Specificity, Sensitivity, PR-AUC (95% Bootstrap CI) |
| **Disc Edema** | `ODE` | **Exact** | **96** | 3,104 | **Moderate** | Pachade et al. (Data, MDPI 2021), Table 3 | Optic Disc Edema (blurred disc margins). Distinct from Drusen (`DN`). Published Table 3 lists 96 positive cases across 3,200 images. | One-vs-Rest Binary AUROC, Specificity, Sensitivity (95% Bootstrap CI) |
| **Retinitis Pigmentosa** | `RP` | **Exact** | **10** | 3,190 | **Severely Inadequate** ($N=10$) | Pachade et al. (Data, MDPI 2021), Table 3 | Direct correspondence to hereditary pigmentary retinopathy presenting with bone-spicule deposits. **CRITICAL:** Sample size is severely inadequate for reliable statistical estimation (only 10 cases in full cohort; fewer in sub-partitions). | **Exploratory Descriptive Analysis Only; Ineligible for standalone statistical validation claims** |
| **Myopia** | `MYA` | **Approximate** | **167** | 3,033 | **Adequate** (Sign-level) | Pachade et al. (Data, MDPI 2021), Table 3 | RFMiD v1 annotates Myopic Atrophy (`MYA`) with 167 total cases. Source class includes broader high/pathological myopia fundus signs. | One-vs-Rest Binary AUROC (Conditional on `MYA`), Specificity |
| **Glaucoma** | `ODC` | **Approximate** | **445** | 2,755 | **Robust** (Sign-level only) | Pachade et al. (Data, MDPI 2021), Table 3 | RFMiD v1 annotates Optic Disc Cupping (`ODC`). Cupping is an anatomical sign, NOT an established clinical diagnosis of Glaucoma (physiological cupping exists; normal-tension glaucoma may have subtle cupping). | Ranking Agreement / AUROC vs `ODC`; **Ineligible for definitive Clinical Diagnostic Specificity/F1** |
| **Retinal Detachment** | None (`RT` for exploratory) | **Unavailable** | **25** | 3,175 | **Non-Equivalent & Underpowered** ($N=25$) | Pachade et al. (Data, MDPI 2021), Table 3 | **CRITICAL CORRECTION:** Published Table 3 lists `RT` for Retinal Traction Detachment (25 cases total); no standalone `RD` label exists in the v1 data release. Tractional detachment is etiologically distinct from primary rhegmatogenous or serous detachment. **Direct external evaluation is UNAVAILABLE.** `RT` may only be evaluated as an exploratory secondary analysis and must NOT be reported as definitive external retinal detachment diagnostic performance. | **Direct External Evaluation UNAVAILABLE; Exploratory RT AUROC Only** |

#### Crucial Distinctions and External Evaluation Governance:
1. **Full-Cohort Totals vs. Labelled Evaluation Support:** The published counts above reflect the total RFMiD v1 cohort ($N=3,200$, Table 3). In official challenge releases, the test set (640 images) frequently withholds labels for competition evaluation. Before calculating statistical power or confidence intervals, the actual CSV headers, labels, and row counts must be verified on the exact acquired files. If test set labels are withheld in that release, only the labelled partitions (e.g. train + val, $N=2,560$) constitute the evaluation cohort.
2. **Dataset Version Integrity & Column Verification:** RFMiD v1 and RFMiD v2 possess distinct label inventories. In RFMiD v1, `MS` stands for Macular Scars, while Retinal Detachment is represented only by `RT` (Retinal Traction Detachment). Prior to running inference on any external RFMiD release, the exact CSV header columns must be verified directly against the dataset distribution rather than assuming generic abbreviations like `RD`.
3. **Exact Label Correspondence vs. Clinical Diagnostic Equivalence:** Even where a morphological label correspondence is marked "Exact" (`DR`, `MS`, `CSR`, `RP`, `ODE`), morphological similarity under fundus imaging does not imply full clinical diagnostic identity. For "Approximate" classes (`ODC` $\rightarrow$ Glaucoma, `MYA` $\rightarrow$ Myopia), the evaluation tests feature correlation and lesion sensitivity under explicit caveats, not gold-standard clinical diagnostic concordance. Direct evaluation of Retinal Detachment is marked **Unavailable**.
4. **Multi-Label vs. Mutually Exclusive Evaluation:** RFMiD provides multi-label annotations where single images frequently exhibit comorbid findings (e.g., `DR` + `ODC` + `MS`). Therefore, external evaluation must NEVER force RFMiD images into the internal benchmark's 9-class mutually exclusive softmax task. Instead, external evaluation must be executed as isolated, disease-specific One-vs-Rest binary tasks using model class-probability thresholds, AUROC curves, and Precision-Recall curves.
5. **Statistical Uncertainty & Pre-Specified Metrics:** For all eligible classes, uncertainty will be quantified using 95% non-parametric bootstrap confidence intervals (1,000 stratified resamples). Rare classes (`RP`, `MS`, `RT`) must carry prominent statistical power caveats.
6. **Strict Data Quarantine & Zero Contamination:** External evaluation datasets must remain strictly quarantined from model selection, hyperparameter tuning, and probability calibration. Thresholds calibrated on internal validation must be evaluated out-of-the-box without post-hoc tuning on the external cohort.

### 5.2 Messidor-2 Audit
* **Access Terms:** Permissive research use (Messidor-2 Consortium / Eyepacs).
* **Scope:** 1,200 color fundus images graded on the 5-point ICDR scale for Diabetic Retinopathy and macular edema.
* **Role:** Single-disease external validation for Diabetic Retinopathy sensitivity, specificity, and probability calibration.

### 5.3 ODIR-5K Audit & Limitations
* **Access Terms:** Academic challenge dataset (Peking University, 2019).
* **Limitation:** Labels are assigned at the **patient level** based on bilateral pairs (OD and OS) and clinical text. A monocular fundus model cannot be evaluated cleanly against bilateral patient-level diagnoses without introducing substantial label noise. ODIR-5K non-target classes (Cataract, AMD, Hypertension) will be evaluated exclusively as Out-of-Distribution (OOD) test inputs for **RQ3** (abstention).

---

## 6. Experimental Design & Controlled Model Comparison

### Baseline Architectures
1. **EfficientNet-B0 (Primary Lightweight Baseline):** 5.3M parameters, standard mobile inverted bottleneck CNN.
2. **ResNet-50d (Clinical Standard Comparator):** 25.6M parameters, deep residual network with modified downsampling.
3. **ConvNeXt-Tiny (Modern Pure-ConvNet):** 28.6M parameters, 7x7 depthwise convolutions.

### Controlled Training Protocol
* **Loss Formulation:** Standard Cross-Entropy Loss (unweighted baseline) followed by Class-Weighted Cross-Entropy to address the 18:1 imbalance.
* **Optimization:** AdamW optimizer, cosine annealing learning rate scheduler without restarts.
* **Input Resolution:** $224 \times 224$ pixels (standard) and $384 \times 384$ pixels (high-resolution comparison).
* **Augmentation:** Conservative training-only: Random horizontal/vertical flip, random affine rotation ($\pm 15^\circ$), mild brightness/contrast ($\pm 10\%$).
* **Preprocessing:** Bounding-box fundus extraction (`crop_fundus_area(threshold=10)`: locates the intensity-thresholded non-black bounding box enclosing the fundus aperture) $\rightarrow$ `Resize` $\rightarrow$ `ImageNet Normalize`.
* **Multi-Seed Protocol:** Fixed evaluations across **Seeds 42, 1337, and 2026**.
  * *Interpretation:* Repeated seeds evaluate **training and optimization variability under a fixed split** (stochastic weight initialization and mini-batch ordering), **not random-split variance**.
* **Inference Standard:** Deterministic, autocast-disabled FP32 inference for all reported validation checkpoints.

---

## 7. Metrics, Calibration, and Selective Prediction Framework

### Primary Metric (Checkpoint Selection)
* **Validation Macro-F1 across 9 Classes:**
  $$\text{Macro-F1} = \frac{1}{9} \sum_{c=1}^{9} F1_c$$
  Equal weighting across all 9 disease categories to directly penalize failures on minority classes.

### Secondary Classification Metrics
* **Balanced Accuracy:** Unweighted average of class-specific sensitivities (recalls).
* **Overall Accuracy:** Sample-weighted proportion of correct predictions.
* **Per-Class Metrics:** Sensitivity (Recall), Specificity, Precision, and F1-score.
* **Statistical Uncertainty:**
  * **Aggregate Metrics (Macro-F1, Balanced Accuracy):** 95% non-parametric stratified bootstrap confidence intervals (1,000 resamples).
  * **Rate Metrics (Sensitivity, Specificity):** 95% Wilson score or exact Clopper-Pearson binomial confidence intervals.
* *Metric Exclusion:* Quadratic Weighted Kappa (QWK) is **permanently excluded** from the primary metric suite because nominal disease categories possess no natural ordinal ordering.

### Probability Calibration Metrics
* **Expected Calibration Error (ECE):** Evaluated using $M=15$ bins with confidence histograms.
* **Brier Score:** Mean squared error between predicted probability distributions and one-hot ground truth vectors.
* **Negative Log-Likelihood (NLL).**

### Selective Prediction & Abstention Analysis (RQ3)
* **Risk-Coverage Curve:** Diagnostic error rate plotted as a function of coverage (fraction of predictions retained).
* **Area Under the Risk-Coverage Curve (AURC):** Quantifies uncertainty ranking performance.

---

## 8. Catalog of Preserved Historical Experiments

All prior experiments remain cataloged under their original 10-class task definition:

| Run Identifier | Task & Cohort | Key Hyperparameters | Validation Macro-F1 | Best Epoch | Artifact Location & SHA-256 |
|---|---|---|:---:|:---:|---|
| **Legacy Benchmark** | 10-Class (4,000 pre-augmented images) | Multiple backbones | ~90.5% (Flawed) | N/A | [`docs/audit/LEGACY_BENCHMARK.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/audit/LEGACY_BENCHMARK.md) |
| **Run 1 (Baseline)** | 10-Class Clean (`data_clean/`, $N=562$) | $\text{lr}=3\times 10^{-4}, \text{wd}=1\times 10^{-4}$ | **87.41%** | 19 | `outputs/clean_baseline_efficientnet_b0/`<br>`bac42ea265c881d5447b58498c6a21210c2048d930d812f71fa9b1bdeb7c2edd` |
| **Run 2 (Low LR)** | 10-Class Clean (`data_clean/`, $N=562$) | $\text{lr}=1\times 10^{-4}, \text{wd}=1\times 10^{-4}$ | 86.81% | 19 | `outputs/clean_lr1e4_efficientnet_b0/`<br>`4a2b9aef09d7b9c896c8bcc44416e8f3f92a0d9d83d71ffe5d1521ac9dae2b7c` |
| **Run 3 (High WD)** | 10-Class Clean (`data_clean/`, $N=562$) | $\text{lr}=3\times 10^{-4}, \text{wd}=5\times 10^{-4}$ | 86.85% | 14 | `outputs/clean_wd5e4_efficientnet_b0/`<br>`ff8adae1fe03ce971597ee2e4eb16fa64ba836935de86829385830974fbfbdbb` |

*Reconciliation Note for Run 3:* Run 3 artifacts (`best_model.pth`, `training_history.json`, `val_predictions_reconciled.csv`, `val_confusion_matrix_reconciled.csv`, `val_evaluation_reconciled.json`) were verified on October 10, 2026. The reported validation Macro-F1 of 86.85% and validation accuracy of 87.37% (491/562) are mathematically supported by its checkpoint and prediction ledger.

---

## 9. Reproducibility Requirements & Go/No-Go Gate

### Reproducibility Standards
1. Every training run must record: Git commit hash, config YAML, random seed, training wall-clock time, and checkpoint SHA-256.
2. Every validation evaluation must produce three tied artifacts:
   - `val_predictions_reconciled.csv`
   - `val_confusion_matrix_reconciled.csv`
   - `val_evaluation_reconciled.json`
3. Verification script [`scripts/verify_9class_preflight.py`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/scripts/verify_9class_preflight.py) must pass before model training is launched.

### Go / No-Go Gate Criteria
* **Gate 1 (Task Definition & Protocol):** Protocol v1.1 finalized and committed. *(Passed)*
* **Gate 2 (Manifest & DataLoader Preflight):** 9-class cohort verified (2,612 / 560 / 561) with zero Pterygium samples and test DataLoader untouched. *(Passed)*
* **Gate 3 (External Dataset Mapping Frozen):** RFMiD per-class mapping frozen with documented limitations. *(Passed)*
* **Gate 4 (User Authorization):** Awaiting user sign-off to launch the 3-seed 9-class EfficientNet-B0 baseline training.
