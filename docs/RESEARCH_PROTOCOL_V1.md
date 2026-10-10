# Research Protocol v1.0: Leakage-Aware, Externally Evaluated Retinal Disease Classification

**Project:** Classification of Eye Diseases from Color Fundus Photographs  
**Principal Investigator / Author:** Subhajit Gayen  
**Protocol Version:** 1.0 (Draft for Scientific Review)  
**Date:** October 2026  
**Status:** FROZEN PENDING REVIEW — NO ACTIVE MODEL TRAINING  

---

## 1. Executive Summary & Research Motivation

The initial phase of this project established a 10-class benchmark on the *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`, Rashid et al., *Data in Brief* 2024). Through extensive forensic auditing, we eliminated synthetic duplicate leakage, quarantined cross-split burst frames, and produced a cleaned image-level benchmark (`data_clean/`, 3,747 images).

However, rigorous scientific scrutiny reveals a fundamental task-definition flaw in the 10-class formulation:
1. **Imaging Modality Mismatch:** The source dataset combines **nine categories of posterior-segment retinal fundus photographs** with **one category of anterior-segment photography (Pterygium)**. Pterygium affects the conjunctiva and cornea on the outer surface of the eye. Classifying Pterygium against retinal fundus images is not a diagnostic test of retinal pathology; it is an anatomical shortcut distinguishing external facial/ocular structures from the retina.
2. **Severe Class Scarcity:** Pterygium contains only 14 unique images in `data_clean/` (10 train, 2 validation, 2 test). Reporting 100% sensitivity on 2 validation images provides an illusory sense of clinical capability.
3. **Absence of Patient Identifiers:** The source hospital providers permanently scrubbed patient identifiers, visit dates, and laterality tags (OD/OS). Thus, the benchmark is **image-level/source-group-isolated**, not patient-independent.
4. **Generalization Gap:** Achieving high internal validation Macro-F1 does not prove clinical validity or external transferability.

This protocol redefines the primary task as a **homogeneous 9-class retinal fundus classification problem** and establishes an auditable research program centered on **modality consistency, external generalization, probability calibration, and uncertainty-aware selective prediction**.

---

## 2. Research Questions and Hypotheses

### Primary Research Questions

* **RQ1 (Benchmark Validity & Shortcut Mitigation):**  
  Does eliminating the anterior-segment modality mismatch and near-duplicate leakage reveal the true baseline performance of deep neural networks on posterior-segment retinal disease classification?  
  *Hypothesis 1.1:* When evaluated on a pure 9-class fundus task, baseline classification metrics will reflect genuine retinal diagnostic difficulty, eliminating artificial 100% metrics driven by imaging modality shortcuts.

* **RQ2 (Cross-Dataset External Generalization):**  
  How effectively do retinal classifiers trained on a single-center, single-device fundus benchmark generalize to independent multicenter datasets (e.g., RFMiD, ODIR-5K) acquired under diverse clinical protocols?  
  *Hypothesis 2.1:* Models achieving >85% internal validation Macro-F1 will experience statistically significant performance degradation on external datasets due to domain shift, differences in camera optics, and label co-morbidity.

* **RQ3 (Uncertainty-Aware Prediction & Abstention):**  
  Can calibrated predictive uncertainty (via temperature scaling or Monte Carlo dropout) reliably identify ambiguous, out-of-distribution, or erroneous classifications, enabling safe clinical referral/deferral?  
  *Hypothesis 3.1:* Selective prediction with confidence-based abstention will significantly reduce high-risk diagnostic errors (e.g., misclassifying Retinal Detachment or Diabetic Retinopathy as Healthy) across a coverage-risk trade-off curve.

---

## 3. Final Task Definition & Class Dictionary

### Primary Task: 9-Class Posterior-Segment Retinal Disease Classification
The primary classification task operates strictly on color fundus photographs showing the posterior segment (retina, optic disc, macula, and vascular arcade).

| Class ID | Standard Class Label | Anatomical Target | ICD-10 Alignment | Primary Visual Features |
|---|---|---|---|---|
| 0 | `Central Serous Chorioretinopathy [Color Fundus]` | Macula / RPE | H35.71 | Serous neurosensory retinal detachment at macula, localized elevation |
| 1 | `Diabetic Retinopathy` | Retinal Vasculature | E11.3 / H36.0 | Microaneurysms, hemorrhages, hard exudates, cotton wool spots |
| 2 | `Disc Edema` | Optic Nerve Head | H47.33 | Blurring of disc margins, hyperemia, peripapillary flame hemorrhages |
| 3 | `Glaucoma` | Optic Nerve Head | H40.9 | Increased cup-to-disc ratio (CDR > 0.6), neuroretinal rim thinning, notching |
| 4 | `Healthy` | Entire Posterior Pole | Z01.00 | Crisp disc margins, normal CDR (<0.4), intact macula, clear vasculature |
| 5 | `Macular Scar` | Central Macula | H35.30 | Fibrotic white/yellow chorioretinal scarring, hyperpigmentation |
| 6 | `Myopia` (Pathological / High) | Posterior Pole | H44.2 | Peripapillary atrophy (temporal crescent), tilted disc, lacquer cracks |
| 7 | `Retinal Detachment` | Peripheral/Posterior Retina | H33.0 / H33.2 | Corrugated grey-white elevated retinal folds, vascular undulation |
| 8 | `Retinitis Pigmentosa` | Mid-Peripheral Retina | H35.54 | Bone-spicule pigment deposits, arteriolar attenuation, waxy pallor of disc |

### Explicit Modality Separation: Pterygium
* **Modality:** Anterior-segment photography (cornea, sclera, iris, conjunctiva).
* **Clinical Description:** Fibrovascular growth of the bulbar conjunctiva encroaching onto the cornea.
* **Protocol Action:** Permanently excluded from the primary retinal fundus classification task.
* **Accounting:** All 14 Pterygium images (10 train, 2 val, 2 test) are archived and documented in `outputs/audit/excluded_pterygium_anterior_images.csv`. They are not deleted from the physical repository to preserve `data_clean/` immutability, but are masked out from all 9-class fundus dataloaders and evaluation matrices.

---

## 4. Dataset Inventory & Known Limitations

### Internal Dataset (`data_clean/` 9-Class Subset)
* **Physical Dataset Root:** `data_clean/` (Frozen, 3,747 physical image files).
* **Fundus-Only Evaluation Cohort:** Exactly 3,733 images across 9 classes.

| Split | Number of Images | Percentage | Role in Protocol |
|---|:---:|:---:|---|
| `data_clean/train` | **2,612** | 69.97% | Model parameter optimization (stochastic gradient descent) |
| `data_clean/val` | **560** | 15.00% | Hyperparameter tuning, checkpoint selection, calibration fitting |
| `data_clean/test` | **561** | 15.03% | **STRICTLY QUARANTINED** internal test cohort; evaluated only at final milestone |
| **Total 9-Class** | **3,733** | 100.0% | Posterior-segment cohort |

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
| **Sum** | **2,612** | **560** | **561** | **3,733** | **100.0%** |

### Known Methodological & Clinical Limitations
1. **Absence of Patient Identifiers:** No hospital IDs, patient demographic data, or eye laterality tags are present. DINOv2 visual grouping mitigates burst-capture leakage, but true patient-level split independence cannot be asserted.
2. **Severe Class Imbalance:** Diabetic Retinopathy and Glaucoma represent 56.25% of the dataset, whereas CSCR, Disc Edema, and Retinal Detachment each represent <3%.
3. **Single Institutional Source:** Collected from two regional eye hospitals in Bangladesh, introducing geographic, demographic, and device-specific domain priors.

---

## 5. External Dataset Feasibility & Mapping Audit

To test RQ2 (generalization), independent external retinal datasets must be evaluated without fine-tuning or threshold manipulation.

| External Dataset | Source / Reference | Cohort Size & Format | Compatible Disease Classes | Mapping Feasibility & Caveats |
|---|---|---|---|---|
| **RFMiD** (Retinal Fundus Multi-Disease) | Pachade et al. (*Sci. Data* 2021) | 3,200 color fundus images, multicenter (India), 45-degree FOV | DR, Glaucoma, Pathological Myopia, Macular Scar, Retinitis Pigmentosa, CSCR, Disc Edema, Retinal Detachment, Normal | **High Feasibility.** Matches all 9 classes. **Caveat:** Annotations are multi-label. Requires mapping to mutually exclusive primary diagnosis or evaluating via disease-specific One-vs-Rest recall. |
| **ODIR-5K** (Ocular Disease Intelligent Recognition) | Peking University / Shanggong Medical (2019) | 10,000 images (5,000 bilateral patients), multi-hospital | Normal, DR, Glaucoma, Pathological Myopia | **Moderate Feasibility.** Direct mapping for 4 classes (Normal, DR, Glaucoma, Myopia). Other classes (Cataract, AMD, Hypertension) serve as Out-Of-Distribution (OOD) tests for abstention. |
| **Messidor-2** | Decencière et al. (France) | 1,200 color fundus images, multiple French hospitals | Diabetic Retinopathy, Normal | **High Feasibility (Sub-task).** Ideal for evaluating external DR sensitivity, specificity, and probability calibration. |
| **STARE** | Hoover et al. (UCSD) | 397 color fundus images, 35-degree FOV | DR, Glaucoma, Retinal Detachment, Retinitis Pigmentosa, CSCR | **Moderate Feasibility.** Small historical dataset; significant optical and color variance. |

### Cross-Dataset Contamination Safeguard
Prior to executing inference on any external cohort:
1. Run cryptographic SHA-256 and perceptual hashing (pHash/dHash, Hamming distance $\le 10$) between `data_clean/` and the external dataset.
2. Compute DINOv2 cosine similarities to verify zero cross-dataset image overlap.

---

## 6. Experimental Design & Controlled Model Comparison

### Model Architectures Under Comparison
All architectures will be evaluated under an identical training pipeline:

1. **EfficientNet-B0 (Primary Baseline):** Standard lightweight CNN baseline (5.3M parameters).
2. **ResNet-50d (Clinical Standard Comparison):** Deep residual baseline widely cited in medical literature (25.6M parameters).
3. **ConvNeXt-Tiny (Modern Pure-ConvNet):** Modern 7x7 depthwise separable architecture (28.6M parameters).

### Controlled Training Protocol
* **Loss Function:** Standard unweighted Cross-Entropy Loss (baseline) followed by Class-Weighted Cross-Entropy / Focal Loss ($\gamma=2.0$) to address the 18:1 imbalance.
* **Optimization:** AdamW optimizer, cosine annealing learning rate scheduler without restarts.
* **Input Resolution:** $224 \times 224$ pixels (standard) and $384 \times 384$ pixels (high-resolution comparison).
* **Augmentations:** Conservative training-only: Random horizontal/vertical flip, random affine rotation ($\pm 15^\circ$), mild brightness/contrast ($\pm 10\%$).
* **Deterministic Inference:** Autocast-disabled FP32 inference, deterministic center-crop preprocessing (`crop_fundus_area(threshold=10)` $\rightarrow$ `Resize` $\rightarrow$ `ImageNet Normalize`).
* **Random Seed Protocol:** Fixed multi-seed evaluation across **Seeds 42, 1337, and 2026**. Report mean $\pm$ standard deviation across seeds.

---

## 7. Metrics, Calibration, and Selective Prediction Framework

### Primary Metric (Single Endpoint for Checkpoint Selection)
* **Validation Macro-F1 (9 classes):**
  $$\text{Macro-F1} = \frac{1}{9} \sum_{c=1}^{9} F1_c$$
  Chosen because it weights all 9 disease classes equally, directly penalizing failures on minority classes (CSCR, Disc Edema, Retinal Detachment).

### Secondary Performance Metrics
* **Balanced Accuracy:** Average class-specific recall.
* **Overall Accuracy:** Sample-weighted correct predictions.
* **Cohen's Quadratic Weighted Kappa ($\kappa$):** Inter-rater reliability measure against ground truth.
* **Per-Class Metrics:** Sensitivity (Recall), Specificity, Precision, and F1 with 95% Wilson score confidence intervals.

### Probability Calibration Metrics
* **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  using $M=15$ equal-frequency or equal-width bins.
* **Brier Score:** Mean squared error between predicted probability vectors and one-hot ground-truth vectors.
* **Negative Log-Likelihood (NLL).**

### Selective Prediction & Abstention Analysis (RQ3)
* **Coverage vs. Risk Curves:** Plot diagnostic error rate as a function of coverage (fraction of patients retained vs. referred for specialist human review).
* **Area Under the Risk-Coverage Curve (AURC):** Quantifies uncertainty ranking quality.

---

## 8. Catalog of Preserved Historical Experiments

To preserve scientific audit integrity, prior experiments remain frozen, logged, and isolated:

| Experiment Identifier | Task Definition | Architecture | Key Hyperparameters | Best Val Macro-F1 | Best Val Epoch | Artifact Location |
|---|---|---|---|:---:|:---:|---|
| **Legacy Benchmark** | 10-Class (Synthetic Twins Contaminated) | Multiple (ConvNeXt, ViT, etc.) | Pre-augmented 4,000 images | ~90.5% (Flawed) | N/A | `docs/audit/LEGACY_BENCHMARK.md` |
| **Run 1 (Baseline)** | 10-Class Clean (`data_clean/`) | EfficientNet-B0 | $\text{lr}=3\times 10^{-4}, \text{wd}=1\times 10^{-4}$ | **87.41%** | 19 | `outputs/clean_baseline_efficientnet_b0/` |
| **Run 2 (Low LR)** | 10-Class Clean (`data_clean/`) | EfficientNet-B0 | $\text{lr}=1\times 10^{-4}, \text{wd}=1\times 10^{-4}$ | 86.81% | 19 | `outputs/clean_lr1e4_efficientnet_b0/` |
| **Run 3 (Higher WD)** | 10-Class Clean (`data_clean/`) | EfficientNet-B0 | $\text{lr}=3\times 10^{-4}, \text{wd}=5\times 10^{-4}$ | 86.85% | 14 | `outputs/clean_wd5e4_efficientnet_b0/` |

*Note: All three completed runs evaluated the 10-class problem and cannot be compared directly with future 9-class fundus-only models.*

---

## 9. Reproducibility Requirements & Go/No-Go Gate

### Reproducibility Standards
1. Every training run must log its full Git commit hash, config YAML file, random seed, training duration, and checkpoint SHA-256.
2. Every evaluation run must generate three tied artifacts derived from identical prediction tensors:
   - `val_predictions_reconciled.csv` (sample-level predictions, true labels, softmax probabilities)
   - `val_confusion_matrix_reconciled.csv` (integer matrix)
   - `val_evaluation_reconciled.json` (scalar metrics verified with assertion tests)

### Go / No-Go Gate Criteria Prior to Any New Training Run
* **Gate 1 (Task Definition Sign-off):** Confirmation of the 9-class fundus task definition and Pterygium exclusion.
* **Gate 2 (External Dataset Protocol Review):** Alignment on the external evaluation dataset selection (e.g., RFMiD multi-label mapping protocol).
* **Gate 3 (Test Set Quarantine):** Internal test split (`data_clean/test`) locked with evaluation flags disabled (`evaluate_test: false`).
* **Current Status:** **TRAINING PAUSED. Awaiting User Review and Authorization.**
