# Experiment Protocol: Controlled 9-Class Posterior-Segment Fundus Baseline (EXP-01)

**Document Identifier:** `docs/experiments/EXP_9CLASS_FUNDUS_PROTOCOL.md`  
**Protocol Version:** 1.1 (Pre-Registration Specification)  
**Creation Date:** 2026-10-13  
**Status:** SPECIFICATION LOCKED — READY FOR PRE-FLIGHT VERIFICATION — NO ACTIVE TRAINING  
**Principal Investigator:** Subhajit Gayen  

---

## 1. Research Questions and Hypotheses

* **Primary Research Question (RQ1 - Unconfounded Benchmark Validity):**  
  What is the true baseline diagnostic performance of a standard convolutional neural network (EfficientNet-B0) on posterior-segment retinal disease classification when imaging modality shortcuts (anterior-segment Pterygium) and cross-split near-duplicate data leakage are strictly eliminated?
  * *Hypothesis 1.1 (Elimination of Modality Shortcuts):* Under a pure 9-class posterior-segment formulation, model evaluation metrics will reflect genuine retinal diagnostic difficulty, eliminating artificial 100% sensitivity/specificity artifacts previously driven by distinguishing anterior ocular surface photographs from retinal fundus photographs.
  * *Hypothesis 1.2 (Optimization Variance across Seeds):* Training across multiple predeclared random seeds will reveal measurable variance across minority classes (e.g. Central Serous Chorioretinopathy, Disc Edema, Retinal Detachment), establishing empirical variance bounds before evaluating subsequent loss or architecture modifications.

* **Secondary Research Question (RQ2 - Reference Loss Characterization):**  
  Does unweighted standard cross-entropy loss provide an adequate supervisory signal across severe class prevalence disparities (ranging from 1.9% to 34.1%), or does it cause minority-class recall degradation compared to majority classes?
  * *Hypothesis 2.1:* Unweighted cross-entropy will optimize global accuracy and majority-class performance (Diabetic Retinopathy, Glaucoma, Healthy) at the expense of minority-class recall. This establishes the exact reference baseline against which subsequent class-weighted and cost-sensitive ablations will be formally contrasted.

---

## 2. Dataset Provenance, Class Mapping, and Frozen Split Counts

### 2.1 Provenance and Physical Acquisition
* **Source Corpus:** *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`, Rashid et al., *Data in Brief* 2024).
* **Clinical Sites:** Two tertiary ophthalmic centers in Bangladesh (Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital).
* **Acquisition Devices:** Topcon TRC-50DX and Topcon NW400 fundus camera systems.
* **Metadata Limitations (CLAIM 2024):** The source dataset release provides no subject identifiers, visit timestamps, eye laterality tags (OD vs. OS), or camera serial linkage. Consequently, partitions are strictly **image-level / source-group-isolated** and cannot be claimed as patient-independent.

### 2.2 Anatomical Modality Isolation (Quarantine of Pterygium)
* Pterygium is an anterior-segment fibrovascular lesion of the conjunctiva and cornea. It does not involve the posterior segment (retina, macula, or optic nerve).
* The 14 Pterygium images (10 Train, 2 Val, 2 Test) in the frozen benchmark `data_clean/` remain permanently quarantined from all 9-class fundus dataloaders, classification heads, and evaluation matrices.
* The frozen directory structure in `data_clean/` remains completely unmodified to preserve cryptographic immutability.

### 2.3 Class Mapping & Label Indexing
The 9-class fundus classification problem is formulated as a single-label, mutually exclusive categorical task using the canonical integer mapping:

| Class Index | Canonical Directory Label | Target Anatomy | Train ($N$) | Val ($N$) | Test ($N$) | Total ($N$) | Prevalence (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| `0` | `Central Serous Chorioretinopathy [Color Fundus]` | Macula / RPE | 49 | 11 | 11 | 71 | 1.90% |
| `1` | `Diabetic Retinopathy` | Retinal Vasculature | 891 | 191 | 191 | 1,273 | 34.10% |
| `2` | `Disc Edema` | Optic Nerve Head | 68 | 15 | 15 | 98 | 2.63% |
| `3` | `Glaucoma` | Optic Nerve Head | 579 | 124 | 124 | 827 | 22.15% |
| `4` | `Healthy` | Posterior Pole | 492 | 105 | 105 | 702 | 18.80% |
| `5` | `Macular Scar` | Central Macula | 217 | 47 | 47 | 311 | 8.33% |
| `6` | `Myopia` | Posterior Pole | 172 | 37 | 37 | 246 | 6.59% |
| `7` | `Retinal Detachment` | Neurosensory Retina | 63 | 13 | 14 | 90 | 2.41% |
| `8` | `Retinitis Pigmentosa` | Mid-Peripheral Retina | 81 | 17 | 17 | 115 | 3.08% |
| **Total** | **9-Class Posterior Fundus Benchmark** | | **2,612** | **560** | **561** | **3,733** | **100.0%** |

*Support Verification:* Minimum positive validation support is 11 (`CSCR`); minimum positive test support is 11 (`CSCR`). All classes have non-zero positive and negative support across all three splits.

---

## 3. Image Preprocessing and Data Augmentation

### 3.1 Preprocessing Pipeline (Deterministic)
1. **Color Decoding & Space:** Load images via OpenCV BGR $\rightarrow$ Convert to sRGB 3-channel format.
2. **Fundus Aperture Border Cropping (`crop_fundus_area`):**
   - Extract the green channel ($G$).
   - Identify non-background mask where $G > 15$.
   - Compute the bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ of the masked pixels and crop the image to excise dead peripheral black camera borders.
   - If mask area is empty ($< 100$ pixels, e.g. severe underexposure), fallback to the original image dimensions without cropping.
3. **Spatial Resizing:** Resize cropped image to fixed square input dimensions **$224 \times 224$ pixels** using bilinear interpolation (`cv2.INTER_LINEAR`).
4. **Channel Normalization:**
   - Scale pixel intensities to $[0.0, 1.0]$.
   - Standardize channels using standard ImageNet parameters:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 3.2 Augmentation Strategy
* **Training Transforms (`augmentation_mode = "conservative"` via Albumentations):**
  - `HorizontalFlip(p=0.5)`
  - `VerticalFlip(p=0.5)`
  - `RandomRotate90(p=0.5)`
  - `ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=0, p=0.5)`
  - `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3)`
* **Validation & Test Transforms:**
  - Strictly deterministic: Aperture crop $\rightarrow$ Resize to $224 \times 224$ $\rightarrow$ ImageNet normalization. Zero stochastic augmentations.

---

## 4. Baseline Architecture, Initialization, and Precision

### 4.1 Architecture & Weights
* **Backbone Architecture:** `EfficientNet-B0` (Tan & Le, ICML 2019; ~5.3M parameters).
  - Selected for continuity with existing project baselines and computational efficiency.
* **Pretrained Weights Identifier:** PyTorch official ImageNet-1k weights:
  `torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1` (alias `DEFAULT`).
* **Classification Head:**
  - Adaptive Average Pooling $\rightarrow$ Flatten ($1,280$ features).
  - Dropout layer with rate $p = 0.2$.
  - Linear projection layer: $\mathbf{W} \in \mathbb{R}^{9 \times 1280}$, $\mathbf{b} \in \mathbb{R}^9$.

### 4.2 Numerical Precision Policy
* **Execution Precision:** FP32 (Single Precision, 32-bit IEEE 754 floating point) throughout training, validation, and evaluation.
* **Mixed Precision Disabled:** PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`) is explicitly disabled during evaluation to prevent non-deterministic numerical rounding in predicted softmax probabilities and operating thresholds.

---

## 5. Optimization, Loss, and Hyperparameters

### 5.1 Training Specifications
* **Loss Function:** Standard Unweighted Multi-Class Cross-Entropy:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^8 y_{i,c} \log \hat{p}_{i,c}$$
  *(Class-weighted cross-entropy, focal loss, and cost-sensitive reweighting are explicitly reserved for subsequent comparative ablation experiments).*
* **Optimizer:** AdamW (`torch.optim.AdamW`).
  - Base Learning Rate: $\eta = 3 \times 10^{-4}$ ($0.0003$).
  - Weight Decay: $\lambda = 1 \times 10^{-4}$ ($0.0001$).
  - Betas: $(\beta_1, \beta_2) = (0.9, 0.999)$.
  - Epsilon: $\epsilon = 1 \times 10^{-8}$.
* **Learning Rate Schedule:** Cosine Annealing (`torch.optim.lr_scheduler.CosineAnnealingLR`):
  - $T_{\max} = 20$ epochs, $\eta_{\min} = 1 \times 10^{-6}$.
* **Batch Size:** $16$ images per mini-batch.
* **Epoch Budget:** Fixed budget of **20 epochs** (no early stopping; peak checkpoint selected retrospectively via validation Macro-F1).
* **Dataloader Configuration:** `num_workers = 2`, `pin_memory = True`.

---

## 6. Reproducibility Settings and Environment Controls

### 6.1 Seed Protocol
* **Predeclared Seeds:** `Seed 42`, `Seed 1337`, `Seed 2026`.
* **RNG Initialization:**
  - Python standard library: `random.seed(seed)`
  - NumPy: `np.random.seed(seed)`
  - PyTorch CPU: `torch.manual_seed(seed)`
  - PyTorch CUDA: `torch.cuda.manual_seed_all(seed)`

### 6.2 Determinism Controls & Hardware Non-Associativity Disclaimer
* **Deterministic Flags & Workspace Configuration:**
  ```python
  import os
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
  import torch

  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
* **Scope of Determinism:**
  - Setting seed controls, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, and cuDNN determinism guarantees internal algorithmic consistency across repeated runs on identical software and hardware.
  - **Important Technical Limitation:** Multi-threaded parallel CUDA kernels (e.g. atomic gradient additions in backpropagation kernels and floating-point reductions) exhibit non-associative floating-point summation ($a + b + c \neq c + b + a$). Therefore, while variance across seeds measures stochastic optimization stability, **strict bitwise cross-platform identity is neither claimed nor guaranteed**.
* **Runtime Software & Hardware Stack (Verified):**
  - Operating System: Linux 6.6.137+bpo-amd64
  - Python: 3.10.12 (venv `/home/silentbyte/Documents/GitHub/eye-disease-classification/.venv`)
  - Framework: PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121
  - Data & ML Libraries: Albumentations 2.0.8, Scikit-learn 1.7.2, Pandas 2.3.3, NumPy 2.2.6
  - Acceleration Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)

---

## 7. Model Selection & High-Specificity Threshold Algorithm (Validation-Only)

### 7.1 Primary Checkpoint Selection Rule
* **Metric:** Peak **Validation Macro-F1** across epochs $1, \dots, 20$.
* **Selection Procedure:** Evaluated on the validation partition (`data_clean/val`, $N=560$) at the end of each epoch using standard mutually exclusive $\text{argmax}_c (\hat{p}_{i,c})$.
* **Tie-Breaking Rule:** If multiple epochs achieve identical Validation Macro-F1 (to 4 decimal places), select the checkpoint with the lower Validation Cross-Entropy Loss.
* **Persistence:** Save selected epoch index, checkpoint file `best_model.pth`, and its SHA-256 digest.

### 7.2 Secondary High-Specificity Operating Threshold Selection Algorithm
For clinical triage applications requiring high specificity operating points, decision thresholds are tuned for each disease category using a **One-vs-Rest (OvR)** formulation:

1. **Model Output Formulation:**
   The network outputs 9-class softmax probabilities $\hat{p}_i \in [0, 1]^9$. Primary multi-class predictions use the standard mutually exclusive $\text{argmax}_c (\hat{p}_{i,c})$ decision rule. Operating thresholds evaluate each class as a secondary binary screening detector using $\hat{p}_{i,c}$ as the continuous risk score.
2. **Target Specificity Levels:**
   $$S^* \in \{0.90, 0.95\}$$
3. **Threshold Selection on Validation Set ($N=560$):**
   For each class $c \in \{0, \dots, 8\}$:
   - Partition validation samples into positives $\mathcal{P}_c = \{j : y_j = c\}$ and negatives $\mathcal{N}_c = \{j : y_j \neq c\}$.
   - Define candidate threshold set $\mathcal{C}_c$ from sorted unique validation predicted probabilities $\hat{p}_{j,c}$ augmented with exact boundary points:
     $$\mathcal{C}_c = \{0.0\} \cup \{\hat{p}_{j,c} : j \in \mathcal{V}\} \cup \{1.0 + \epsilon\} \quad (\text{where } \epsilon = 1 \times 10^{-6})$$
   - For any threshold $t \in \mathcal{C}_c$, define the binary prediction $\hat{y}_{j,c}(t) = \mathbb{I}(\hat{p}_{j,c} \ge t)$.
   - Compute empirical specificity on negatives:
     $$\text{Spec}_c(t) = \frac{\sum_{j \in \mathcal{N}_c} \mathbb{I}(\hat{p}_{j,c} < t)}{|\mathcal{N}_c|}$$
   - Compute empirical sensitivity on positives:
     $$\text{Sens}_c(t) = \frac{\sum_{j \in \mathcal{P}_c} \mathbb{I}(\hat{p}_{j,c} \ge t)}{|\mathcal{P}_c|}$$
   - Construct the feasible threshold candidate set meeting target specificity:
     $$\mathcal{T}_c(S^*) = \{t \in \mathcal{C}_c : \text{Spec}_c(t) \ge S^*\}$$
4. **Boundary Condition & Optimization:**
   - Note that for $t = 1.0 + \epsilon$, every sample satisfies $\hat{p}_{j,c} < t$, strictly guaranteeing an all-negative state where $\text{Spec}_c = 1.0$ and $\text{Sens}_c = 0.0$ regardless of whether any sample has score exactly $1.0$. Thus $\mathcal{T}_c(S^*)$ is mathematically non-empty for any $S^* \le 1.0$.
   - Select the threshold $t_c^*(S^*)$ that maximizes validation sensitivity:
     $$t_c^*(S^*) = \arg\max_{t \in \mathcal{T}_c(S^*)} \text{Sens}_c(t)$$
   - *Tie-Breaking Convention:* If multiple feasible thresholds achieve the identical maximum validation sensitivity, select the **lowest numerical threshold** among them. This tie-breaking rule is an explicit, pre-registered heuristic convention (prioritizing detection of borderline positive candidates over extreme conservatism); it is a fixed convention, not a theoretical guarantee of superior clinical utility.
5. **Validation Verification & Frozen Test Application:**
   - The implementation explicitly computes and asserts that the achieved validation specificity at $t_c^*(S^*)$ satisfies $\text{Spec}_c(t_c^*) \ge S^*$.
   - The selected thresholds $t_{c, 0.90}^*$ and $t_{c, 0.95}^*$ are frozen and applied out-of-the-box to the test set: $\hat{y}_{i,c} = \mathbb{I}(\hat{p}_{i,c} \ge t_c^*)$.
   - **Crucial Methodological Caveat:** These are *validation-targeted operating points*, not mathematical guarantees that test set specificity will achieve $\ge 90\%$ or $\ge 95\%$. Test performance will be reported exactly as observed, with **zero post-hoc threshold adjustment on the test set**.

---

## 8. Primary and Secondary Evaluation Metrics

### 8.1 Primary Benchmark Metrics
All primary metrics are macro-averaged (unweighted arithmetic mean across the 9 classes) to prevent majority classes from dominating evaluation:

1. **Macro-AUROC (One-vs-Rest):**
   $$\text{Macro-AUROC} = \frac{1}{9} \sum_{c=0}^8 \text{AUROC}_c$$
   Calculated via trapezoidal integration of the true positive rate versus false positive rate curve for each class $c$ versus all other classes.
2. **Macro Average Precision (Macro-AUPRC):**
   $$\text{Macro-AUPRC} = \frac{1}{9} \sum_{c=0}^8 \text{AP}_c, \quad \text{where } \text{AP}_c = \sum_{k} (R_{c,k} - R_{c,k-1}) P_{c,k}$$
   Calculated using `sklearn.metrics.average_precision_score(y_true_binary, y_prob_c)`. Defined strictly as the unweighted arithmetic mean of per-class Average Precision.
3. **Macro-F1 (Standard Argmax Multi-Class Decision Rule):**
   $$\text{Macro-F1} = \frac{1}{9} \sum_{c=0}^8 \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   Evaluated under standard mutually exclusive $\text{argmax}_{c} (\hat{p}_c)$ predictions with zero-division handling fixed to $0.0$.

### 8.2 Secondary Clinical Metrics
* **Per-Class Sensitivity (Recall):** $\text{TPR}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$.
* **Per-Class Specificity:** $\text{TNR}_c = \frac{\text{TN}_c}{\text{TN}_c + \text{FP}_c}$.
* **Balanced Accuracy:** Arithmetic mean of sensitivity across all 9 classes (equivalent to Macro-Recall).
* **Overall Accuracy:** $\frac{\sum_{c} \text{TP}_c}{N}$.
* **Operating Point Sensitivity:** Sensitivity observed at validation-locked $\ge 90\%$ and $\ge 95\%$ specificity thresholds.
* **Confusion Matrix:** Full $9 \times 9$ normalized and unnormalized contingency matrices.

*(Note on Quadratic Weighted Kappa: QWK is explicitly excluded from this protocol because retinal disease categories are nominal clinical diagnoses without natural ordinal progression; applying quadratic penalties implies an artificial ordering).*

---

## 9. Confidence Intervals and Uncertainty Reporting

* **Individual Seed Bootstrap Reporting (Primary):**
  - $B = 1,000$ bootstrap iterations executed on the test set predictions separately for each individual seed run (`Seed 42`, `Seed 1337`, `Seed 2026`).
  - Resampling is **stratified by ground truth class** to guarantee that every bootstrap replicate contains non-zero positive cases for all 9 classes.
  - Confidence Interval Method: 95% Empirical Percentile Bootstrap Confidence Interval:
    $$[\text{Metric}_{2.5\%}, \text{Metric}_{97.5\%}]$$
* **Pooled Prediction Bootstrap Reporting (Descriptive Ensemble Only):**
  - If a pooled multi-seed confidence interval is reported, pooling is defined mathematically as the **unweighted ensemble average predicted probability** for each test image $i$ across the three seeds:
    $$\bar{p}_{i,c} = \frac{1}{3} \sum_{s \in \{42, 1337, 2026\}} \hat{p}_{i,c}^{(s)}$$
  - The ensemble class prediction is $\bar{y}_i = \arg\max_c (\bar{p}_{i,c})$.
  - Bootstrap resampling ($B=1,000$) is then performed on these ensemble predictions $\bar{p}_i$.
  - **Critical Methodological Limitation:** The three seed runs were evaluated on the exact same 561 test images, not on three independent patient cohorts. Therefore, any pooled interval is strictly a **descriptive ensemble uncertainty interval over finite test sample variation**, not an expanded-sample statistical test across independent cohorts.
* **Image-Level Clustering Limitation:**
  - In strict compliance with CLAIM 2024 reporting guidance, all bootstrap confidence intervals are explicitly reported as **image-level resampling intervals**. Because public datasets lack patient identifiers, these intervals quantify finite-sample test evaluation variance but **do not account for potential latent intra-patient clustering**.

---

## 10. Checkpoint Selection, Reporting Format, and Artifact Hierarchy

### 10.1 Reporting Format
To ensure transparency, results will be reported in a complete tabular format displaying all three individual seed runs alongside aggregate summary statistics:

| Run Identifier | Seed | Selected Epoch | Val Macro-F1 | Val Loss | Test Macro-F1 [95% CI] | Test Macro-AUROC [95% CI] | Test Macro-AUPRC [95% CI] | Test Balanced Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | 42 | *Recorded* | *Recorded* | *Recorded* | *Observed* | *Observed* | *Observed* | *Observed* |
| Run 2 | 1337 | *Recorded* | *Recorded* | *Recorded* | *Observed* | *Observed* | *Observed* | *Observed* |
| Run 3 | 2026 | *Recorded* | *Recorded* | *Recorded* | *Observed* | *Observed* | *Observed* | *Observed* |
| **Aggregate** | — | — | $\text{Mean} \pm \text{SD}$ | $\text{Mean} \pm \text{SD}$ | $\text{Mean} \pm \text{SD}$ | $\text{Mean} \pm \text{SD}$ | $\text{Mean} \pm \text{SD}$ | $\text{Mean} \pm \text{SD}$ |

### 10.2 Directory & Artifact Hierarchy
All experiment outputs are isolated under `outputs/experiments/exp_01_9class_effnet_baseline/`:

```
outputs/experiments/exp_01_9class_effnet_baseline/
├── exp_01_protocol_summary.json
├── exp_01_aggregate_results.csv
├── seed_42/
│   ├── run_config.yaml
│   ├── best_model.pth
│   ├── best_model.pth.sha256
│   ├── training_history.json
│   ├── val_predictions.csv
│   ├── val_metrics.json
│   ├── test_predictions.csv
│   ├── test_metrics.json
│   └── test_confusion_matrix.png
├── seed_1337/
│   └── ... (identical hierarchy)
└── seed_2026/
    └── ... (identical hierarchy)
```

---

## 11. Test-Set Historical Exposure Disclosure

* **Mandatory Disclosure Statement:**
  > *The internal benchmark test partition (`data_clean/test`, $N=561$ nine-class posterior-segment fundus images) was evaluated during earlier 10-class baseline experiments in the project's development history. Therefore, the internal test partition cannot and will not be presented as a pristine, never-before-accessed confirmatory test cohort. It serves strictly as a standardized internal comparator across consistent, frozen experimental conditions. Definitive confirmatory validation of generalization capability requires independent external cohorts.*
* **Quarantine Enforcement:**
  - During baseline training and model selection, the test partition is strictly quarantined.
  - The test set is evaluated exactly once per seed after the model checkpoint and operating thresholds are permanently locked.

---

## 12. External RFMiD Exploratory Restrictions

* **Secondary Exploratory Status:** All evaluations on the external RFMiD dataset ($N=3,200$) remain strictly secondary and exploratory.
* **Separation of Multi-Label and Multi-Class Evaluation:** Because RFMiD annotations are multi-label, external evaluation must NEVER force RFMiD images into the internal benchmark's 9-class mutually exclusive softmax task. It must be evaluated exclusively via independent One-vs-Rest binary heads.
* **Diagnostic Non-Equivalence Safeguard:**
  - Retinal Detachment (`RT` on RFMiD is Traction Detachment) is clinically non-equivalent and remains marked as **Unavailable** for confirmatory evaluation.
  - Glaucoma (`ODC` is Optic Disc Cupping) and Myopia (`MYA` is Myopic Atrophy) are sign-level morphological approximations.
* **Prohibition of External Tuning:** Under no circumstances will external RFMiD performance be used to tune models, adjust decision thresholds, or guide feature selection for the internal benchmark.

---

## 13. Pre-Run Checklist & Authorization Gate

Before executing training or generating new weights, the following verification gates must pass:

- [x] **Gate 1 (Benchmark Immutability):** `data_clean/` file hashes and manifest row counts verified unchanged ($N=3,747$ total; $N=3,733$ fundus cohort). Verified via `scripts/verify_9class_preflight.py`.
- [x] **Gate 2 (Pterygium Exclusion):** Data loaders verify exact sample counts: Train $N=2,612$, Val $N=560$, Test $N=561$, zero Pterygium samples loaded. Verified via `scripts/verify_9class_preflight.py`.
- [x] **Gate 3 (QWK Removal):** Quadratic Weighted Kappa removed from primary and secondary metrics.
- [x] **Gate 4 (High-Specificity Threshold Algorithm Specified):** OvR threshold selection on validation negatives ($\ge 0.90, \ge 0.95$), tie-breaking, and validation-targeted caveat formalized.
- [x] **Gate 5 (Reproducibility & Precision Detailed):** Seed controls, cuDNN flags, software/hardware stack, and CUDA floating-point non-associativity limitations recorded.
- [x] **Gate 6 (Reporting Structure Formalized):** Tabular per-seed reporting alongside mean ± SD and image-level stratified bootstrap CIs specified.
- [ ] **Gate 7 (User Explicit Authorization):** Explicit confirmation obtained from the investigator to initiate training.

---
*End of Protocol Specification.*
