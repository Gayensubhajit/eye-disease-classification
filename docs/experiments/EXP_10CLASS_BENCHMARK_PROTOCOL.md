# Experiment Protocol: Controlled 10-Class Full Benchmark (EXP-02 / Track A)

**Document Identifier:** `docs/experiments/EXP_10CLASS_BENCHMARK_PROTOCOL.md`  
**Protocol Version:** 1.0 (Pre-Registration Specification)  
**Creation Date:** 2026-10-13  
**Status:** SPECIFICATION LOCKED — READY FOR PRE-FLIGHT VERIFICATION — NO ACTIVE TRAINING  
**Principal Investigator:** Subhajit Gayen  

---

## 1. Executive Summary & Dual-Track Research Framework

This protocol defines the formal experimental specification for **Track A (10-Class Benchmark)** of the *Classification of Eye Diseases from Color Fundus Images* project.

### 1.1 The Two-Track Orthogonality Mandate
To maintain uncompromising scientific integrity while addressing both the original research project objective and retinal imaging rigor, this project operates two parallel, decoupled research tracks:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                DUAL-TRACK RESEARCH DESIGN                               │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ TRACK A: 10-Class Full Benchmark (EXP-02) │ TRACK B: 9-Class Fundus Benchmark (EXP-01) │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Objective: Complete 10-class scope      │ • Objective: Pure posterior-segment retinal│
│   defined in the original dataset release  │   classification free of modality shortcuts│
│ • Cohort: All 3,747 images in data_clean/ │ • Cohort: 3,733 posterior fundus images    │
│   (Train: 2,622 | Val: 562 | Test: 563)   │   (Train: 2,612 | Val: 560 | Test: 561)    │
│ • Pterygium: Retained (N=14) with audited  │ • Pterygium: Quarantined (N=14 excluded)   │
│   modality mismatch and discrete reporting│ • Comparison Target: Retinal literature    │
│ • Comparison Target: Published 10-class    │   benchmarks (RFMiD, ODIR, APTOS)          │
│   papers on Mendeley dataset              │                                            │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Strict Non-Comparability Rule:** Track A (10-class) and Track B (9-class) evaluate different anatomical scopes, different sample sizes, and different chance baselines ($10.0\%$ vs. $11.11\%$). Their performance metrics MUST NEVER be directly compared as if they were the same task. All manuscripts, reports, and tables must present them in separate, dedicated sections.

---

## 2. Research Questions and Hypotheses (Track A)

* **Primary Research Question (RQ1 - 10-Class Baseline under Leakage Control):**  
  What is the true diagnostic baseline performance of a standard convolutional neural network (EfficientNet-B0) across all 10 nominal disease categories when trained and evaluated on an audited, leakage-controlled partition of the Mendeley dataset?
  * *Hypothesis 1.1 (Defensible Benchmark Baseline):* On an image-level split stripped of duplicate leakage, EfficientNet-B0 will achieve a stable baseline (Macro-F1 ~85–88%), establishing an uninflated foundation against which subsequent architectural ablations can be legitimately measured.

* **Secondary Research Question (RQ2 - Modality Shortcut and Extreme Scarcity Impact):**  
  How does the retention of anterior-segment Pterygium photographs ($N=14$ total; 10 Train, 2 Val, 2 Test) affect model training dynamics, macro-averaged metrics, and clinical interpretability?
  * *Hypothesis 2.1 (Feature Disparity Shortcut):* Because Pterygium images capture the anterior ocular surface (eyelids, eyelashes, sclera, cornea) while the remaining 9 classes capture posterior retinal fundus tissue, deep vision models will readily separate Pterygium with near 100% specificity and sensitivity, artificially elevating Macro-AUROC and Macro-F1 unless per-class metrics and contingency counts are explicitly reported.
  * *Hypothesis 2.2 (Discrete Uncertainty in Severe Imbalance):* With only 2 positive samples in validation and 2 in test, empirical sensitivity for Pterygium is restricted to discrete multiples of $50\%$ ($\{0.0, 0.5, 1.0\}$). Reporting exact contingency counts (TP, FP, TN, FN) and Wilson score intervals is required to prevent misleading claims of diagnostic capability.

* **Tertiary Research Question (RQ3 - Controlled Outperformance of Published Literature):**  
  Can novel architectural components (BiomedCLIP foundation embeddings, CBAM dual attention, multi-scale feature pyramid fusion, and high-resolution ConvNeXt) outperform published literature models (Rashid et al. 2021/2024, PLoS ONE 2023, EyeFusionNet 2023) under identical, leakage-controlled splits?
  * *Hypothesis 3.1:* Controlled architectural enhancements targeting small retinal lesions (microaneurysms, macular fluid, subtle optic disc margin blurring) will produce statistically significant gains in difficult posterior minority classes (CSCR, Glaucoma, Disc Edema, Macular Scar) rather than relying on anterior-segment shortcut learning.

---

## 3. Dataset Provenance, Modality Audit, and Frozen Split Counts

### 3.1 Provenance and Physical Acquisition
* **Source Corpus:** *Eye Disease Image Dataset* (Mendeley Data DOI: `10.17632/s9bfhswzjb.1`, Rashid et al., *Data in Brief* 2024).
* **Clinical Sites:** Two tertiary ophthalmic centers in Bangladesh (Anwara Hamida Eye Hospital and B.N.S.B. Zahurul Haque Eye Hospital).
* **Acquisition Devices:** Topcon TRC-50DX and Topcon NW400 fundus camera systems.
* **CLAIM 2024 Metadata Limitation:** The public dataset provides no subject IDs, visit timestamps, eye laterality indicators (OD vs. OS), or camera serial linkage. Therefore, the partition is strictly **source-group and duplicate-isolated at the image level**, and cannot be claimed as patient-level independent.

### 3.2 Anatomical & Optical Modality Audit: Pterygium
An empirical forensic examination of the 14 Pterygium images confirms the following anatomical and optical realities:
1. **Anatomical Target:** Pterygium is an external ocular surface condition (fibrovascular growth over the conjunctiva and cornea).
2. **Optical Setup:** While digitized at the same camera resolution ($2004 \times 1690$ pixels) and framed with a circular aperture mask, the images were captured using an external anterior-segment photography mode/lens attachment. They depict external ocular anatomy—including eyelashes, eyelid margins, scleral vessels, and the iris—rather than dilated pupil views of the retina.
3. **Clinical Implication:** A classifier distinguishing Pterygium from fundus diseases is primarily learning an *optical and anatomical modality boundary* (external eye vs. posterior pole), not a subtle retinal pathology distinction.
4. **Methodological Handling:** Rather than silently discarding Pterygium or naively celebrating 100% metrics, Track A explicitly retains Pterygium, audits its presence, and requires full contingency disclosure.

### 3.3 Class Mapping & Frozen Partition Accounting
All 10 classes are indexed alphabetically to preserve deterministic canonical mapping:

| Class Index | Canonical Directory Label | Anatomical Segment | Train ($N$) | Val ($N$) | Test ($N$) | Total ($N$) | Prevalence (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| `0` | `Central Serous Chorioretinopathy [Color Fundus]` | Posterior (Macula) | 49 | 11 | 11 | 71 | 1.89% |
| `1` | `Diabetic Retinopathy` | Posterior (Retina) | 891 | 191 | 191 | 1,273 | 33.97% |
| `2` | `Disc Edema` | Posterior (Optic Disc) | 68 | 15 | 15 | 98 | 2.62% |
| `3` | `Glaucoma` | Posterior (Optic Disc) | 579 | 124 | 124 | 827 | 22.07% |
| `4` | `Healthy` | Posterior (Normal Retina) | 492 | 105 | 105 | 702 | 18.73% |
| `5` | `Macular Scar` | Posterior (Macula) | 217 | 47 | 47 | 311 | 8.30% |
| `6` | `Myopia` | Posterior (Posterior Pole) | 172 | 37 | 37 | 246 | 6.57% |
| `7` | `Pterygium` | **Anterior (Cornea/Sclera)** | **10** | **2** | **2** | **14** | **0.37%** |
| `8` | `Retinal Detachment` | Posterior (Retina) | 63 | 13 | 14 | 90 | 2.40% |
| `9` | `Retinitis Pigmentosa` | Posterior (Peripheral) | 81 | 17 | 17 | 115 | 3.07% |
| **Total** | **Track A: 10-Class Benchmark** | | **2,622** | **562** | **563** | **3,747** | **100.0%** |

*Verification:*
$$\text{Train (2,622)} + \text{Val (562)} + \text{Test (563)} = 3,747 \text{ images}$$
$$\text{Track A (3,747)} - \text{Pterygium (14)} = \text{Track B (3,733)}$$

---

## 4. Preprocessing Pipeline and Data Augmentation

### 4.1 Deterministic Preprocessing Pipeline
1. **Color Space Decoding:** Load via OpenCV BGR $\rightarrow$ Convert to sRGB.
2. **Fundus / Aperture Border Cropping (`crop_fundus_area`):**
   - Extract the green channel ($G$).
   - Identify active pixels ($G > 15$).
   - Compute bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ and crop peripheral dead black border.
   - Fallback to full image if active mask area $< 100$ pixels.
   - *Note on Pterygium:* Forensic verification confirms that the circular mask on Pterygium images is properly detected and cropped by this exact same routine.
3. **Spatial Normalization:** Resize cropped image to $224 \times 224$ pixels via bilinear interpolation (`cv2.INTER_LINEAR`).
4. **Channel Standardization:** Scale to $[0.0, 1.0]$ and apply standard ImageNet parameters:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 4.2 Augmentation Strategy
* **Training Transforms (`augmentation_mode = "conservative"`):**
  - `HorizontalFlip(p=0.5)`
  - `VerticalFlip(p=0.5)`
  - `RandomRotate90(p=0.5)`
  - `ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=0, p=0.5)`
  - `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3)`
* **Validation & Test Transforms:**
  - Strictly deterministic: Aperture Crop $\rightarrow$ Bilinear Resize ($224 \times 224$) $\rightarrow$ ImageNet Standardization. Zero stochastic augmentations.

---

## 5. Baseline Architecture, Numerical Precision, and Optimization

### 5.1 Primary Baseline Model
* **Architecture:** `EfficientNet-B0` (~5.3M parameters).
* **Pretrained Weights:** `torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1`.
* **Head Configuration:**
  - Global Average Pooling $\rightarrow$ Flatten ($1,280$ features).
  - Dropout ($p = 0.2$).
  - Linear projection: $\mathbf{W} \in \mathbb{R}^{10 \times 1280}$, $\mathbf{b} \in \mathbb{R}^{10}$.

### 5.2 Numerical Precision Policy
* **Execution Precision:** Pure FP32 (Single Precision IEEE 754) across training, validation, and evaluation.
* **Mixed Precision Disabled:** `use_amp: False` to guarantee deterministic gradient accumulation and prevent rounding discrepancies in softmax risk scores.

### 5.3 Optimization Specifications
* **Reference Loss:** Standard Unweighted Multi-Class Cross-Entropy:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^9 y_{i,c} \log \hat{p}_{i,c}$$
  *(Weighted cross-entropy and focal loss are reserved for explicit comparative ablation experiments).*
* **Optimizer:** AdamW (`lr = 0.0003`, `weight_decay = 0.0001`, `betas = (0.9, 0.999)`, `eps = 1e-8`).
* **LR Schedule:** Cosine Annealing ($T_{\max} = 20$ epochs, $\eta_{\min} = 1 \times 10^{-6}$).
* **Batch Size:** $16$.
* **Epoch Budget:** Fixed budget of 20 epochs (no early stopping).
* **Dataloader Workers:** `num_workers = 2`, `pin_memory = True`.

---

## 6. Reproducibility Controls & Environment Specifications

### 6.1 Random Seeds
* **Predeclared Seeds:** `Seed 42`, `Seed 1337`, `Seed 2026`.
* Initialized synchronously across Python `random`, NumPy `random`, PyTorch CPU, and PyTorch CUDA.

### 6.2 Deterministic Execution Configuration
```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 6.3 Hardware & Software Environment
* **OS:** Linux 6.6.137+bpo-amd64
* **Python:** 3.10.12 (venv `.venv`)
* **PyTorch Stack:** PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121
* **Compute Device:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
* **Non-Associativity Disclaimer:** In accordance with IEEE 754 parallel floating-point standards, multi-threaded parallel reduction operations on GPU architectures exhibit non-associative additions ($a + b + c \neq c + b + a$). Algorithmic determinism is maintained within this hardware/software setup, but bitwise cross-platform identity is not claimed.

---

## 7. Model Selection & Decision Thresholds (Validation-Only)

### 7.1 Primary Checkpoint Selection Rule
* **Metric:** Peak **Validation Macro-F1** across 20 epochs evaluated on `data_clean/val` ($N=562$) using mutually exclusive $\text{argmax}_c (\hat{p}_{i,c})$.
* **Tie-Breaker Rule:** Lower Validation Cross-Entropy Loss.
* **Artifacts Persisted:** Selected epoch number, `best_model.pth`, SHA-256 cryptographic hash, validation confusion matrix, and prediction probability logs.

### 7.2 Secondary High-Specificity Operating Threshold Selection
For secondary One-vs-Rest (OvR) screening evaluation:
1. **Candidate Grid:**
   $$\mathcal{C}_c = \{0.0\} \cup \{\hat{p}_{j,c} : j \in \mathcal{V}\} \cup \{1.0 + \epsilon\} \quad (\epsilon = 1 \times 10^{-6})$$
2. **Target Specificity Levels:** $S^* \in \{0.90, 0.95\}$.
3. **Threshold Optimization:**
   $$t_c^*(S^*) = \arg\max_{t \in \mathcal{T}_c(S^*)} \text{Sens}_c(t), \quad \text{where } \mathcal{T}_c(S^*) = \{t \in \mathcal{C}_c : \text{Spec}_c(t) \ge S^*\}$$
4. **Tie-Breaker:** Lowest numerical threshold among candidates achieving identical maximum sensitivity.
5. **Special Handling for Pterygium ($N=2$ Positives):**
   - Because $|\mathcal{P}_{\text{Pterygium}}| = 2$, validation sensitivity can only take values in $\{0.0, 0.5, 1.0\}$.
   - If both validation Pterygium samples have high predicted probability (e.g. $> 0.95$) and all 560 negative samples have probability $< 0.10$, threshold optimization will naturally yield $t^* \approx 0.10 - 0.95$ with $100\%$ specificity and $100\%$ sensitivity.
   - The protocol requires reporting the exact threshold value, validation specificity, and test set contingency counts.

---

## 8. Evaluation Metrics and Reporting Standards

### 8.1 Primary Metrics
* **Macro-AUROC (OvR):** Unweighted arithmetic mean of per-class One-vs-Rest AUROC curves.
* **Macro-AUPRC (OvR):** Unweighted arithmetic mean of per-class Average Precision (`average_precision_score`).
* **Macro-F1:** Unweighted arithmetic mean of per-class F1-scores under standard multi-class argmax decision rule:
  $$\text{Macro-F1} = \frac{1}{10} \sum_{c=0}^9 \text{F1}_c$$

### 8.2 Secondary Metrics & Contingency Disclosures
* **Per-Class Sensitivity, Specificity, Precision, and F1.**
* **Balanced Accuracy:** Arithmetic mean of sensitivity across all 10 classes.
* **Full $10 \times 10$ Confusion Matrix:** Both absolute integer counts and row-normalized percentages.
* **Pterygium Contingency Reporting:** For class index 7 (`Pterygium`), authors must explicitly disclose:
  $$\text{TP}, \quad \text{FP}, \quad \text{TN}, \quad \text{FN}, \quad \text{Support } (N=2)$$
  alongside the exact Wilson score $95\%$ confidence interval for binomial proportion.
* **Exclusion of Quadratic Weighted Kappa (QWK):** QWK is excluded because the 10 disease categories are nominal clinical diagnoses with no inherent severity order.

### 8.3 Confidence Intervals (Bootstrap)
* $B = 1,000$ bootstrap iterations on test set predictions per seed run.
* Resampling is stratified by class label to guarantee non-zero support in every replicate.
* 95% Empirical Percentile Bootstrap Confidence Interval: $[\text{Metric}_{2.5\%}, \text{Metric}_{97.5\%}]$.
* Explicit CLAIM 2024 disclosure: Bootstrap intervals reflect finite test sample variance at the image level; they do not account for unobservable patient clustering.

---

## 9. Literature Comparison Protocol

### 9.1 Context of Prior Published Benchmarks
Published papers on the Mendeley dataset reported high overall accuracy, but exhibited specific methodological vulnerabilities:

| Study | Reported Architecture | Stated Accuracy / F1 | Known Evaluation Vulnerabilities |
|---|---|:---:|---|
| **Rashid et al. (2021/2024)** | ResNet-50 / DenseNet-121 | ~83.4% Acc / 82.8% F1 | Original dataset paper; random split did not audit for cross-split duplicate leakage or burst frames. |
| **PLoS ONE (2023)** | Swin Transformer (Swin-T) | 86.8% Acc / 86.2% F1 | Evaluated on uncleaned raw splits; patient clustering unverified. |
| **EyeFusionNet (IEEE 2023)** | DenseNet-169 + TNT Fusion | 89.2% Acc / 88.7% F1 | Heavy parameter fusion evaluated on uncleaned splits; Pterygium modality shortcut retained without audit. |
| **RETFound (Nature 2023)** | ViT-Large/16 (MAE) | 88.5% – 91.2% Acc | Foundation model fine-tuning; strong representation but evaluated on varying institutional splits. |

### 9.2 Fair Comparison Rules
When comparing our findings to published literature:
1. **Split Provenance Disclosure:** Explicitly disclose that our models are evaluated on `data_clean/` (where 8,266 high-similarity candidate pairs were audited, duplicate pairs quarantined, and cross-split bursts eliminated).
2. **Dual-Track Reporting:** Report Track A (10-class) against 10-class literature, and Track B (9-class) against fundus-only literature. Never claim a 9-class model "beat" a 10-class paper without qualifying the task difference.
3. **Disclose Shortcut Advantage:** Point out that historical 10-class models benefited from the trivial separation of anterior Pterygium; our analysis reports both overall macro-F1 and fundus-restricted sub-macro-F1.

---

## 10. Novel Architecture Ablation Roadmap

To achieve legitimate, scientifically verified outperformance over published baselines, we predeclare the following sequence of controlled architectural ablations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PRE-REGISTERED ABLATION SEQUENCE (TRACK A & B)                  │
├───────┬───────────────────────────────┬────────────────────────────────────────────────┤
│ Stage │ Model / Modification          │ Scientific Hypothesis Tested                   │
├───────┼───────────────────────────────┼────────────────────────────────────────────────┤
│ EXP-A0│ EfficientNet-B0 (Reference)   │ Standard unweighted baseline under FP32.       │
│ EXP-A1│ Loss Re-weighting (Focal/CBCE)│ Mitigates extreme minority penalty (Pterygium, │
│       │                               │ CSCR, Disc Edema) without degrading majority. │
│ EXP-A2│ High-Resolution (384) + CLAHE │ LAB-CLAHE luminance equalization + 384x384    │
│       │                               │ resolution restores microvascular lesion area. │
│ EXP-A3│ ConvNeXt-Small Backbone       │ 7x7 depthwise convolutions provide large       │
│       │                               │ receptive fields suited for retinal geometry.  │
│ EXP-A4│ BiomedCLIP + CBAM Dual        │ Medical domain ViT foundation tokens fused     │
│       │ Attention + Pyramid Fusion    │ with channel/spatial attention on optic disc & │
│       │ (Novel Hybrid Architecture)   │ macular regions outperform generic ImageNet.   │
└───────┴───────────────────────────────┴────────────────────────────────────────────────┘
```

Each ablation will be evaluated across the identical 3 seeds (`42, 1337, 2026`) under the identical validation-selection rules before any test set inference is executed.

---

## 11. Protocol Freeze and Governance Sign-off

* **Protocol Status:** LOCKED (v1.0)
* **Pre-Flight Verification Script:** `scripts/verify_10class_protocol_compliance.py`
* **Execution Trigger:** Active GPU training is forbidden until this document is reviewed and authorized by the Principal Investigator.
