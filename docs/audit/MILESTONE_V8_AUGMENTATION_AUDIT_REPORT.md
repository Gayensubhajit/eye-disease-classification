# Milestone V8: Clean Benchmark Reconciliation, Literature Audit & Augmentation Study

**Date**: October 24, 2026  
**Task**: Milestone V8 — Clean Benchmark Reconciliation and Controlled Augmentation Audit  
**Scope**: Systematic literature audit (Quantum Preprint, QICAF-Net 2026, IEEE Access 2025, HF 16k Benchmark), controlled 3-seed augmentation experiment on EfficientNet-B0, and Novelty Gate Design Review  
**Deliverables Directory**: [`outputs/clean_10class_effnet_b0_augmentation_audit/`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/outputs/clean_10class_effnet_b0_augmentation_audit/)  

---

## 1. Executive Summary & Audit Overview

Milestone V8 implements the supervisory guidance to **pause premature architecture implementation and strengthen the empirical evidence base**. Recent literature identification—specifically the quantum-enhanced EfficientNet preprint, the 2026 Frontiers QICAF-Net paper, and a 2026 Hugging Face benchmark—demands that our project:
1. **Audit the Quantum Preprint Claims**: Dissect the reported 93.86% validation accuracy, isolate the empirical role of data augmentation, and evaluate split reproducibility.
2. **Audit Novelty Boundaries**: Acknowledge that generic two-branch CNN/Transformer cross-attention architectures already exist in published literature (IEEE Access 2025, Frontiers 2026), raising the bar for what constitutes a defensible architectural contribution.
3. **Execute a Controlled Augmentation Experiment**: Measure the exact empirical impact of stochastic training augmentation versus deterministic preprocessing on EfficientNet-B0 using our verified clean split across three independent seeds (Seeds 42, 1337, and 2026).
4. **Pass Through the Novelty Gate**: Systematically address the 7 design review questions before writing a single line of novel model code.

---

## 2. Phase A: Repository and Evidence Audit

### 2.1 Clean Dataset Split Manifest & Counts
The deduplicated clean dataset (`data_clean/`) contains exactly **3,747 fundus images** across 10 classes, verified against cross-split exact and near-duplicate leakage:
- **Training Partition** (`data_clean/train`): **2,622 images** (70.0%)
- **Validation Partition** (`data_clean/val`): **562 images** (15.0%)
- **Test Partition** (`data_clean/test`): **563 images** (15.0%, strictly quarantined)

### 2.2 Leakage Bounds & Augmentation Execution
> Training augmentation is performed on the fly after partitioning. Validation transformations are deterministic. Duplicate and near-duplicate audits have been performed, but patient-level independence cannot be verified because reliable patient identifiers are unavailable.

- **Strict Validation Quarantine**: During evaluation, a validation-only DataLoader (`FundusDataset(base_cfg['data']['val_dir'])`) is constructed directly. `data_clean/test` is **never instantiated, loaded, or iterated over**.

### 2.3 Verified Baseline Checkpoints on Clean Split

| Model Architecture | Resolution | Preprocessing | Best Epoch | Val Accuracy | Val Macro-F1 | Checkpoint SHA-256 | Source Directory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **EfficientNet-B0** | $224 \times 224$ | Crop Fundus (No CLAHE) | 19 | 87.01% (489/562) | 87.41% | `bac42ea265c881d5447b58498c6a21210c2048d930d812f71fa9b1bdeb7c2edd` | `clean_baseline_efficientnet_b0` |
| **ConvNeXt-Small** | $384 \times 384$ | Crop Fundus + CLAHE | 16 | 89.50% (503/562) | 90.18% | `238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d` | `clean_10class_convnext_small_384_clean` |
| **Swin-Base (Seed 42)** | $384 \times 384$ | Crop Fundus + CLAHE | 13 | 90.93% (511/562) | 89.68% | `6d118211b29368d71e49ad1dbb5e439ba7f798e416bb4fa32a963e3eac8fbef2` | `clean_10class_swin_base_384_clean` |
| **Swin-Base (3-Seed Mean)** | $384 \times 384$ | Crop Fundus + CLAHE | — | **90.75% ± 0.18%** | **90.19% ± 0.49%** | Multi-Seed Verified | `clean_10class_swin_base_3seed_stability` |

> [!IMPORTANT]
> **Separation of Experimental Cohorts**: The new EfficientNet-B0 experiments use $224 \times 224$ inputs, no CLAHE, a 20-epoch budget, and a learning rate of $3 \times 10^{-4}$. Our previous ConvNeXt-Small and Swin-Base results used $384 \times 384$ inputs with CLAHE. The new no-augmentation and augmentation arms are a controlled comparison with each other. Do not directly interpret a change against the historical ConvNeXt or Swin scores as an augmentation-only effect.

### 2.4 Critical Literature Audit & Unresolved Methodologies

#### 1. The Quantum Preprint Audit
- **Citation**: *Quantum-Enhanced EfficientNet-B0 for Multi-Class Retinal Disease Classification on Fundus Images* (Preprint on Research Square / ResearchGate, DOI: 10.21203/rs.3.rs-5079637/v1).
- **Platform Clarification**: ResearchGate and Research Square host the **identical underlying preprint text and figures**.
- **Reported Performance Breakdown**:
  - Classical EfficientNet-B0, non-augmented: **75.61%** validation accuracy
  - Quantum-Enhanced EfficientNet-B0, non-augmented: **77.45%** validation accuracy
  - Classical EfficientNet-B0, augmented: **86.37%** validation accuracy (+10.76 percentage points over original data)
  - Quantum-Enhanced EfficientNet-B0, augmented: **93.86%** validation accuracy
- **Supervisory Correction & Rigorous Interpretation**:
  > The quantum preprint reports 86.37% validation accuracy for augmented classical EfficientNet-B0 versus 75.61% for the original-data experiment, an increase of 10.76 percentage points. It reports 93.86% for the augmented quantum-enhanced model. Those are the paper's reported results. Without a verified source-image split manifest and enough methodological detail to rule out partition contamination, we should describe the increase as associated with the augmented-data experiment, not proof that augmentation alone explains the entire gain.
- **Methodological Vulnerabilities**:
  1. **Validation Score Only**: 93.86% is a validation score, not an independent test set score.
  2. **No Image-Level Split Manifest**: The preprint does not provide an image-level split manifest. An 80:10:10 partition on an augmented cohort risks duplicate copies of identical underlying eye fundus images appearing across train and validation folds.
  3. **Simulated Classical Circuit**: The quantum module is a classical simulation of parameterized rotation gates ($R_X, R_Y, R_Z$) executed on classical GPUs via PennyLane, providing no quantum speedup.

#### 2. The 2026 Frontiers QICAF-Net Paper
- **Citation**: *Quantum-inspired cross-attention fusion network for multi-class eye disease classification* (Frontiers in Medicine, October 2026).
- **Architecture & Dataset**: Evaluates Swin Transformer + EfficientNet with quantum-inspired cross-attention on a 4,217-image balanced 4-class retinal dataset (not our 10-class task).
- **Critical Empirical Finding**: Its reported ablations found **no statistically significant improvement** from its quantum-inspired components over strong classical fusion baselines.
- **Methodological Implication**: Proves that a generic CNN-Transformer dual-branch cross-attention model is **already published**. Proposing a generic ConvNeXt + Swin concatenation or standard cross-attention network would represent a weak, derivative novelty claim.

#### 3. 2026 Hugging Face 16k Benchmark
- **Artifact**: Self-reported benchmark artifact on Hugging Face for the same underlying 10-class dataset (expanded 16,242 images).
- **Reported Results**: Reports **90.18% accuracy for Inception-v3** across 5 folds and **87.00% for Swin-Base** using a perceptual-hash-based, group-aware splitting procedure.
- **Status & Verification**: Self-reported benchmark artifact (not a peer-reviewed paper); serves as an informative literature reference point on the expanded dataset that requires code and split manifest verification before relying on it.

---

## 3. Phase B & C: Controlled Augmentation Experiment

To measure the empirical role of data augmentation under strict control, we executed a multi-seed comparison on **EfficientNet-B0** ($224 \times 224$, 20 epochs, AdamW lr=3e-4, cosine scheduler, batch size 16):
- **Arm A (No Augmentation)**: Purely deterministic training preprocessing (`Resize -> Normalize -> ToTensorV2`).
- **Arm B (Moderate Augmentation)**: Stochastic training augmentation (`Resize -> HFlip p=0.5 -> Affine p=0.5 -> ColorJitter p=0.3 -> Normalize -> ToTensorV2`).
- Both arms evaluated on identical `data_clean/val` ($N=562$) with deterministic validation preprocessing.

### 3.1 Headline Metric Comparison Across 3 Seeds ($N=3$)

| Metric | Arm A: No Augmentation (Mean ± SD) | Arm B: Moderate Augmentation (Mean ± SD) | Absolute Delta (Arm B - Arm A) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 86.36% ± 0.45% | 87.43% ± 0.91% | **+1.07 percentage points** |
| **Macro-F1** | 86.14% ± 0.36% | 86.93% ± 0.38% | **+0.79 percentage points** |
| **Balanced Accuracy** | 85.31% ± 1.19% | 87.41% ± 0.45% | **+2.10 percentage points** |
| **Weighted F1** | 86.32% ± 0.42% | 87.37% ± 0.93% | **+1.06 percentage points** |
| **Validation Loss** | 1.0662 ± 0.0751 | 0.8012 ± 0.0693 | **-0.2650** |

### 3.2 Individual Seed-Level Results

Before relying on aggregate statistics, each run is verified individually across its 20-epoch trajectory:

| Experiment Arm | Seed | Best Epoch | Val Loss | Val Accuracy | Macro-F1 | Balanced Acc | Weighted F1 | Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Arm A (No Aug) | **42** | 20 | 1.1347 | 86.30% (485/562) | 85.75% | 84.71% | 86.20% | `7ddee363591fb3bc...` |
| Arm A (No Aug) | **1337** | 9 | 0.9859 | 86.83% (488/562) | 86.47% | 86.68% | 86.78% | `4fc959cee758fa96...` |
| Arm A (No Aug) | **2026** | 9 | 1.0780 | 85.94% (483/562) | 86.19% | 84.53% | 85.96% | `4cbccedfe617b26e...` |
| Arm B (Aug) | **42** | 18 | 0.8055 | 87.19% (490/562) | 86.94% | 87.73% | 87.10% | `33bac50145d39fec...` |
| Arm B (Aug) | **1337** | 12 | 0.7298 | 88.43% (497/562) | 87.31% | 87.60% | 88.41% | `c9df27db5028e37e...` |
| Arm B (Aug) | **2026** | 20 | 0.8682 | 86.65% (487/562) | 86.54% | 86.89% | 86.61% | `b7a7c117753b2a1b...` |

### 3.3 Per-Class Recall & Minority Class Breakdown

Overall accuracy can obscure shifts in minority-class sensitivity. All supports and recall metrics below are derived directly from the experimental evaluation data:

| Class Name | Validation Support ($N$) | Arm A: No Aug (Mean ± SD) | Arm B: Moderate Aug (Mean ± SD) | Recall Delta (Arm B - Arm A) |
| :--- | :---: | :---: | :---: | :---: |
| **Central Serous Chorioretinopathy [Color Fundus]** | 11 | 72.73% ± 9.09% | 87.88% ± 5.25% | **+15.15 pp** |
| **Diabetic Retinopathy** | 191 | 96.68% ± 0.80% | 95.81% ± 0.91% | **-0.87 pp** |
| **Disc Edema** | 15 | 97.78% ± 3.85% | 91.11% ± 3.85% | **-6.67 pp** |
| **Glaucoma** | 124 | 78.49% ± 4.66% | 80.91% ± 6.11% | **+2.42 pp** |
| **Healthy** | 105 | 87.30% ± 4.70% | 89.84% ± 1.45% | **+2.54 pp** |
| **Macular Scar** | 47 | 73.05% ± 5.35% | 73.76% ± 1.23% | **+0.71 pp** |
| **Myopia** | 37 | 70.27% ± 2.70% | 70.27% ± 2.70% | **+0.00 pp** |
| **Pterygium** | 2 | 100.00% ± 0.00% | 100.00% ± 0.00% | **+0.00 pp** |
| **Retinal Detachment** | 13 | 84.62% ± 7.69% | 92.31% ± 0.00% | **+7.69 pp** |
| **Retinitis Pigmentosa** | 17 | 92.16% ± 3.40% | 92.16% ± 3.40% | **+0.00 pp** |

#### Minority Class Sensitivity Analysis:
Minority classes are examined individually to assess whether stochastic augmentation disproportionately benefits or harms lower-frequency retinal conditions (supports derived from audited validation partition):

- **Pterygium** ($N=2$ val): Arm A: 100.00% ± 0.00% | Arm B: 100.00% ± 0.00% | Delta: +0.00 percentage points. Note: Pterygium is an anterior segment photograph rather than a posterior fundus image. With $N=2$, sample size is too limited to draw definitive conclusions about generalization.
- **Central Serous Chorioretinopathy [Color Fundus]** ($N=11$ val): Arm A: 72.73% ± 9.09% | Arm B: 87.88% ± 5.25% | Delta: +15.15 percentage points. Presents subtle macular neurosensory detachment fluid.
- **Retinal Detachment** ($N=13$ val): Arm A: 84.62% ± 7.69% | Arm B: 92.31% ± 0.00% | Delta: +7.69 percentage points. Features subretinal fluid accumulation and retinal folds.
- **Disc Edema** ($N=15$ val): Arm A: 97.78% ± 3.85% | Arm B: 91.11% ± 3.85% | Delta: -6.67 percentage points. Characterized by optic disc elevation and margin blurring.
- **Retinitis Pigmentosa** ($N=17$ val): Arm A: 92.16% ± 3.40% | Arm B: 92.16% ± 3.40% | Delta: +0.00 percentage points. Presents peripheral bone-spicule pigmentation and arteriolar attenuation.
- **Myopia** ($N=37$ val): Arm A: 70.27% ± 2.70% | Arm B: 70.27% ± 2.70% | Delta: +0.00 percentage points. Features peripapillary crescents and chorioretinal atrophy.
- **Macular Scar** ($N=47$ val): Arm A: 73.05% ± 5.35% | Arm B: 73.76% ± 1.23% | Delta: +0.71 percentage points. Features central foveal fibrous tissue formation.

---

## 4. Phase D: Novelty Gate Design Review

Before implementing any new architecture, the research gate requires formal answers to seven critical architectural questions:

### Question 1: Which shortcomings are documented by our own aligned validation predictions?
- In Milestone V7.1, we proved that **31 error images are persistently misclassified across all three seeds**.
- Explanations such as subtle optic-disc changes or specific retinal lesion morphology remain hypotheses until reviewed by a qualified ophthalmologist. A confusion matrix alone cannot establish the cause of a model's error.
- The current observational hypotheses for the 31 persistent errors cluster into three categories:
  1. **Optic Nerve Head Boundary Ambiguity** (15 errors in Glaucoma $\leftrightarrow$ Healthy): Model difficulty distinguishing subtle neuroretinal rim thinning from physiological cupping.
  2. **Peripapillary Crescent Confusion** (5 errors in Myopia $\leftrightarrow$ Glaucoma): Myopic chorioretinal atrophy confused with glaucomatous cupping.
  3. **Macular Lesion Texture vs. Fluid** (8 errors in CSCR $\leftrightarrow$ Macular Scar / DR): Inability to separate neurosensory detachment fluid from fibrous scars or exudates.

### Question 2: Which similar architectural mechanisms already exist in published literature?
- **2025 IEEE Access**: Proposes generic dual-branch CNN + Vision Transformer feature fusion for broad-spectrum fundus classification.
- **2026 Frontiers QICAF-Net**: Proposes Swin Transformer + EfficientNet with cross-attention fusion on four-class fundus images.
- **Strategic Directive**: We **cannot** simply concatenate ConvNeXt and Swin features or apply generic cross-attention. That design is already published and lacks novelty.

### Question 3: Does any proposed module have a precise mathematical definition and testable hypothesis?
- Yes. Future proposals must define explicit operations (e.g. Anatomical Coordinate Attention with learned optic/macula spatial priors, or Confusion-Penalizing Pairwise Loss) that can be isolated and mathematically ablated.

### Question 4: Does the design actually use anatomical location information, or merely generic spatial tokens?
- Existing models (Swin, ConvNeXt, ViT) treat fundus images as generic ImageNet patches. A biologically valid ophthalmological model must incorporate **polar or anatomical coordinate priors** centered on the optic nerve head and fovea.

### Question 5: Can each proposed component be evaluated through an ablation?
- Yes. Any novel mechanism must be tested against: (1) Backbone alone, (2) Naive feature concatenation, (3) Generic cross-attention, and (4) Proposed anatomical mechanism.

### Question 6: Is the model feasible on our 4 GB RTX 3050 GPU?
- Running two large 384-resolution backbones (ConvNeXt-Small + Swin-Base) simultaneously would draw ~5.8 GB VRAM during backward pass, exceeding our 4 GB limit.
- A feasible architecture must use lightweight stems (e.g. ConvNeXt-Femto/Atto or MobileNetV4) or single-backbone anatomical modulation heads.

### Question 7: How will we evaluate the 10-class task and 9-class posterior-fundus task separately?
- We will maintain strict dual reporting: official 10-class benchmark ($N=562$) for literature comparability, and 9-class posterior fundus benchmark ($N=560$, excluding Pterygium) to eliminate anterior-segment modality bias.

---

## 5. Scientifically Justified Next Steps

1. **Complete ConvNeXt-Small Multi-Seed Stability (Milestone V8.1)**: Run Seeds 1337 and 2026 for ConvNeXt-Small to finalize our comparative baseline scoreboard against Swin-Base.
2. **Design an Anatomically-Grounded Model (Milestone V9)**: Replace generic CNN/Transformer fusion with an anatomically-guided coordinate cross-attention mechanism tailored specifically to resolve the 31 persistent errors.
3. **Submit Official Supervisor Update**: Post project progress strictly to the professor's designated email thread, documenting the augmentation audit findings and literature reconciliations.
