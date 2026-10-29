# Research Review, Literature Matrix & Novel Architecture Specification

**Project**: Multi-Class Eye Disease Classification from Color Fundus Photographs  
**Scope**: Final-Year Capstone / Master's Dissertation Research Pivot  
**Date**: October 24, 2026  
**Document Purpose**: Literature gap synthesis, empirical motivation from Milestones V1–V7.1, mathematical formulation of proposed novel architecture, and rigorous experimental ablation protocol.

---

## 1. Executive Summary & The Professor's Mandate

The supervisory guidance establishes four non-negotiable research requirements that elevate this project from empirical model training to an original scientific contribution:

```
                  ┌──────────────────────────────────────────────┐
                  │          The Four Research Mandates          │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│  Requirement 1  │             │  Requirement 2  │             │  Requirement 3  │
│ Deep Literature │             │ Original Model  │             │ Controlled Fair │
│ Gap Synthesis   │             │   Formulation   │             │   Evaluation    │
└────────┬────────┘             └────────┬────────┘             └────────┬────────┘
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         ▼
                                ┌─────────────────┐
                                │  Requirement 4  │
                                │ Defensible Data │
                                │   & Validation  │
                                └─────────────────┘
```

1. **Requirement 1 — Grounding in Prior Work**: Identify exact architectures, datasets, evaluation paradigms, reported metrics, and fundamental limitations in published literature. Never compare raw accuracy numbers across different datasets or class counts.
2. **Requirement 2 — Theoretical Architectural Novelty**: Propose a bespoke deep-learning architecture with rigorous mathematical formulation, addressing an identified failure mode of prior models rather than combining generic off-the-shelf blocks.
3. **Requirement 3 — Fair, Controlled Empirical Proof**: Compare the proposed architecture against strong baselines under identical splits, identical input resolutions ($384 \times 384$), identical compute budgets (20 epochs), identical preprocessing (Crop Fundus + CLAHE), multi-seed stability audits ($N=3$), and systematic ablation studies.
4. **Requirement 4 — Epistemic Honesty & Methodological Defense**: Account for duplicate leakage, class imbalance, validation reuse, the modality discrepancy of Pterygium, and external dataset transferability without overconfident clinical claims.

---

## 2. Comprehensive Literature Review & Benchmark Matrix

| Dimension | Paper 1: Alsohemi & Dardouri (*J. Imaging*, 2025) | Paper 2: Worku et al. (*Sci. Reports*, 2026) | Paper 3: Sharmin et al. (*Data in Brief*, 2024) | Paper 4: Rieck et al. (*PLOS ONE*, 2025) | Paper 5: Hybrid DL (*IEEE Access*, 2025) | **Our Project (Current State)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Scope** | Transfer learning CNN baseline | Pretrained CNN comparative study | Dataset release & annotation | Domain-specific preprocessing | Generic CNN-Transformer hybrid | Novel Multi-Scale Anatomical Architecture |
| **Dataset Source** | Kaggle open fundus dataset | Univ. of Gondar Referral Hospital, Ethiopia | 2 Hospitals in Bangladesh (Topcon TRC-NW400 & NW8F) | 16,242-image expanded version of Sharmin et al. | Multi-source fundus collection | Cleaned Sharmin et al. ($N=3,747$) + RFMiD ($N=3,200$) |
| **Class Count** | **4 classes** (Cataract, DR, Glaucoma, Normal) | **4 classes** (DR, Normal, Glaucoma, AMD) | **10 categories** (DR, Glaucoma, Normal, MS, Myopia, CSCR, ODE, RD, RP, PTY) | **10 categories** (Same 10 classes) | Broad-coverage disease set | **10 classes** benchmark + **9-class** posterior fundus |
| **Imaging Modality** | Posterior fundus (diffuse cataract opacities) | Posterior fundus | **Mixed**: 9 posterior fundus + **1 anterior segment (Pterygium)** | Mixed (same 10 classes) | Posterior fundus | **Explicitly segregated**: 10-class vs 9-class posterior |
| **Architecture** | Pretrained EfficientNet-B3 | Pretrained DenseNet-201 (vs VGG, ResNet, Inception) | Dataset descriptive paper (no novel architecture) | ResNet-50 with Gaussian software lens | CNN backbone + standard Transformer encoder | Dual-Stream Multi-Scale Cross-Attention Network |
| **Resolution** | $224 \times 224$ | $225 \times 225$ | $1500 \times 1000$ to $2124 \times 2056$ | Resized / cropped | $224 \times 224$ | **$384 \times 384$** (preserves microvascular lesions) |
| **Reported Metric** | Acc: **95.12%**, F1: 95.00% | Test Acc: **92.78%**, F1: 0.93 | Descriptive statistics & label distributions | Balanced Acc: **82.52%** (vs 79.40% baseline) | Macro-F1: ~87–89% | Val Acc: **90.75% ± 0.18%**, Macro-F1: **90.19% ± 0.49%** |
| **Key Limitation 1** | Trivial 4-class setting; cataract easily separable | Single center; no external validation | Severe class imbalance (Pterygium $N=17$, DR $N=1,509$) | Static hand-crafted Gaussian filter; no learned fusion | Generic feature concatenation without anatomical guidance | Validation reuse; patient-level leakage unverified |
| **Key Limitation 2** | No external camera domain transfer | AMD class heavily underrepresented | **Modality mixing**: Pterygium is anterior segment | Data volume inflated to 16k without leakage audit | No lesion-specific or anatomical cup/disc priors | RFMiD transfer is exploratory ranking only |
| **Relevance to Us** | Reference baseline (our EffNet-B3 gets 86.47% on 10-class) | Explores class weighting; proves DenseNet baseline | **Our primary source dataset**; mandates leakage control | Proves domain filtering works; bars us from simple filtering | Proves generic CNN+ViT is not novel; bars simple hybrids | Establishes baselines to beat with genuine novelty |

---

## 3. The Pterygium Scientific Dilemma: Modality Mixing vs. Retinal Disease

A critical scientific insight emerges from dissecting Sharmin et al. (*Data in Brief*, 2024):

> [!CAUTION]
> **Modality Incoherence in the 10-Class Benchmark**:
> - 9 classes (*Diabetic Retinopathy, Glaucoma, Healthy, Macular Scar, Myopia, CSCR, Disc Edema, Retinal Detachment, Retinitis Pigmentosa*) represent **posterior pole retinal fundus photography**.
> - 1 class (**Pterygium**, $N=17$ original, $N=2$ validation) represents **anterior segment external slit-lamp / corneal photography** (fibrovascular conjunctival overgrowth across the cornea).
> - **The Confound**: A standard CNN or Transformer does not need to extract retinal disease features to classify Pterygium. It can achieve 100% precision and recall by simply detecting that the image is an anterior-segment photo (cornea, sclera, iris, eyelids) rather than an orange-red retinal fundus with an optic disc and vascular tree!
> - In Milestone V7.1, Swin-Base achieved **100.0% ± 0.0% recall** on Pterygium across all three seeds. This high metric reflects modality separation, not retinal pathology learning.

### The Scientific Mandate:
In our paper and final thesis, we must present:
1. **The Official 10-Class Benchmark**: Evaluated for full reproducibility against prior published literature (Sharmin et al., Rieck et al., Rashid et al.).
2. **The Purified 9-Class Posterior Fundus Benchmark**: Conducted by excluding Pterygium ($N=3,730$ clean samples). This eliminates modality bias and isolates true retinal diagnostic performance.
3. Explicit disclosure that a classifier cannot claim to be a "unified retinal disease diagnostic tool" if 10% of its class space is driven by anterior segment modality artifacts.

---

## 4. Empirical Motivation from Milestones V1–V7.1: The Three Error Archetypes

Our extensive multi-seed stability audits (Seeds 42, 1337, and 2026 on $N=562$ validation samples) revealed that model errors are not random noise. They cluster into three distinct, persistent clinical archetypes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     The Three Clinical Error Archetypes                     │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Archetype 1: Local Lesion │ Archetype 2: Anatomical  │ Archetype 3: Macular  │
│      vs. Normal Fundus   │      Ratio Ambiguity     │   Morphological Shift │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ • Microvascular leakage  │ • Optic cup-to-disc ratio│ • Subretinal fluid vs │
│   (DR microaneurysms)    │ • Peripapillary atrophy  │   fibrotic scarring   │
│ • Borderline Healthy/DR  │ • Glaucoma vs Normal     │ • CSCR vs Macular Scar│
│                          │ • Myopia vs Glaucoma     │   vs Exudative DR     │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

### 4.1 Quantitative Evidence of Persistent Errors
In Milestone V7.1, we proved from aligned per-image predictions:
- **31 images** are misclassified by **all three independent training runs** (Seeds 42, 1337, 2026).
- **10 of 31** shared errors are **Glaucoma $\to$ Healthy** (subtle optic cup enlargement missed by standard global pooling).
- **5 of 31** shared errors are **Healthy $\to$ Glaucoma** (physiological large cupping misclassified as glaucomatous neuropathy).
- **5 of 31** shared errors are **Myopia $\to$ Glaucoma / Normal** (temporal crescent peripapillary atrophy confounding optic disc margins).
- **6 of 31** shared errors are **Macular Scar $\to$ CSCR / DR** (fibrous cicatricial tissue confused with neurosensory fluid or hard exudates).
- **2 of 2** validation CSCR errors are shared across all seeds (confused with DR and Macular Scar).

### 4.2 The Identified Architectural Failure Mode of Existing Backbones
1. **Standard ViTs / Swin Transformers**: Excel at long-range dependencies and global contextual contrast (yielding 98% recall on Diabetic Retinopathy), but their fixed window self-attention ($12 \times 12$ patches of size $4 \times 4$) dilutes fine local vascular signals and optic nerve head boundary gradients.
2. **Standard CNNs (ConvNeXt, ResNet, EfficientNet)**: Preserve local spatial features through inductive bias (ConvNeXt achieved 90.18% Macro-F1), but their progressive spatial downsampling loses fine-grained lesion details before reaching the classification head, and they lack cross-regional anatomical reasoning between the optic disc and the macula.
3. **Generic Hybrids (IEEE Access 2025, Rieck et al. PLOS ONE 2025)**: Simple sequential CNN-Transformer stacks or naive feature concatenation combine representations blindly, without explicitly modeling the anatomical relationship between the **Optic Nerve Head (Glaucoma, Disc Edema)**, the **Fovea/Macula (CSCR, Macular Scar)**, and the **Microvasculature (DR, RP)**.

---

## 5. Proposed Novel Architecture: **Retina-DualNet** (Dual-Stream Anatomical Cross-Attention Network)

To fulfill the professor's mandate for a **complex novel deep-learning model**, we propose **Retina-DualNet** (Dual-Stream Anatomical Cross-Attention Network with Disease-Confusion Loss).

### 5.1 Architectural Philosophy
Rather than generic concatenation, Retina-DualNet models human ophthalmologist diagnostic workflow:
1. **Stream 1 (Local Microvascular Detail Branch)**: High-resolution convolutional path focused on high-frequency textural cues (microaneurysms, hemorrhages, bone-spicule pigmentation, subretinal fluid margins).
2. **Stream 2 (Global Anatomical Context Branch)**: Hierarchical shifted-window self-attention path focused on structural geometry (cup-to-disc ratio, macular symmetry, retinal detachment borders, arcade vessel tortuosity).
3. **Anatomical Cross-Attention Bridge (ACAB)**: A learned, bi-directional cross-attention module where global anatomical tokens query local feature maps to dynamically spotlight pathology-relevant regions.
4. **Disease-Confusion-Aware Margin Loss (DCML)**: A regularized loss function that dynamically enforces wider angular margins between historically confused disease pairs (Glaucoma $\leftrightarrow$ Healthy, Myopia $\leftrightarrow$ Glaucoma, CSCR $\leftrightarrow$ Macular Scar).

```
                              RETINA-DUALNET ARCHITECTURE
                              
                              Input Image: 384x384x3 (CLAHE)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         Stream 1: Local Detail                          Stream 2: Global Context
        ConvNeXt High-Res Stem                           Swin Hierarchical Stem
         (7x7 DW-Conv, Stage 1-3)                       (Shifted Window Multi-Head)
                    │                                               │
                    ▼ [B, 512, 24, 24]                              ▼ [B, 512, 12, 12]
                    │                                               │
                    └───────────────┬───────────────────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ Anatomical Cross-Attention Bridge     │
                │ Q = Global Tokens (Optic/Macula)      │
                │ K, V = Local Feature Map (Lesions)    │
                │ Multi-Head Cross-Attention (8 heads)  │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼ [B, 768]
                ┌───────────────────────────────────────┐
                │ Confusion-Aware Feature Fusion (CAFF) │
                │ Gate g = sigmoid(W_g * [F_local, F_glob])│
                │ F_fused = g * F_local + (1-g) * F_glob│
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ Classification Head + DCML Loss       │
                │ L = L_CE + lambda * L_Pairwise_Margin │
                │ Enforces separation:                  │
                │ Glaucoma <-> Normal                   │
                │ Myopia <-> Glaucoma                   │
                │ CSCR <-> Macular Scar                 │
                └───────────────────────────────────────┘
```

---

### 5.2 Mathematical Formulation

#### 5.2.1 Multi-Scale Feature Extraction
Given an input fundus image $X \in \mathbb{R}^{H \times W \times 3}$ ($H=W=384$):
- **Stream 1 Feature Map**: $\mathbf{F}_{\text{local}} = \Phi_{\text{CNN}}(X) \in \mathbb{R}^{C_1 \times H_1 \times W_1}$, where $C_1=512, H_1=W_1=24$.
- **Stream 2 Feature Map**: $\mathbf{F}_{\text{global}} = \Psi_{\text{Swin}}(X) \in \mathbb{R}^{C_2 \times H_2 \times W_2}$, where $C_2=512, H_2=W_2=12$.

#### 5.2.2 Anatomical Cross-Attention Bridge (ACAB)
We project global context features to Query tokens and local detail features to Key and Value tokens:
$$\mathbf{Q} = \mathbf{F}_{\text{global}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{F}_{\text{local}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{F}_{\text{local}} \mathbf{W}_V$$
where $\mathbf{W}_Q \in \mathbb{R}^{C_2 \times d_k}$, $\mathbf{W}_K \in \mathbb{R}^{C_1 \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{C_1 \times d_v}$.

The cross-attention feature map is computed across $h$ attention heads:
$$\mathbf{A}_{\text{cross}} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}_{\text{rel}}\right) \mathbf{V}$$
where $\mathbf{B}_{\text{rel}}$ is a learnable 2D relative position bias preserving anatomical retinal spatial coordinates.

#### 5.2.3 Confusion-Aware Margin Loss (DCML)
To penalize chronic confusion pairs without distorting the global feature manifold, we define the pairwise angular margin penalty:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CrossEntropy}} + \lambda \sum_{(c_i, c_j) \in \mathcal{P}_{\text{confused}}} \max\left(0, m - \left\|\mathbf{z}_{c_i} - \mathbf{z}_{c_j}\right\|_2^2\right)$$
where $\mathcal{P}_{\text{confused}} = \{(\text{Glaucoma}, \text{Healthy}), (\text{Myopia}, \text{Glaucoma}), (\text{CSCR}, \text{Macular Scar})\}$, $m > 0$ is the minimum inter-class margin hyperparameter, and $\mathbf{z}_c$ represents the normalized class prototype representation.

---

## 6. Rigorous Experimental & Ablation Protocol

To fulfill **Requirement 3 (Demonstrate Improvement)** and **Requirement 4 (Defensible Evidence)**:

### 6.1 Baseline Battery Under Identical Protocol
All models evaluated under:
- Resolution: $384 \times 384$
- Preprocessing: Crop Fundus Area + CLAHE
- Optimizer: AdamW, $\text{lr}=3 \times 10^{-5}$, cosine schedule, 20 epochs
- Compute budget: 2 micro-batches $\times$ 2 accumulation steps
- Multi-seed validation: Seeds 42, 1337, and 2026

Baselines:
1. **ResNet-50d** (Classic CNN baseline from Rieck et al. / Rashid et al.)
2. **DenseNet-201** (Scientific Reports 2026 champion baseline)
3. **EfficientNet-B3** (Alsohemi & Dardouri 2025 baseline)
4. **ConvNeXt-Small** (Our strongest standalone CNN baseline)
5. **Swin-Base** (Our strongest standalone Transformer baseline)
6. **Proposed Retina-DualNet** (Novel dual-stream cross-attention model)

### 6.2 Ablation Study Matrix
To demonstrate that any observed improvement is caused by our novel design rather than extra parameters:
- **Ablation 1 (Parameter-Matched Concatenation)**: Simple concatenation of ConvNeXt + Swin features without ACAB or DCML.
- **Ablation 2 (No Anatomical Bridge)**: Dual streams with standard addition fusion (no cross-attention).
- **Ablation 3 (No DCML Margin Loss)**: Dual streams with ACAB cross-attention, trained under standard Cross-Entropy alone.
- **Ablation 4 (Full Retina-DualNet)**: Dual streams + ACAB cross-attention + DCML margin loss.

### 6.3 Dual Evaluation: 10-Class vs. 9-Class Posterior Fundus
Every model and ablation will be scored on:
- **Cohort A (Official 10-Class Benchmark)**: $N=562$ validation images.
- **Cohort B (Purified 9-Class Posterior Fundus)**: $N=560$ validation images (excluding Pterygium to eliminate anterior-segment modality bias).

---

## 7. Next Actionable Milestone Roadmap

```
Milestone V7.1 (Completed)
├── Evidence Reconciliation of Swin-Base stability runs
├── Aligned per-image validation predictions saved
└── Pretrained weight hashes and best-epoch discrepancies verified
         │
         ▼
Milestone V8: ConvNeXt-Small Multi-Seed Stability Audit (Immediate Next)
├── Train Seeds 1337 & 2026 under identical frozen protocol
├── Directly compare ConvNeXt stability (Mean ± SD) against Swin-Base
└── Confirm baseline performance bounds before implementing novel model
         │
         ▼
Milestone V9: Retina-DualNet Architecture Specification & Implementation
├── Implement ACAB module and DCML loss in src/models/
├── Parameter parity verification against single-backbone baselines
└── Offline unit testing on GPU VRAM budget (<3.5 GB)
         │
         ▼
Milestone V10: Fair Ablation Battery & Multi-Seed Benchmarking
├── 10-class vs 9-class posterior fundus evaluation
├── Statistical significance testing (paired bootstrap & McNemar test)
└── Cross-dataset exploratory ranking on RFMiD cohort (N=3,200)
```

This roadmap moves our project from empirical experimentation to a publication-ready dissertation contribution that satisfies your professor's criteria.
