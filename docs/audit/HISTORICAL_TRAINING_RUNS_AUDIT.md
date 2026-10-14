# Master Audit of Historical Training Runs & Experimental Benchmarks

**Document Identifier:** `docs/audit/HISTORICAL_TRAINING_RUNS_AUDIT.md`  
**Creation Date:** 2026-10-13  
**Status:** VERIFIED AUDIT REPORT  
**Scope:** Complete inventory and provenance audit of all historical 10-class, 4-class, and cleaned benchmark experiments.

---

## 1. Executive Summary & Audit Mandate

This document establishes the official experimental inventory of all training runs, model configurations, dataset splits, random seeds, and evaluation metrics produced across the lifetime of the *Classification of Eye Diseases from Color Fundus Images* project.

In strict adherence to scientific rigor:
1. **Preservation of Past Work:** No historical checkpoint, log, prediction array, or report is deleted or overwritten.
2. **Provenance & Split Auditing:** Every run is mapped to its exact dataset split, disclosing whether it was evaluated on uncleaned (leakage-vulnerable) partitions or on the audited, leakage-controlled benchmark (`data_clean/`).
3. **Test-Set Integrity Accounting:** All test-set evaluations are explicitly classified as either **Historical Exploratory** (previously evaluated during model exploration) or **Pristine Quarantined** (untouched).
4. **Hypothesis vs. Fact Discipline:** Architectural claims (e.g. that novel attention and fusion models will surpass published literature) are strictly classified as **scientific hypotheses to be empirically tested**, not established findings.

---

## 2. Experimental Era 1: Historical 10-Class Experiments (August–September 2026)

### 2.1 Dataset Partition & Provenance
* **Source Corpus:** Mendeley Data (Rashid et al., 2024; DOI: `10.17632/s9bfhswzjb.1`).
* **Original Partitioning:** 4,000 raw images artificially balanced to 400 images per class across 10 categories.
  - `data/train/`: 2,800 images (70%, 280/class)
  - `data/val/`: 600 images (15%, 60/class)
  - `data/test/`: 600 images (15%, 60/class)
* **Leakage & Quality Audit Status:** **CONTAMINATED / UNCLEANED.**
  - A subsequent forensic audit (October 2026) revealed that this raw dataset contained cross-split near-duplicates, burst captures, and synthetic augmentations.
  - Furthermore, all 10 classes included 400 images, which had been artificially expanded from unbalanced clinical cohorts (e.g., Pterygium only had 14 genuine clinical images, artificially expanded into hundreds).
* **Test Exposure Status:** **HISTORICAL EXPLORATORY.** The 600-image test set was evaluated across all models in Era 1 to guide development.

### 2.2 Historical 10-Class Benchmark Results (Held-Out Test Set, $N=600$)

These results are preserved in `paper/tables/table1_10class_benchmarks.tex`, `outputs/statistical_significance_results.json`, and `docs/SESSION_HANDOVER.md`:

| Experiment ID | Architecture | Input Res & Preprocessing | Optimization / Loss | Test Acc [95% CI] | Macro F1 [95% CI] | ROC-AUC [95% CI] | Cohen's $\kappa$ [95% CI] | Key Notes / Findings |
|---|---|---|---|:---:|:---:|:---:|:---:|---|
| **EXP-001** | EfficientNet-B0 | $224\times 224$, ImageNet | AdamW, CE (35% Train Split) | 83.38% | 83.11% | 0.9774 | 0.9115 | Initial project baseline. |
| **EXP-002** | BiomedCLIP (ViT-B/16) | $224\times 224$, PubMedBERT | AdamW, CE | 83.85% | 83.69% | 0.9802 | 0.9128 | Medical domain foundation weights. |
| **EXP-003** | BiomedCLIP + CBAM | $224\times 224$, Dual Attention | AdamW, CE | 84.23% | 84.09% | 0.9796 | 0.9128 | Channel + Spatial attention on optic disc. |
| **EXP-004** | Weighted Ensemble | $224\times 224$, 4-View TTA | Multi-model averaging | 85.85% | 85.72% | 0.9839 | 0.9263 | Peak on 35% train split. |
| **EXP-005** | EfficientNet-B3 | $384\times 384$, CLAHE | AdamW, CE (70% Train Split) | 90.17% | 90.03% | 0.9891 | 0.9628 | First model crossing 90% accuracy. |
| **EXP-006** | BiomedCLIP + CBAM | $224\times 224$, CLAHE | AdamW, CE (70% Train Split) | 87.83% | 87.83% | 0.9894 | 0.9447 | Glaucoma F1 surged to 63.93%. |
| **EXP-007** | Mega-Ensemble | EffNet-B3 (384) + BiomedCLIP | Multi-scale blending | 90.50% | 90.44% | 0.9923 | 0.9704 | Blended CNN spatial & ViT global tokens. |
| **EXP-008** | ConvNeXt-Small | $384\times 384$, CLAHE | AdamW, lr=5e-5, CE | 90.17% [87.8, 92.5] | 90.15% [88.0, 92.4] | 0.9915 [0.988, 0.994] | 0.8907 [0.865, 0.917] | Standalone SOTA; Glaucoma F1 reached 68.25%. |
| **EXP-010** | EfficientNet-B3 MS-TTA | $384\times 384$, 2-Scale TTA | MS-TTA [1.0, 1.15] | 91.00% [88.7, 93.2] | 90.97% [88.9, 93.0] | 0.9907 [0.987, 0.994] | 0.9000 [0.874, 0.924] | Multi-scale test-time augmentation. |
| **EXP-011** | ConvNeXt + EffNet MS-TTA | Dual 2-Scale TTA | Dual MS-TTA | 91.00% | 90.91% | 0.9921 | 0.9720 | Peak ROC-AUC (0.9921). |
| **EXP-023** | **Unified Quad Ensemble** | ConvNeXt + EffNet + Biomed + ResNet | Weighted Softmax Fusion | **91.83% [89.7, 93.8]** | **91.78% [89.8, 93.8]** | **0.9923 [0.989, 0.995]** | **0.9093 [0.885, 0.931]** | **Best All-Time Ensemble SOTA (49 errors / 600).** |
| **EXP-024** | *5-Model Oracle Bound* | Theoretical Upper Bound | Oracle Selection | *94.83%* | — | — | — | Theoretical ceiling of model diversity. |

*Statistical Significance:* McNemar's paired test (`paper/tables/table3_mcnemar_significance.tex`) confirmed that the Unified Quad Ensemble is statistically significantly superior to ConvNeXt-Small ($p = 0.0213$), BiomedCLIP-CBAM ($p = 0.0059$), ViT-Base-384 ($p = 0.0052$), and ResNet-50d ($p < 0.001$).

---

## 3. Experimental Era 2: Kaggle 4-Class Benchmark (September 2026)

### 3.1 Context & Dataset Partitioning
* **Target Objective:** Reproduce and outperform the published literature benchmark in Rahaf Alsohemi & Samia Dardouri (*Journal of Imaging*, Aug 2025; 11(8), 279), which reported 95.12% accuracy using EfficientNet-B3.
* **Target Dataset:** Kaggle `gunavenkatdoddi/eye-diseases-classification` (4,217 images; 4 classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal).
* **Partitions (Stratified, Seed 42):**
  - Train (70%): 2,949 images
  - Val (20%): 845 images
  - Test (10%): 423 images (Held-out)
* **Test Exposure Status:** **HISTORICAL EXPLORATORY.** Evaluated and fully documented.

### 3.2 4-Class Benchmark Results (Held-Out Test Set, $N=423$)

Preserved in `paper/tables/table2_4class_benchmarks.tex` and `docs/SESSION_HANDOVER.md`:

| Architecture / Model | Input Res & Setup | Test Acc [95% CI] | Macro F1 [95% CI] | Macro ROC-AUC [95% CI] | Cohen's $\kappa$ [95% CI] | Status vs. Alsohemi Paper (95.12%) |
|---|---|:---:|:---:|:---:|:---:|---|
| **MobileNetV2** (Alsohemi 2025 Ref) | Standard | 93.50% | — | — | — | Baseline literature |
| **Alsohemi & Dardouri (2025)** | EfficientNet-B3 | 95.12% | 95.10% | — | — | Published target paper |
| **Hybrid Feature Fusion** (Ref 6) | Multi-feature | 95.70% | — | — | — | Prior literature SOTA |
| ResNet-50d | $384\times 384$, CLAHE | 92.91% [90.3, 95.3] | 92.78% [90.0, 95.3] | 0.9866 [0.979, 0.993] | 0.9054 [0.870, 0.937] | Standalone backbone |
| EfficientNet-B3 | $384\times 384$, CLAHE | 92.91% [90.3, 95.3] | 92.73% [90.2, 95.2] | 0.9895 [0.983, 0.995] | 0.9054 [0.871, 0.937] | Re-implementation |
| DenseNet-121 | $224\times 224$ | 93.14% [90.8, 95.5] | 92.98% [90.6, 95.3] | 0.9883 [0.981, 0.994] | 0.9085 [0.876, 0.940] | Standalone backbone |
| ViT-Base-384 | $384\times 384$ | 93.14% [90.5, 95.5] | 93.03% [90.5, 95.3] | 0.9904 [0.985, 0.995] | 0.9086 [0.874, 0.940] | Standalone backbone |
| ConvNeXt-Small v2 | $384\times 384$ | 94.33% [92.0, 96.7] | 94.25% [91.8, 96.5] | 0.9928 [0.988, 0.997] | 0.9243 [0.893, 0.956] | Standalone backbone |
| BiomedCLIP + CBAM | $224\times 224$ | 94.56% [92.2, 96.5] | 94.49% [92.3, 96.5] | 0.9915 [0.986, 0.997] | 0.9275 [0.896, 0.953] | Highest standalone |
| **ResNet-Integrated Quad Ensemble** | **Biomed+ConvNeXt+ResNet+EffNet** | **95.74% [93.6, 97.6]** | **95.70% [93.6, 97.6]** | **0.9929 [0.988, 0.997]** | **0.9432 [0.915, 0.968]** | **✅ SURPASSES ALSOHEMI (+0.62%)** |

*Key Clinical Finding:* In the 4-class setting, Diabetic Retinopathy achieved **100.0% Sensitivity and 100.0% Specificity** across all ensemble iterations.

---

## 4. Experimental Era 3: Cleaned Benchmark Runs (October 2026)

Following the forensic DINOv2 audit that excised duplicate clusters and synthetic expansions, the dataset was frozen as `data_clean/` (3,747 images total: Train 2,622, Val 562, Test 563).

### 4.1 Run 3A: Clean 10-Class Baseline (`outputs/clean_baseline_efficientnet_b0/`)
* **Task:** 10-Class Benchmark (Track A) on `data_clean/`.
* **Architecture:** EfficientNet-B0 (ImageNet-1k pretrained, $224\times 224$).
* **Loss & Optimization:** Standard unweighted Cross-Entropy, AdamW (lr=3e-4, wd=1e-4), Cosine Annealing (20 epochs).
* **Random Seed:** 42 (Single seed).
* **Validation Performance ($N=562$, Peak Epoch 19):**
  - Accuracy: **87.01%** (489/562 correct)
  - Balanced Accuracy: **87.62%**
  - Macro-F1: **87.41%**
* **Test Set Status:** **HISTORICAL EXPLORATORY.** Evaluated on Oct 8 / Oct 10 on the 563 test images (`test_evaluation_reconciled.json`):
  - Test Accuracy: **87.21%** (491/563 correct)
  - Test Balanced Accuracy: **85.08%**
  - Test Macro-F1: **85.76%**
* **Ablation Offshoots (Validation Only):**
  - `outputs/clean_lr1e4_efficientnet_b0/`: Tested lr=1e-4 (Val Macro-F1: 86.82%).
  - `outputs/clean_wd5e4_efficientnet_b0/`: Tested wd=5e-4 (Val Macro-F1: 86.35%).
* **Saved Checkpoint:** `outputs/clean_baseline_efficientnet_b0/best_model.pth` (SHA-256: `bac42ea265c881d5447b58498c6a21210c2048d930d812f71fa9b1bdeb7c2edd`).

### 4.2 Run 3B: Clean 9-Class Posterior Fundus Baseline (`outputs/clean_9class_baseline_effnet/`)
* **Task:** 9-Class Posterior-Segment Benchmark (Track B) on `data_clean/` (Pterygium quarantined; $N=3,733$).
* **Architecture:** EfficientNet-B0 (ImageNet-1k pretrained, $224\times 224$, linear head $1280 \rightarrow 9$).
* **Loss & Optimization:** Standard unweighted Cross-Entropy, AdamW (lr=3e-4, wd=1e-4), Cosine Annealing (20 epochs).
* **Predeclared Seeds:** 42, 1337, 2026.
* **Validation Results ($N=560$ across 3 Seeds):**
  - Seed 42: Val Macro-F1 = **85.14%**, Balanced Acc = 85.42%, Acc = 87.32% (489/560).
  - Seed 1337: Val Macro-F1 = **85.69%**, Balanced Acc = 87.46%, Acc = 87.50% (490/560).
  - Seed 2026: Val Macro-F1 = **86.37%**, Balanced Acc = 86.93%, Acc = 88.04% (493/560).
  - **Aggregate Validation Performance:**
    $$\text{Macro-F1} = 85.73\% \pm 0.62\% \quad (\text{Range: } 85.14\% - 86.37\%)$$
    $$\text{Balanced Accuracy} = 86.60\% \pm 1.06\%$$
    $$\text{Overall Accuracy} = 87.62\% \pm 0.37\%$$
* **Test Set Status:** **PRISTINE QUARANTINED.**
  - `test_inferences_executed = 0`.
  - The 561-image test set has **NEVER been evaluated** for this 3-seed experiment, completely preserving test set integrity.
* **Saved Checkpoints:**
  - Seed 42: `outputs/clean_9class_baseline_effnet/seed_42/best_model.pth` (SHA-256: `e54f0f86fafaa95f7921a83b383f2b9928111383a68761e7f915b8b63abda38d`)
  - Seed 1337: `outputs/clean_9class_baseline_effnet/seed_1337/best_model.pth` (SHA-256: `887f3a16aeb029054e9a31f75837c409fd73b76c1b8ad2677d745af9928cbe41`)
  - Seed 2026: `outputs/clean_9class_baseline_effnet/seed_2026/best_model.pth` (SHA-256: `6fa94ffbe1f8877bc954fa16b0808b89814c1d6832dbff09ce24ee59bb13f3ae`)

---

## 5. Comparative Evaluation: Does the Project Need New Training?

Based on this audit, we evaluate whether running new baseline training is scientifically necessary:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING NECESSITY ASSESSMENT ACROSS TRACKS                      │
├────────────────────┬─────────────────────────────┬─────────────────────────────────────┤
│ Research Track     │ Current Checkpoint State    │ Scientific Recommendation           │
├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
│ TRACK B            │ Completed: 3 seeds trained  │ DO NOT RE-TRAIN. The 3-seed baseline│
│ (9-Class Fundus)   │ & validated (outputs/clean_ │ is already finished, verified, and  │
│                    │ 9class_baseline_effnet/)    │ test-quarantined. Re-running it     │
│                    │                             │ would waste compute without benefit.│
├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
│ TRACK A            │ Partially complete: Single  │ OPTIONAL / HYBRID. We have Seed 42  │
│ (10-Class Full)    │ seed (42) trained on        │ evaluated. If 3-seed variance is    │
│                    │ data_clean/. Test set was   │ required for Track A, train only    │
│                    │ evaluated historically.     │ seeds 1337 and 2026 on validation.  │
├────────────────────┼─────────────────────────────┼─────────────────────────────────────┤
│ NOVEL ARCHITECTURE │ Code implemented            │ PRIMARY TARGET. Evaluate proposed   │
│ (BiomedCLIP-CBAM-  │ (src/models/backbone.py);   │ architecture as controlled ablation │
│ Fusion & ConvNeXt) │ not yet trained on clean    │ against existing baselines.         │
│                    │ data_clean/.                │                                     │
└────────────────────┴─────────────────────────────┴─────────────────────────────────────┘
```

### 5.1 Key Scientific Clarifications
1. **No Blank Slate:** The project has already produced high-performing models on both the 10-class (Unified Quad Ensemble 91.83%) and 4-class (ResNet Quad Ensemble 95.74%) tasks. These benchmarks are preserved in paper manuscripts and audit logs.
2. **Exploratory vs. Confirmatory Test Reporting:** The 10-class test set ($N=563$ on `data_clean/`, and $N=600$ on `data/`) was previously evaluated. Under medical AI reporting standards (CLAIM 2024), its metrics must be explicitly disclosed as **historical exploratory findings** rather than pristine confirmatory results.
3. **Hypothesis vs. Demonstrated Fact:** The claim that our novel architecture (BiomedCLIP + CBAM Dual Attention + Multi-Scale Feature Fusion) will beat published models is an **active scientific hypothesis**. It will be demonstrated through controlled empirical comparison against our own verified baselines.
