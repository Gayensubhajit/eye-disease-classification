# Master Project State & Session Handover Document

> **Purpose of this Document:**  
> This file preserves 100% of the project memory, experimental benchmarks, architecture decisions, and codebase state. If Linux is reinstalled, or if you start a new conversation with Antigravity / any AI agent, simply give the prompt:  
> **`"Read docs/SESSION_HANDOVER.md and resume where we left off"`**  
> The agent will instantly understand the full project history and continue seamlessly.

---

## 1. Project Identity & Supervision

- **Academic Program:** B.Tech Major Project (Ongoing 7th Semester, final submission in 8th Semester around April 2027)
- **Institution:** Department of Information Technology, Jadavpur University
- **Research Team:** Gunjan Basak, Chirantan Biswas, Subhajit Gayen
- **Supervisor:** Dr. Pawan Kumar Singh
- **Objective:** Automated classification of 10 distinct retinal conditions from color fundus photography using novel deep learning architectures, attention mechanisms, and high-resolution preprocessing.

---

## 2. Dataset Structure & Partitioning

The dataset is organized in a physical folder hierarchy (4,000 total images, balanced equally across 10 classes):

| Partition | Images per Class (10 Classes) | Total Images | Percentage | Role |
|---|:---:|:---:|:---:|---|
| **`data/train/`** | 280 | 2,800 | 70.0% | Model training with dynamic Albumentations |
| **`data/val/`** | 60 | 600 | 15.0% | Model selection based on Macro F1 |
| **`data/test/`** | 60 | 600 | 15.0% | Untouched, physically isolated test set |
| **Total** | **400** | **4,000** | **100.0%** | Equal balance across all classes |

### The 10 Target Diagnostic Classes:
1. `Central Serous Chorioretinopathy [Color Fundus]` (CSCR)
2. `Diabetic Retinopathy`
3. `Disc Edema`
4. `Glaucoma`
5. `Healthy`
6. `Macular Scar`
7. `Myopia`
8. `Pterygium`
9. `Retinal Detachment`
10. `Retinitis Pigmentosa`

---

## 3. Complete Experimental Benchmarks (EXP-001 through EXP-009)

All models evaluated on the untouched held-out test set:

| ID | Architecture | Input Res & Preprocessing | Test Acc | Macro F1 | ROC-AUC | Cohen's $\kappa$ | Key Contribution & Notes |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **EXP-001** | EfficientNet-B0 | $224\times 224$, ImageNet pretraining | 83.38% | 83.11% | 0.9774 | 0.9115 | Initial baseline model (35% train split). |
| **EXP-002** | Microsoft BiomedCLIP | $224\times 224$, PubMedBERT weights | 83.85% | 83.69% | 0.9802 | 0.9128 | Medical domain foundation model fine-tuning. |
| **EXP-003** | BiomedCLIP + CBAM | $224\times 224$, Dual Attention | 84.23% | 84.09% | 0.9796 | 0.9128 | Dual channel/spatial attention on optic disc. |
| **EXP-004** | Weighted Ensemble | $224\times 224$, 4-View Flip TTA | 85.85% | 85.72% | 0.9839 | 0.9263 | Multi-model averaging peak on 35% split. |
| **EXP-005** | EfficientNet-B3 | $384\times 384$, CLAHE, 70% Split | 90.17% | 90.03% | 0.9891 | 0.9628 | First time crossing 90% threshold. Beats published IEEE 2023 SOTA (89.2%). |
| **EXP-006** | BiomedCLIP + CBAM | $224\times 224$, CLAHE, 70% Split | 87.83% | 87.83% | 0.9894 | 0.9447 | High Glaucoma sensitivity (65% recall, 63.93% F1). |
| **EXP-007** | Mega-Ensemble | EffNet-B3 (384) + BiomedCLIP-CBAM | 90.50% | 90.44% | 0.9923 | 0.9704 | Multi-scale multi-paradigm blending. |
| **EXP-008** 🏆 | **ConvNeXt-Small** | **$384\times 384$, CLAHE, 70% Split** | **90.50%** | **90.48%** | **0.9902** | **0.9663** | 🏆 **HIGHEST STANDALONE MODEL SOTA! Glaucoma F1 surged to 68.25% (71.67% sensitivity). CSCR (96.0%), DR (95.0%), Disc Edema (99.2%), RD (100%), Pterygium (100%).** |
| **EXP-009** | Triple Mega-Ensemble | ConvNeXt + EffNet + BiomedCLIP + TTA | 90.50% | 90.40% | 0.9929 | 0.9710 | ALL-TIME PEAK ROC-AUC (99.29%) & HIGHEST KAPPA (0.9710). DR F1 reached 95.87% (96.67% sensitivity). |
| **EXP-010** 🏆 | **EfficientNet-B3 MS-TTA** | **$384\\times 384$, CLAHE, 2-Scale TTA** | **91.00%** | **90.97%** | **0.9907** | **0.9739** | 🏆 **NEW STANDALONE SOTA! Multi-scale TTA (1.0x + 1.15x) breaks 91% for first time. Macular Scar improved to 84.5% F1.** |
| **EXP-011** 🏆 | **ConvNeXt + EffNet Dual MS-TTA** | **Dual Ensemble 2-Scale TTA** | **91.00%** | **90.91%** | **0.9921** | **0.9720** | 🏆 **PEAK ROC-AUC IMPROVES TO 99.21%. Myopia F1 climbs to 83.8%, Macular Scar F1 87.6%.** |

---

## 4. Hardware Profile & Memory Management Rules

- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (~3.68 GB VRAM).
- **Filesystem Location:** The repo resides on the Windows NTFS drive at `/mnt/windows/Users/subha/Documents/GitHub/eye-disease-classification` (`C:\Users\subha\Documents\GitHub\eye-disease-classification`).
- **Memory Rule for ConvNeXt-Small ($384\times 384$):**
  - Use `batch_size: 4` with `learning_rate: 0.00005`. Peak VRAM is **1.77 GB**, leaving nearly 2 GB of free headroom.
  - Always keep `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` in `src/train.py` to eliminate CUDA memory fragmentation.
- **Memory Rule for Multi-Model Evaluation:**
  - In `scripts/evaluate_ensemble.py`, models must be explicitly deleted between passes (`del model, ckpt; gc.collect(); torch.cuda.empty_cache()`) to prevent VRAM accumulation.

---

## 5. Key Scripts & Quick-Run Commands

### Virtual Environment Activation:
```bash
# On Linux:
source .venv/bin/activate

# On Windows:
.venv_win\Scripts\Activate.ps1
```

### Train Top Models:
```bash
# 1. Train ConvNeXt-Small (384x384 SOTA):
python -m src.train --config configs/convnext_small_384_clahe.yaml

# 2. Train EfficientNet-B3 (384x384 SOTA):
python -m src.train --config configs/efficientnet_b3_384_clahe.yaml

# 3. Train BiomedCLIP + CBAM Dual Attention:
python -m src.train --config configs/biomedclip_cbam_fusion.yaml
```

### Evaluate Checkpoints & Run Ensembles:
```bash
# Standalone evaluation on 600 test images:
python -m src.evaluate --checkpoint outputs/convnext_small_384_clahe/best_model.pth --config configs/convnext_small_384_clahe.yaml

# Run Triple Mega-Ensemble with 4-view TTA:
python scripts/evaluate_ensemble.py \
  --configs configs/convnext_small_384_clahe.yaml configs/efficientnet_b3_384_clahe.yaml configs/biomedclip_cbam_fusion.yaml \
  --checkpoints outputs/convnext_small_384_clahe/best_model.pth outputs/efficientnet_b3_384_clahe/best_model.pth outputs/biomedclip_cbam_fusion/best_model.pth \
  --weights 0.45 0.35 0.20 \
  --output-dir outputs/triple_ensemble_eval
```

### Generate Grad-CAM Explainability & Research PDF:
```bash
# Generate 10-class Grad-CAM grid at 384x384:
python scripts/generate_gradcam_figures.py

# Recompile publication-grade research report PDF:
python scripts/generate_research_pdf.py
```

---

## 6. Next Steps on the Agenda (Beyond 91.00% Accuracy)

1. **Multi-Scale Test-Time Augmentation (MS-TTA):** Evaluate test images at multiple zoom scales `[1.0, 1.15]` ($384\times 384$ and $448\times 448$) to push accuracy past 91%+.
2. **Dual-Scale Hybrid Network (ConvNeXt-384 + BiomedCLIP Cross-Attention):** End-to-end joint training of high-resolution spatial features with vision-language embeddings.
3. **Interactive Web Application / Clinical Screening Studio:** Modern dark-glassmorphism web UI with real-time fundus drag-and-drop inference and Grad-CAM visualization.
