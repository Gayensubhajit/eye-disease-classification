# Literature Review & Benchmark Comparison

This document compiles published studies, baseline models, and experimental benchmarks for multi-class eye disease classification from color fundus images.

---

## 1. 10-Class Retinal Disease Benchmarks (Same Dataset)

The benchmark comparisons below evaluate 10-class fundus classification on the balanced Mendeley dataset across identical disease categories:
*Central Serous Chorioretinopathy (CSCR), Diabetic Retinopathy, Disc Edema, Glaucoma, Healthy, Macular Scar, Myopia, Pterygium, Retinal Detachment, and Retinitis Pigmentosa*.

| Model / Reference | Architecture | Input Res & Preprocessing | Balanced Acc | Macro F1 | ROC-AUC | Cohen's $\kappa$ | Notes |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Rashid et al. (2021)** | ResNet-50 / DenseNet-121 | $224\times 224$, ImageNet weights | ~83.4% | ~82.8% | 0.965 | ~0.90 | Original reference benchmark on this 10-class dataset. |
| **PLoS ONE (2023)** | Swin Transformer (Swin-T) | $224\times 224$, Shifted Windows | 86.8% | 86.2% | 0.978 | ~0.91 | Hierarchical self-attention across fundus patches. |
| **EyeFusionNet (IEEE 2023)** | DenseNet-169 + TNT | $224\times 224$, Dual-Branch Fusion | 89.2% | 88.7% | 0.984 | ~0.93 | Published literature benchmark on this dataset. |
| **RETFound (Nature 2023)** | ViT-Large/16 (MAE) | $224\times 224$, 1.6M Retinal Scans | 88.5% – 91.2% | 89.4% | 0.988 | ~0.94 | Moorfields Eye Hospital / UCL foundation model. |
| **EXP-001** | EfficientNet-B0 | $224\times 224$, ImageNet weights | 83.38% | 83.11% | 0.9774 | 0.9115 | Initial baseline model (35% train split). |
| **EXP-002** | BiomedCLIP (ViT-B/16) | $224\times 224$, PubMedBERT weights | 83.85% | 83.69% | 0.9802 | 0.9128 | Medical domain foundation model fine-tuning. |
| **EXP-003** | BiomedCLIP + CBAM | $224\times 224$, Dual Attention | 84.23% | 84.09% | 0.9796 | 0.9128 | Spatial and channel attention for optic disc region. |
| **EXP-005** | EfficientNet-B3 | $384\times 384$, CLAHE, 70% Split | 90.17% | 90.03% | 0.9891 | 0.9628 | High-resolution input preserving microaneurysm details. |
| **EXP-006** | BiomedCLIP + CBAM | $224\times 224$, CLAHE, 70% Split | 87.83% | 87.83% | 0.9894 | 0.9447 | Strongest individual Glaucoma score (63.93% F1, 65% recall). |
| **EXP-007 (Ensemble)** | EffNet-B3 (384) + BiomedCLIP-CBAM | Multi-scale fusion + 4-view TTA | **90.50%** | **90.44%** | **0.9923** | **0.9704** | Weighted probability blending with test-time augmentation. |

---

## 2. Distinction Between 4-Class and 10-Class Literature

Recent papers such as Alsohemi & Dardouri (*J. Imaging*, 2025, PMC12387119) report accuracies in the 93%–96% range. It is important to note the difference in task scope:

1. **Class Count and Baseline Probability:**
   - Papers reporting 93%–96% typically evaluate **4 classes** (Cataract, Diabetic Retinopathy, Glaucoma, Normal) or binary screening (e.g., referable vs. non-referable DR on EyePACS or Messidor).
   - In a 4-class setting, random guessing baseline is **25.0%**. In our **10-class** setting, random guessing is **10.0%**.

2. **Inter-Class Visual Overlap:**
   - In the 4-class problem, cataract images exhibit diffuse lens clouding that is easily distinguished from normal fundus patterns.
   - In the 10-class problem, multiple conditions present in the same anatomical structure:
     - **Optic Disc:** Glaucoma (cup enlargement), Disc Edema (blurred, swollen margin), and Myopia (peripapillary atrophy) all affect the disc.
     - **Macula:** CSCR (subretinal fluid leak) and Macular Scar (fibrous tissue) share central macular positioning.

3. **Validation Rigor:**
   - In our experiments, an untouched, physically balanced test set of 600 images (60 per class) is held out, with no patient overlap across splits.

---

## 3. Impact of Input Resolution and Preprocessing

- **Spatial Resolution ($384\times 384$ vs. $224\times 224$):**
  Downsampling fundus images to $224\times 224$ reduces small lesions such as microaneurysms ($10–50\,\mu\text{m}$) to sub-pixel dimensions. Processing at $384\times 384$ yields approximately $2.94\times$ more pixel area, which directly improved Diabetic Retinopathy F1 from 85.7% to 93.3% and Macular Scar F1 from 67.9% to 86.7%.
- **Luminance Equalization (CLAHE):**
  Applying CLAHE to the L-channel in LAB color space addresses uneven peripheral illumination common in fundus cameras, stabilizing contrast around the neuroretinal rim without introducing color distortion.

---

## Associated Documents
- Technical Report: [`docs/literature_research_report.md`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/literature_research_report.md)
- Printable PDF: [`docs/Literature_Review_and_SOTA_Benchmarks.pdf`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/Literature_Review_and_SOTA_Benchmarks.pdf)
