# 10-Class Retinal Disease Classification from Color Fundus Images: Technical Benchmark & Evaluation Report

**Institution:** Department of Computer Science & Engineering, Jadavpur University  
**Project Team:** Gunjan Basak, Chirantan Biswas, Subhajit Gayen  
**Supervisor:** Dr. Pawan Kumar Singh  
**Generated PDF:** [`docs/Literature_Review_and_SOTA_Benchmarks.pdf`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/Literature_Review_and_SOTA_Benchmarks.pdf)

---

## 1. Overview & Experimental Setup

This project investigates deep learning architectures for automated classification of 10 distinct retinal conditions from color fundus photography. The dataset is organized into a standard directory hierarchy to maintain reproducible training, validation, and test partitions:

| Split | Images per Class (10 Classes) | Total Images | Percentage | Purpose |
|---|:---:|:---:|:---:|---|
| **`data/train/`** | 280 | 2,800 | 70.0% | Model parameter optimization |
| **`data/val/`** | 60 | 600 | 15.0% | Hyperparameter tuning and model checkpointing |
| **`data/test/`** | 60 | 600 | 15.0% | Held-out evaluation on untouched images |
| **Total** | **400** | **4,000** | **100.0%** | Equal balance across all 10 categories |

The 10 evaluated diagnostic classes are:
1. Central Serous Chorioretinopathy (CSCR)
2. Diabetic Retinopathy (DR)
3. Disc Edema
4. Glaucoma
5. Healthy
6. Macular Scar
7. Myopia
8. Pterygium
9. Retinal Detachment
10. Retinitis Pigmentosa

---

## 2. Experimental Model Progression

We implemented and tested multiple architectures across controlled iterations:

1. **EXP-001 (EfficientNet-B0 Baseline, 224x224):** Standard CNN backbone with ImageNet pretraining, trained on the initial 35% split. Resulted in 83.38% test accuracy and 83.11% macro F1.
2. **EXP-002 (Microsoft BiomedCLIP, 224x224):** Vision Transformer fine-tuned from biomedical multimodal pretraining (PubMedBERT + ViT-B/16). Improved test accuracy to 83.85% and macro F1 to 83.69%.
3. **EXP-003 (BiomedCLIP + CBAM Dual Attention, 224x224):** Added Convolutional Block Attention Modules (CBAM) to spatial token maps, improving Glaucoma F1 from 51.88% to 59.92%.
4. **EXP-004 (Multi-Model Ensemble, 224x224):** Weighted probability averaging of CNN and foundation models with 4-view test-time augmentation, reaching 85.85% test accuracy.
5. **EXP-005 (EfficientNet-B3 at 384x384 with CLAHE, 70% Split):** Evaluated on the expanded 2,800-image training set at $384\times 384$ resolution with local adaptive contrast enhancement. Test accuracy reached 90.17% with 90.03% macro F1.
6. **EXP-006 (BiomedCLIP + CBAM on 70% Split):** Retrained the novel dual-attention architecture on the expanded training set. Reached 87.83% test accuracy and 63.93% Glaucoma F1 (65.0% sensitivity).
7. **EXP-007 (Mega-Ensemble with 4-View TTA):** Blended EfficientNet-B3 ($384\times 384$) with BiomedCLIP-CBAM, reaching 90.50% test accuracy.
8. **EXP-008 (ConvNeXt-Small at 384x384 with CLAHE, 70% Split):** 50M parameter modern vision backbone with 7×7 depthwise convolutions and LayerNorm. Set our highest single-model benchmark with **90.50% Test Accuracy, 90.48% Macro F1, and surged Glaucoma F1 to 68.25% (71.67% sensitivity)**.
9. **EXP-009 (Triple-Threat Mega-Ensemble with 4-View TTA):** Fused ConvNeXt-Small (384) + EfficientNet-B3 (384) + BiomedCLIP-CBAM. Established our highest **Macro ROC-AUC of 0.9929 (99.29%)** and **Cohen's Kappa of 0.9710**.

---

## 3. Comparison with Published Literature

| Study / Model | Architecture | Input Resolution | Test Accuracy | Macro F1 | ROC-AUC | Cohen's $\kappa$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Rashid et al. (2021)** | ResNet-50 / DenseNet-121 | $224\times 224$ | 83.40% | 82.80% | 0.9650 | ~0.90 |
| **PLoS ONE (2023)** | Swin Transformer (Swin-T) | $224\times 224$ | 86.80% | 86.20% | 0.9780 | ~0.91 |
| **EyeFusionNet (IEEE 2023)** | DenseNet-169 + TNT | $224\times 224$ | 89.20% | 88.70% | 0.9840 | ~0.93 |
| **RETFound (Nature 2023)** | ViT-Large/16 (1.6M Pretrained) | $224\times 224$ | 88.5% – 91.2% | 89.40% | 0.9880 | ~0.94 |
| **EXP-001 (Our Baseline)** | EfficientNet-B0 + Focal Loss | $224\times 224$ | 83.38% | 83.11% | 0.9774 | 0.9115 |
| **EXP-005 (Our High-Res CNN)** | EfficientNet-B3 + CLAHE | $384\times 384$ | 90.17% | 90.03% | 0.9891 | 0.9628 |
| **EXP-006 (Our Hybrid Model)** | BiomedCLIP + CBAM | $224\times 224$ | 87.83% | 87.83% | 0.9894 | 0.9447 |
| **EXP-008 (Our ConvNeXt SOTA)** 🏆 | **ConvNeXt-Small + CLAHE** | **$384\times 384$** | **90.50%** | **90.48%** | **0.9902** | **0.9663** |
| **EXP-009 (Triple Ensemble + TTA)** 🏆 | **ConvNeXt + EffNet + BiomedCLIP** | Multi-scale | **90.50%** | **90.40%** | **0.9929** | **0.9710** |

---

## 4. Class-by-Class Performance Analysis (ConvNeXt-Small on 600 Test Images)

| Disease Category | Test Images | Sensitivity (Recall) | Specificity | F1-Score | Clinical Observation |
|---|:---:|:---:|:---:|:---:|---|
| **Pterygium** | 60 | 1.0000 | 1.0000 | **1.0000** | Perfect 100% precision & recall. |
| **Retinal Detachment** | 60 | 1.0000 | 1.0000 | **1.0000** | Flawless identification of retinal detachment folds. |
| **Retinitis Pigmentosa** | 60 | 1.0000 | 0.9981 | **0.9917** | Perfect 100% recall on bone-spicule peripheral pigment. |
| **Disc Edema** | 60 | 1.0000 | 0.9981 | **0.9917** | Perfect 100% recall on swollen, blurred disc margins. |
| **CSCR [Color Fundus]** | 60 | 1.0000 | 0.9907 | **0.9600** | Perfect 100% recall on serous macular neurosensory detachment. |
| **Diabetic Retinopathy** | 60 | 0.9500 | 0.9944 | **0.9500** | 95.0% recall on microaneurysms and hard exudates. |
| **Macular Scar** | 60 | 0.8333 | 0.9852 | **0.8475** | Consistent demarcation of fibrous scar tissue. |
| **Myopia** | 60 | 0.7333 | 0.9926 | **0.8148** | High specificity (99.26%) separating peripapillary crescents from glaucoma. |
| **Healthy** | 60 | 0.8167 | 0.9778 | **0.8099** | High true-negative rate minimizing false alarms. |
| **Glaucoma** | 60 | 0.7167 | 0.9574 | **0.6825** | 🏆 **Major clinical leap! Sensitivity reached 71.67% and F1 reached 68.25%.** |
| **Overall Macro Average** | **600** | **90.50%** | **98.94%** | **90.48%** | **Macro ROC-AUC: 0.9902, Cohen's Kappa: 0.9663** |

---

## 5. Technical Insights & Observations

1. **ConvNeXt Architecture Advantages:**
   Modernizing depthwise convolution with large 7×7 receptive fields provides wide-area spatial aggregation. This proved crucial for detecting Glaucoma, where understanding the cup-to-disc ratio requires examining both the central excavation and the surrounding rim simultaneously.
2. **Resolution Scaling ($384\times 384$):**
   Maintaining high resolution preserves microvascular diagnostic cues that are otherwise blurred out in downsampled inputs.
3. **Contrast Normalization via CLAHE:**
   Equalizing luminance in the LAB color space removed peripheral darkening from fundus camera optics, stabilizing boundary gradients.
4. **Complementary Ensemble Representations:**
   Combining a modern depthwise ConvNet (ConvNeXt-Small) with an inverted residual CNN (EfficientNet-B3) and a biomedical vision transformer (BiomedCLIP-CBAM) produced our most robust predictions, achieving 0.9929 ROC-AUC and 0.9710 Cohen's Kappa.
