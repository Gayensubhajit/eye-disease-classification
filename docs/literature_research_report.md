# 10-Class Retinal Disease Classification from Color Fundus Images: Technical Benchmark & Evaluation Report

**Institution:** Department of Computer Science & Engineering, Jadavpur University  
**Project Team:** Gunjan Basak, Chirantan Biswas, Subhajit Gayen  
**Supervisor:** Dr. Pawan Kumar Singh  
**Generated PDF:** [`docs/Literature_Review_and_SOTA_Benchmarks.pdf`](file:///home/silentbyte/Documents/GitHub/eye-disease-classification/docs/Literature_Review_and_SOTA_Benchmarks.pdf)

---

## 1. Overview & Experimental Setup

This project investigates deep learning architectures for automated classification of 10 distinct retinal conditions from color fundus photography. The dataset is organized into a folder-based structure to maintain reproducible training, validation, and test partitions:

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
5. **EXP-005 (EfficientNet-B3 at 384x384 with CLAHE, 70% Split):** Evaluated on the expanded 2,800-image training set at $384\times 384$ resolution with local adaptive contrast enhancement. Test accuracy reached 90.17% with 90.03% macro F1, improving fine lesion detection.
6. **EXP-006 (BiomedCLIP + CBAM on 70% Split):** Retrained the novel dual-attention architecture on the expanded training set. Reached 87.83% test accuracy and achieved our highest individual Glaucoma F1-score of 63.93% (65.0% sensitivity).
7. **EXP-007 (Mega-Ensemble with 4-View TTA):** Blended the high-resolution features of EfficientNet-B3 ($384\times 384$) with the semantic embeddings of BiomedCLIP-CBAM using multi-view flip averaging. Achieved 90.50% test accuracy, 90.44% macro F1, and 0.9923 ROC-AUC.

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
| **EXP-007 (Our Ensemble + TTA)** | EffNet-B3 + BiomedCLIP-CBAM | Multi-scale | **90.50%** | **90.44%** | **0.9923** | **0.9704** |

---

## 4. Class-by-Class Performance Analysis (EXP-007 on 600 Test Images)

| Disease Category | Test Images | Sensitivity (Recall) | Specificity | F1-Score | Clinical Finding |
|---|:---:|:---:|:---:|:---:|---|
| **Pterygium** | 60 | 1.0000 | 1.0000 | **1.0000** | Distinct corneal/conjunctival lesion; completely separated. |
| **Retinal Detachment** | 60 | 1.0000 | 1.0000 | **1.0000** | Retinal elevation folds reliably identified without false positives. |
| **Retinitis Pigmentosa** | 60 | 1.0000 | 1.0000 | **1.0000** | Bone-spicule pigmentation patterns detected across peripheral retina. |
| **Disc Edema** | 60 | 1.0000 | 0.9981 | **0.9917** | Blurring of the optic disc margin captured with high specificity. |
| **CSCR [Color Fundus]** | 60 | 0.9833 | 0.9944 | **0.9672** | Serous macular elevation distinguished from surrounding parenchyma. |
| **Diabetic Retinopathy** | 60 | 0.9500 | 0.9926 | **0.9421** | Microaneurysms and exudate clusters resolved effectively at higher input resolution. |
| **Macular Scar** | 60 | 0.8500 | 0.9852 | **0.8571** | Fibrous demarcations separated from healthy central macula. |
| **Myopia** | 60 | 0.8167 | 0.9833 | **0.8305** | Peripapillary crescents and chorioretinal thinning differentiated from normal discs. |
| **Healthy** | 60 | 0.8167 | 0.9796 | **0.8167** | High specificity (97.96%) minimizing false alarms in regular screenings. |
| **Glaucoma** | 60 | 0.6333 | 0.9611 | **0.6387** | Optic cup excavation identified; CBAM spatial attention provided noticeable improvement over baseline. |
| **Overall Macro Average** | **600** | **90.50%** | **98.94%** | **90.44%** | **Macro ROC-AUC: 0.9923, Cohen's Kappa: 0.9704** |

---

## 5. Technical Insights & Observations

1. **Resolution vs. Micro-Lesion Visibility:**
   Fundus cameras capture wide-angle images where diagnostic cues (such as early microaneurysms or subtle cup-to-disc margins) are small. Elevating input resolution to $384\times 384$ tripled the available pixel area, providing the network with sharper gradient information that significantly boosted Diabetic Retinopathy and Macular Scar recognition.
2. **Local Contrast Normalization (CLAHE):**
   Retinal images naturally suffer from illumination fall-off toward the image perimeter. Applying CLAHE within the luminance channel of the LAB color space balanced local dynamic range without distorting physiological color cues.
3. **Attention in the Optic Disc Region:**
   Glaucoma and Disc Edema present in the same anatomical structure. The CBAM module's spatial attention mechanism helped the model concentrate features on the optic disc margin, improving sensitivity on these overlapping categories.
4. **Multi-Model Complementarity:**
   The ensemble combined the high-resolution spatial feature extraction of EfficientNet-B3 with the semantic biomedical representations of BiomedCLIP, yielding lower variance and our highest overall classification metrics.
