"""Clinical Screening Inference Engine.
Handles image preprocessing, model loading, ensemble probability fusion,
uncertainty quantification, and Grad-CAM explainability maps.
"""

import base64
import gc
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.data.preprocessing import crop_fundus_area, apply_clahe
from src.models.backbone import FundusClassifier
from src.utils.gradcam import GradCAM, overlay_heatmap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 10-Class Disease Descriptions & Clinical Guidance
DISEASE_INFO_10CLASS = {
    "Central Serous Chorioretinopathy [Color Fundus]": {
        "short_name": "CSCR",
        "description": "Fluid accumulation under the retina resulting in serous neurosensory retinal detachment, typically centered around the macula.",
        "urgency": "Urgent (Within 1-2 weeks)",
        "urgency_level": "amber",
        "recommendations": [
            "Perform Optical Coherence Tomography (OCT) to quantify subretinal fluid thickness.",
            "Schedule Fluorescein Angiography (FA) to detect active focal pigment epithelial leaks.",
            "Advise discontinuation of any systemic corticosteroid medication if approved by prescribing physician."
        ]
    },
    "Diabetic Retinopathy": {
        "short_name": "Diabetic Retinopathy",
        "description": "Microvascular retinal complications secondary to diabetes mellitus, characterized by microaneurysms, hemorrhages, hard exudates, and neovascularization.",
        "urgency": "High Priority (Prompt Specialist Evaluation)",
        "urgency_level": "red",
        "recommendations": [
            "Perform multi-field fundus photography and OCT Macula to evaluate Diabetic Macular Edema (DME).",
            "Refer to a vitreoretinal specialist for anti-VEGF therapy or panretinal photocoagulation (PRP) if proliferative changes exist.",
            "Coordinate with endocrinologist for strict glycemic (HbA1c < 7.0%) and systemic blood pressure control."
        ]
    },
    "Disc Edema": {
        "short_name": "Disc Edema / Papilledema",
        "description": "Swelling of the optic nerve head with blurred disc margins, obscuration of surface vessels, and elevated peripapillary retinal folds.",
        "urgency": "Critical / Emergency",
        "urgency_level": "crimson",
        "recommendations": [
            "Immediate neurological and neuro-ophthalmic evaluation to rule out elevated intracranial pressure (ICP).",
            "Urgent brain neuroimaging (MRI Brain + MRV) before any lumbar puncture.",
            "Measure blood pressure to exclude malignant hypertension crisis."
        ]
    },
    "Glaucoma": {
        "short_name": "Glaucoma",
        "description": "Progressive optic neuropathy with characteristic optic cup enlargement (increased cup-to-disc ratio), neuroretinal rim thinning, and retinal nerve fiber layer (RNFL) loss.",
        "urgency": "High Priority",
        "urgency_level": "amber",
        "recommendations": [
            "Perform Goldmann applanation tonometry to determine Intraocular Pressure (IOP).",
            "Order Humphrey Visual Field (HVF 24-2 or 30-2) perimetry and RNFL OCT.",
            "Initiate IOP-lowering topical prostaglandin analogues or beta-blockers as directed by ophthalmologist."
        ]
    },
    "Healthy": {
        "short_name": "Normal / Physiological Fundus",
        "description": "Normal retinal vascular architecture, crisp optic disc margins with physiological cupping, sharp foveal light reflex, and absent pathological lesions.",
        "urgency": "Routine Surveillance",
        "urgency_level": "emerald",
        "recommendations": [
            "Maintain routine annual comprehensive eye examination.",
            "Continue standard systemic preventative health monitoring."
        ]
    },
    "Macular Scar": {
        "short_name": "Macular Scar",
        "description": "Fibrotic cicatricial tissue or chorioretinal atrophy in the central macular region, typically following end-stage exudative AMD, toxoplasmosis, or prior trauma.",
        "urgency": "Moderate",
        "urgency_level": "amber",
        "recommendations": [
            "OCT of the macula to evaluate active subretinal neovascularization vs inactive fibrovascular scarring.",
            "Provide Amsler grid self-monitoring chart to detect sudden metamorphopsia or acute visual changes.",
            "Assess low vision aids and visual rehabilitation resources if central visual acuity is reduced."
        ]
    },
    "Myopia": {
        "short_name": "Pathological Myopia",
        "description": "Degenerative structural alterations associated with excessive axial elongation, such as temporal peripapillary crescent, tilted disc, posterior staphyloma, or lacquer cracks.",
        "urgency": "Moderate Priority",
        "urgency_level": "blue",
        "recommendations": [
            "Annual dilated peripheral retinal examination to screen for asymptomatic retinal lattice degeneration or retinal tears.",
            "Perform axial length biometry and widefield fundus imaging.",
            "Educate patient regarding symptoms of acute posterior vitreous detachment (photopsia, floaters, curtain-like vision loss)."
        ]
    },
    "Pterygium": {
        "short_name": "Pterygium",
        "description": "Fibrovascular subepithelial growth of degenerative bulbar conjunctival tissue encroaching past the limbus onto the corneal surface.",
        "urgency": "Low to Moderate",
        "urgency_level": "amber",
        "recommendations": [
            "Assess visual axis encroachment, astigmatism, and corneal topography.",
            "Advise UV-protective sunglasses and lubricating ocular artificial tears.",
            "Refer for surgical conjunctival autograft excision if visually significant or chronically inflamed."
        ]
    },
    "Retinal Detachment": {
        "short_name": "Retinal Detachment",
        "description": "Separation of the neurosensory retina from the underlying retinal pigment epithelium (RPE) by subretinal fluid, visible as corrugations or mobile folds.",
        "urgency": "Critical / Emergency",
        "urgency_level": "crimson",
        "recommendations": [
            "Immediate emergency referral to a vitreoretinal surgeon (same-day evaluation).",
            "Instruct patient to minimize head movements and avoid physical exertion.",
            "Evaluate macula status (Macula-ON vs Macula-OFF) to determine urgency of surgical repair."
        ]
    },
    "Retinitis Pigmentosa": {
        "short_name": "Retinitis Pigmentosa",
        "description": "Hereditary progressive retinal dystrophy characterized by classic bone-spicule hyperpigmentation, arteriolar attenuation, and waxy disc pallor.",
        "urgency": "Moderate",
        "urgency_level": "amber",
        "recommendations": [
            "Perform visual field perimetry (Goldmann or automated) and full-field Electroretinography (ERG).",
            "Offer genetic counseling and molecular genetic testing for specific causative gene variants.",
            "Screen for associated cystoid macular edema (CME) via macular OCT."
        ]
    }
}

# 4-Class Disease Descriptions
DISEASE_INFO_4CLASS = {
    "cataract": {
        "short_name": "Cataract",
        "description": "Opacification and clouding of the crystalline lens impairing light transmission to the retina.",
        "urgency": "Moderate Priority",
        "urgency_level": "amber",
        "recommendations": [
            "Slit-lamp biomicroscopy and visual acuity testing.",
            "Surgical phacoemulsification evaluation with intraocular lens (IOL) implantation."
        ]
    },
    "diabetic_retinopathy": {
        "short_name": "Diabetic Retinopathy",
        "description": "Microvascular retinal complications secondary to diabetes, featuring hemorrhages, exudates, or neovascularization.",
        "urgency": "High Priority",
        "urgency_level": "red",
        "recommendations": [
            "OCT Macula evaluation for diabetic macular edema (DME).",
            "Vitreoretinal consultation for possible anti-VEGF or laser photocoagulation.",
            "Strict glycemic and blood pressure management."
        ]
    },
    "glaucoma": {
        "short_name": "Glaucoma",
        "description": "Optic neuropathy showing neuroretinal rim thinning, cup excavation, and nerve fiber layer loss.",
        "urgency": "High Priority",
        "urgency_level": "amber",
        "recommendations": [
            "Applanation tonometry (IOP check) and pachymetry (corneal thickness).",
            "Visual field perimetry and RNFL OCT imaging.",
            "IOP reduction therapy."
        ]
    },
    "normal": {
        "short_name": "Normal Retinal Fundus",
        "description": "Clear physiological optic disc, healthy vasculature, and absence of ocular pathology.",
        "urgency": "Routine Surveillance",
        "urgency_level": "emerald",
        "recommendations": [
            "Annual comprehensive preventative ocular exam."
        ]
    }
}

# Standard Model Definitions & Checkpoints
MODEL_REGISTRY = {
    "10class": {
        "classes": [
            "Central Serous Chorioretinopathy [Color Fundus]",
            "Diabetic Retinopathy",
            "Disc Edema",
            "Glaucoma",
            "Healthy",
            "Macular Scar",
            "Myopia",
            "Pterygium",
            "Retinal Detachment",
            "Retinitis Pigmentosa",
        ],
        "models": {
            "ensemble_quad": {
                "name": "Unified Quad Ensemble (SOTA)",
                "accuracy": "91.83%",
                "macro_f1": "91.78%",
                "is_ensemble": True,
                "components": [
                    ("configs/efficientnet_b3_384_clahe.yaml", "outputs/efficientnet_b3_384_clahe/best_model.pth", 0.429),
                    ("configs/resnet50d_384_clahe.yaml", "outputs/resnet50d_384_clahe/best_model.pth", 0.286),
                    ("configs/convnext_small_384_clahe.yaml", "outputs/convnext_small_384_clahe/best_model.pth", 0.143),
                    ("configs/biomedclip_cbam_fusion.yaml", "outputs/biomedclip_cbam_fusion/best_model.pth", 0.143),
                ]
            },
            "efficientnet_b3": {
                "name": "EfficientNet-B3 (384x384 CLAHE)",
                "accuracy": "91.00%",
                "macro_f1": "90.97%",
                "is_ensemble": False,
                "config": "configs/efficientnet_b3_384_clahe.yaml",
                "checkpoint": "outputs/efficientnet_b3_384_clahe/best_model.pth"
            },
            "convnext_small": {
                "name": "ConvNeXt-Small (384x384 CLAHE)",
                "accuracy": "90.50%",
                "macro_f1": "90.48%",
                "is_ensemble": False,
                "config": "configs/convnext_small_384_clahe.yaml",
                "checkpoint": "outputs/convnext_small_384_clahe/best_model.pth"
            },
            "resnet50d": {
                "name": "ResNet-50d (384x384 CLAHE)",
                "accuracy": "88.17%",
                "macro_f1": "88.10%",
                "is_ensemble": False,
                "config": "configs/resnet50d_384_clahe.yaml",
                "checkpoint": "outputs/resnet50d_384_clahe/best_model.pth"
            },
            "vit_base": {
                "name": "ViT-Base-384 (Vision Transformer)",
                "accuracy": "89.17%",
                "macro_f1": "88.75%",
                "is_ensemble": False,
                "config": "configs/vit_base_384_clahe.yaml",
                "checkpoint": "outputs/vit_base_384_clahe/best_model.pth"
            }
        }
    },
    "4class": {
        "classes": [
            "cataract",
            "diabetic_retinopathy",
            "glaucoma",
            "normal"
        ],
        "models": {
            "ensemble_quad_4class": {
                "name": "ResNet-Integrated Quad Ensemble (SOTA)",
                "accuracy": "95.74%",
                "macro_f1": "95.70%",
                "is_ensemble": True,
                "components": [
                    ("configs/kaggle_4class_biomedclip_cbam.yaml", "outputs/kaggle_4class_biomedclip_cbam/BEST_95.27pct_biomed.pth", 0.312),
                    ("configs/kaggle_4class_convnext_small_384.yaml", "outputs/kaggle_4class_convnext_v2/BEST_95.27pct_convnext.pth", 0.260),
                    ("configs/kaggle_4class_resnet50d.yaml", "outputs/kaggle_4class_resnet50d/best_model.pth", 0.234),
                    ("configs/kaggle_4class_efficientnet_b3_384.yaml", "outputs/kaggle_4class_efficientnet_b3_384/best_model.pth", 0.195),
                ]
            },
            "convnext_v2_4class": {
                "name": "ConvNeXt-Small v2 (384x384)",
                "accuracy": "94.33%",
                "macro_f1": "94.11%",
                "is_ensemble": False,
                "config": "configs/kaggle_4class_convnext_small_384.yaml",
                "checkpoint": "outputs/kaggle_4class_convnext_v2/BEST_95.27pct_convnext.pth"
            },
            "biomedclip_4class": {
                "name": "BiomedCLIP + CBAM Fusion",
                "accuracy": "95.04%",
                "macro_f1": "93.76%",
                "is_ensemble": False,
                "config": "configs/kaggle_4class_biomedclip_cbam.yaml",
                "checkpoint": "outputs/kaggle_4class_biomedclip_cbam/BEST_95.27pct_biomed.pth"
            }
        }
    }
}


def _get_target_layer_for_gradcam(model: torch.nn.Module):
    """Dynamically discover the most appropriate convolutional target layer for Grad-CAM."""
    if hasattr(model, "backbone"):
        bb = model.backbone
        if hasattr(bb, "conv_head") and bb.conv_head is not None:
            return bb.conv_head
        if hasattr(bb, "act2") and bb.act2 is not None:
            return bb.act2
        if hasattr(bb, "stages") and len(bb.stages) > 0:
            stage = bb.stages[-1]
            if hasattr(stage, "blocks") and len(stage.blocks) > 0:
                block = stage.blocks[-1]
                if hasattr(block, "conv_dw"):
                    return block.conv_dw
                return block
            return stage
        if hasattr(bb, "layer4") and len(bb.layer4) > 0:
            block = bb.layer4[-1]
            if hasattr(block, "conv3"):
                return block.conv3
            return block
        if hasattr(bb, "cbam") and hasattr(bb.cbam, "sa"):
            return bb.cbam.sa.conv

    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None


class InferenceEngine:
    """Manages model loading, caching, execution, and explainability."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_model_key = None
        self.active_model = None
        self.active_config = None

    def _load_single_model(self, config_path: str, checkpoint_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        full_cfg_path = PROJECT_ROOT / config_path
        full_ckpt_path = PROJECT_ROOT / checkpoint_path

        if not full_ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {full_ckpt_path}")

        with open(full_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_name = cfg["model"].get("name", "efficientnet_b0")
        num_classes = cfg["data"]["num_classes"]
        dropout = float(cfg["model"].get("dropout", 0.2))

        model = FundusClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        ).to(self.device)

        ckpt = torch.load(str(full_ckpt_path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        return model, cfg

    def _prepare_tensor(self, rgb_image: np.ndarray, image_size: int = 384, crop_fundus: bool = True, apply_clahe_flag: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        processed_rgb = rgb_image.copy()
        if crop_fundus:
            processed_rgb = crop_fundus_area(processed_rgb)
        if apply_clahe_flag:
            processed_rgb = apply_clahe(processed_rgb)

        resized_rgb = cv2.resize(processed_rgb, (image_size, image_size))

        norm = (resized_rgb / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        return tensor, resized_rgb

    def predict(
        self,
        rgb_image: np.ndarray,
        benchmark: str = "10class",
        model_id: Optional[str] = None,
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """Perform clinical inference and Grad-CAM explainability."""
        if benchmark not in MODEL_REGISTRY:
            benchmark = "10class"

        bench_info = MODEL_REGISTRY[benchmark]
        class_names = bench_info["classes"]
        available_models = bench_info["models"]

        if not model_id or model_id not in available_models:
            model_id = "ensemble_quad" if benchmark == "10class" else "ensemble_quad_4class"

        selected_model_meta = available_models[model_id]
        is_ensemble = selected_model_meta.get("is_ensemble", False)

        disease_dict = DISEASE_INFO_10CLASS if benchmark == "10class" else DISEASE_INFO_4CLASS

        if is_ensemble:
            # Multi-architecture ensemble fusion
            components = selected_model_meta["components"]
            fused_probs = np.zeros(len(class_names), dtype=np.float32)
            total_weight = sum(w for _, _, w in components)

            cam_model = None
            cam_tensor = None
            cam_resized_rgb = None

            for cfg_path, ckpt_path, weight in components:
                model, cfg = self._load_single_model(cfg_path, ckpt_path)
                img_size = cfg["data"].get("image_size", 384)
                crop_f = cfg["data"].get("crop_fundus", True)
                use_cl = cfg["data"].get("apply_clahe", True)

                tensor, resized_rgb = self._prepare_tensor(rgb_image, img_size, crop_f, use_cl)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    fused_probs += (weight / total_weight) * probs

                # Retain primary CNN for Grad-CAM
                if cam_model is None and "efficientnet" in cfg_path:
                    cam_model = model
                    cam_tensor = tensor
                    cam_resized_rgb = resized_rgb
                elif cam_model is None:
                    cam_model = model
                    cam_tensor = tensor
                    cam_resized_rgb = resized_rgb
                else:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            final_probs = fused_probs
            primary_model = cam_model
            eval_tensor = cam_tensor
            display_rgb = cam_resized_rgb
        else:
            # Standalone single model
            cfg_path = selected_model_meta["config"]
            ckpt_path = selected_model_meta["checkpoint"]

            if self.active_model_key == (benchmark, model_id) and self.active_model is not None:
                model = self.active_model
                cfg = self.active_config
            else:
                if self.active_model is not None:
                    del self.active_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                model, cfg = self._load_single_model(cfg_path, ckpt_path)
                self.active_model = model
                self.active_config = cfg
                self.active_model_key = (benchmark, model_id)

            img_size = cfg["data"].get("image_size", 384)
            crop_f = cfg["data"].get("crop_fundus", True)
            use_cl = cfg["data"].get("apply_clahe", True)

            tensor, resized_rgb = self._prepare_tensor(rgb_image, img_size, crop_f, use_cl)

            with torch.no_grad():
                logits = model(tensor)
                final_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

            primary_model = model
            eval_tensor = tensor
            display_rgb = resized_rgb

        # Identify top prediction and ranked distribution
        top_idx = int(np.argmax(final_probs))
        top_confidence = float(final_probs[top_idx])
        top_class_name = class_names[top_idx]

        # Uncertainty quantification: Normalized Shannon Entropy
        num_c = len(class_names)
        entropy = -float(np.sum(final_probs * np.log2(final_probs + 1e-12)))
        max_entropy = np.log2(num_c)
        norm_entropy = float(entropy / max_entropy)

        if top_confidence >= 0.85 and norm_entropy < 0.25:
            certainty_level = "High Confidence"
            certainty_badge = "success"
        elif top_confidence >= 0.60:
            certainty_level = "Moderate Confidence"
            certainty_badge = "warning"
        else:
            certainty_level = "Ambiguous / Manual Review Recommended"
            certainty_badge = "danger"

        # Ranked probability distribution
        ranked_indices = np.argsort(-final_probs)
        distribution = []
        for rank, idx in enumerate(ranked_indices):
            c_name = class_names[idx]
            p = float(final_probs[idx])
            info = disease_dict.get(c_name, {})
            distribution.append({
                "rank": rank + 1,
                "class_name": c_name,
                "short_name": info.get("short_name", c_name),
                "probability": p,
                "percentage": round(p * 100, 2),
                "urgency": info.get("urgency", "Standard"),
                "urgency_level": info.get("urgency_level", "blue"),
            })

        # Grad-CAM explainability generation
        gradcam_data = None
        if generate_cam and primary_model is not None and eval_tensor is not None:
            try:
                target_layer = _get_target_layer_for_gradcam(primary_model)
                if target_layer is not None:
                    cam = GradCAM(primary_model, target_layer)
                    with torch.enable_grad():
                        heatmap = cam.generate(eval_tensor, class_idx=top_idx)

                    overlaid = overlay_heatmap(display_rgb, heatmap, alpha=0.55)

                    # Encode into base64 images
                    def _np_to_base64_png(arr_rgb: np.ndarray) -> str:
                        im = Image.fromarray(arr_rgb.astype(np.uint8))
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                    gradcam_data = {
                        "preprocessed_b64": _np_to_base64_png(display_rgb),
                        "heatmap_b64": _np_to_base64_png(heatmap_colored),
                        "overlay_b64": _np_to_base64_png(overlaid),
                    }
            except Exception as e:
                print(f"Grad-CAM generation error: {e}")

        # Clinical details for the top prediction
        top_info = disease_dict.get(top_class_name, {
            "short_name": top_class_name,
            "description": "Retinal diagnosis classification.",
            "urgency": "Standard",
            "urgency_level": "blue",
            "recommendations": ["Comprehensive ocular examination."]
        })

        return {
            "benchmark": benchmark,
            "model_id": model_id,
            "model_name": selected_model_meta["name"],
            "model_accuracy": selected_model_meta["accuracy"],
            "top_prediction": {
                "class_name": top_class_name,
                "short_name": top_info["short_name"],
                "confidence": top_confidence,
                "percentage": round(top_confidence * 100, 2),
                "urgency": top_info["urgency"],
                "urgency_level": top_info["urgency_level"],
                "description": top_info["description"],
                "recommendations": top_info["recommendations"]
            },
            "uncertainty": {
                "entropy": round(entropy, 3),
                "normalized_entropy": round(norm_entropy, 3),
                "certainty_level": certainty_level,
                "certainty_badge": certainty_badge
            },
            "distribution": distribution,
            "gradcam": gradcam_data
        }
