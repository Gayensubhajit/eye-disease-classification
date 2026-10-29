#!/usr/bin/env python3
"""
Milestone V6: RFMiD External Transferability Preflight Evaluation Script.

Evaluates the transferability of frozen 10-class models on the external
Retinal Fundus Multi-Disease Image Dataset (RFMiD v1.0, N=3,200).

Methodological Principles:
1. Pure Inference: No training, fine-tuning, calibration, or threshold optimization.
2. Exploratory Ranking Only: Evaluated using One-vs-Rest (OvR) AUROC and Average Precision (AP),
   treating 10-class softmax outputs strictly as ranking scores for mapped external labels.
   Does NOT treat softmax outputs as calibrated multi-label probabilities or argmax predictions.
3. Strict Quarantine: data_clean/test is NEVER accessed.
4. Defensible Mappings Only: Direct pathological correspondence where clinically valid;
   explicit surrogate labels (ODC, MYA); ineligible labels explicitly marked (Pterygium, RD).
"""

import os
import sys
import json
import hashlib
import gc
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.backbone import build_model
from src.data.preprocessing import crop_fundus_area, apply_clahe, get_val_transforms

# Checkpoint configurations
CHECKPOINTS = {
    "convnext_small": {
        "display": "ConvNeXt-Small (384x384 CLAHE)",
        "config": "configs/clean_10class_convnext_small_384_clean.yaml",
        "checkpoint": "outputs/clean_10class_convnext_small_384_clean/best_model.pth",
        "expected_sha256": "238e5d081f59fbe4c5d3148d38972bdf4eb209df16972c74aee47f76df1aa60d",
    },
    "swin_base": {
        "display": "Swin-Base (384x384 CLAHE)",
        "config": "configs/clean_10class_swin_base_patch4_window12_384.yaml",
        "checkpoint": "outputs/clean_10class_swin_base_384_clean/best_model.pth",
        "expected_sha256": "6d118211b29368d71e49ad1dbb5e439ba7f798e416bb4fa32a963e3eac8fbef2",
    },
}


# Explicit canonical disease name mapping and alias resolution
DISEASE_CANONICAL_MAP = {
    "CSCR": "Central Serous Chorioretinopathy [Color Fundus]",
    "Central Serous Chorioretinopathy": "Central Serous Chorioretinopathy [Color Fundus]",
    "Central Serous Chorioretinopathy [Color Fundus]": "Central Serous Chorioretinopathy [Color Fundus]",
    "DR": "Diabetic Retinopathy",
    "Diabetic Retinopathy": "Diabetic Retinopathy",
    "Disc Edema": "Disc Edema",
    "ODE": "Disc Edema",
    "Glaucoma": "Glaucoma",
    "Healthy": "Healthy",
    "Macular Scar": "Macular Scar",
    "MS": "Macular Scar",
    "Myopia": "Myopia",
    "MYA": "Myopia",
    "Pterygium": "Pterygium",
    "Retinal Detachment": "Retinal Detachment",
    "RD": "Retinal Detachment",
    "Retinitis Pigmentosa": "Retinitis Pigmentosa",
    "RP": "Retinitis Pigmentosa",
}

LABEL_MAPPINGS = [
    {
        "internal_class": "Diabetic Retinopathy",
        "rfmid_col": "DR",
        "mapping_type": "Exact Disease",
        "notes": "Direct pathological correspondence to non-proliferative/proliferative diabetic retinopathy.",
        "status": "ELIGIBLE",
    },
    {
        "internal_class": "Healthy",
        "rfmid_col": "Disease_Risk_0",
        "mapping_type": "Exact Screening Normal",
        "notes": "Direct correspondence: RFMiD Disease_Risk == 0 designates images with no detected disease.",
        "status": "ELIGIBLE",
    },
    {
        "internal_class": "Central Serous Chorioretinopathy [Color Fundus]",
        "rfmid_col": "CSR",
        "mapping_type": "Exact Disease",
        "notes": "Direct correspondence to central serous chorioretinopathy / neurosensory detachment.",
        "status": "ELIGIBLE",
    },
    {
        "internal_class": "Disc Edema",
        "rfmid_col": "ODE",
        "mapping_type": "Exact Pathology",
        "notes": "Direct correspondence to optic disc edema (papilledema / blurred margins).",
        "status": "ELIGIBLE",
    },
    {
        "internal_class": "Macular Scar",
        "rfmid_col": "MS",
        "mapping_type": "Exact Pathology",
        "notes": "Direct correspondence to macular scar (cicatricial fibrosis). Limited positive support (N=27); wide CIs.",
        "status": "ELIGIBLE (Limited Support)",
    },
    {
        "internal_class": "Glaucoma",
        "rfmid_col": "ODC",
        "mapping_type": "Anatomical Surrogate",
        "notes": "Anatomical sign surrogate only (Optic Disc Cupping). Does NOT constitute clinical glaucoma diagnosis.",
        "status": "ELIGIBLE (Surrogate Sign)",
    },
    {
        "internal_class": "Myopia",
        "rfmid_col": "MYA",
        "mapping_type": "Morphological Surrogate",
        "notes": "Morphological surrogate only (Myopic Atrophy). Internal class trained on broader high-myopia fundus.",
        "status": "ELIGIBLE (Surrogate Feature)",
    },
    {
        "internal_class": "Retinitis Pigmentosa",
        "rfmid_col": "RP",
        "mapping_type": "Exact Pathology",
        "notes": "Direct correspondence, but severely inadequate sample size (N=10 across 3,200). Exploratory only.",
        "status": "EXPLORATORY ONLY (N=10)",
    },
    {
        "internal_class": "Retinal Detachment",
        "rfmid_col": "RT",
        "mapping_type": "Non-Equivalent",
        "notes": "RFMiD annotates Retinal Traction Detachment (RT, N=25). General/rhegmatogenous RD unavailable.",
        "status": "INELIGIBLE (Traction Only)",
    },
    {
        "internal_class": "Pterygium",
        "rfmid_col": None,
        "mapping_type": "Unavailable",
        "notes": "Anterior segment surface condition. Completely absent from posterior fundus RFMiD.",
        "status": "INELIGIBLE (Absent from Dataset)",
    },
]


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class RFMiDDataset(Dataset):
    """External RFMiD Dataset with standard validation preprocessing."""
    def __init__(self, image_paths: List[str], image_size: int = 384):
        self.image_paths = image_paths
        self.transform = get_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_fundus_area(image)
        image = apply_clahe(image)
        augmented = self.transform(image=image)
        return augmented["image"]


def compute_stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float, float, float]:
    """Compute AUROC and AP with 95% stratified bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Point estimates
    try:
        point_auc = roc_auc_score(y_true, y_score)
    except Exception:
        point_auc = float("nan")

    try:
        point_ap = average_precision_score(y_true, y_score)
    except Exception:
        point_ap = float("nan")

    boot_aucs = []
    boot_aps = []

    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        b_y = y_true[b_idx]
        b_s = y_score[b_idx]

        try:
            boot_aucs.append(roc_auc_score(b_y, b_s))
        except Exception:
            pass

        try:
            boot_aps.append(average_precision_score(b_y, b_s))
        except Exception:
            pass

    if len(boot_aucs) > 0:
        auc_ci_lower = float(np.percentile(boot_aucs, 2.5))
        auc_ci_upper = float(np.percentile(boot_aucs, 97.5))
    else:
        auc_ci_lower, auc_ci_upper = float("nan"), float("nan")

    if len(boot_aps) > 0:
        ap_ci_lower = float(np.percentile(boot_aps, 2.5))
        ap_ci_upper = float(np.percentile(boot_aps, 97.5))
    else:
        ap_ci_lower, ap_ci_upper = float("nan"), float("nan")

    return point_auc, auc_ci_lower, auc_ci_upper, point_ap, ap_ci_lower, ap_ci_upper


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("Milestone V6: RFMiD External Preflight Evaluation")
    print(f"Device: {device}")
    print("=======================================================\n")

    # Step 1: Checkpoint & Configuration Verification
    print("--- 1. Checkpoint & Configuration Verification ---")
    import yaml
    model_class_orders = {}
    for model_key, info in CHECKPOINTS.items():
        ckpt_p = REPO_ROOT / info["checkpoint"]
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        actual_sha = compute_sha256(ckpt_p)
        assert actual_sha == info["expected_sha256"], f"SHA mismatch for {model_key}: {actual_sha} vs {info['expected_sha256']}"
        print(f"✅ {info['display']}: Verified SHA-256 = {actual_sha}")

        cfg_p = REPO_ROOT / info["config"]
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)
        classes_in_cfg = cfg_data["data"]["class_names"]
        assert len(classes_in_cfg) == 10, f"Expected 10 classes in {cfg_p}, found {len(classes_in_cfg)}"
        model_class_orders[model_key] = classes_in_cfg
        print(f"✅ {info['display']}: Loaded and verified 10 class names from {info['config']}")

    # Assert class orders match identically across both models
    assert model_class_orders["convnext_small"] == model_class_orders["swin_base"], (
        "Class names order mismatch between ConvNeXt and Swin: " + str(model_class_orders)
        
        
    )
    declared_class_names = model_class_orders["convnext_small"]
    print("✅ Verified identical class ordering across ConvNeXt-Small and Swin-Base configs:")
    for idx, name in enumerate(declared_class_names):
        print(f"   [{idx}] {name}")

    # Step 2: Load and Verify RFMiD Labels & Image Paths
    print("\n--- 2. RFMiD Dataset & Image Alignment Verification ---")
    splits_info = [
        ("train", "data_external/rfmid_v1/download_raw/Training_Set/Training_Set/RFMiD_Training_Labels.csv", "data_external/rfmid_v1/download_raw/Training_Set/Training_Set/Training"),
        ("val", "data_external/rfmid_v1/download_raw/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv", "data_external/rfmid_v1/download_raw/Evaluation_Set/Evaluation_Set/Validation"),
        ("test", "data_external/rfmid_v1/download_raw/Test_Set/Test_Set/RFMiD_Testing_Labels.csv", "data_external/rfmid_v1/download_raw/Test_Set/Test_Set/Test")
    ]

    all_dfs = []
    all_image_paths = []

    for split_name, csv_rel, img_dir_rel in splits_info:
        csv_path = REPO_ROOT / csv_rel
        img_dir = REPO_ROOT / img_dir_rel
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
        df["split"] = split_name
        
        # Verify file existence
        img_paths_split = []
        for img_id in df["ID"]:
            p = img_dir / f"{img_id}.png"
            if not p.exists():
                raise FileNotFoundError(f"Missing RFMiD image: {p}")
            img_paths_split.append(str(p))
            all_image_paths.append(str(p))

        df["image_path"] = img_paths_split
        all_dfs.append(df)
        print(f"  -> Split '{split_name}': {len(df)} images verified in {img_dir_rel}")

    full_rfmid_df = pd.concat(all_dfs, ignore_index=True)
    assert len(full_rfmid_df) == 3200
    assert len(set(all_image_paths)) == 3200
    print(f"✅ All 3,200 RFMiD images verified and mapped 1-to-1 with unique label rows.")

    # Create synthesized Disease_Risk == 0 column for normal fundus
    full_rfmid_df["Disease_Risk_0"] = (full_rfmid_df["Disease_Risk"] == 0).astype(int)

    # Step 3: Run Inference
    print("\n--- 3. Running FP32 Inference on RFMiD Cohort (N=3,200) ---")
    rfmid_ds = RFMiDDataset(all_image_paths, image_size=384)
    rfmid_loader = DataLoader(
        rfmid_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model_probs = {}

    for model_key, info in CHECKPOINTS.items():
        print(f"\nRunning inference for {info['display']}...")
        cfg_file = REPO_ROOT / info["config"]
        import yaml
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = build_model(cfg).to(device)
        ckpt = torch.load(REPO_ROOT / info["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs_list = []
        start_t = time.time()
        with torch.no_grad():
            for batch_idx, images in enumerate(rfmid_loader):
                images = images.to(device, non_blocking=True)
                logits = model(images).float()
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(rfmid_loader):
                    print(f"  Processed {min((batch_idx + 1) * 8, 3200)}/3200 images...")

        elapsed = time.time() - start_t
        p_arr = np.concatenate(probs_list, axis=0)
        assert p_arr.shape == (3200, 10), f"Unexpected output shape: {p_arr.shape}"
        model_probs[model_key] = p_arr
        print(f"  -> Completed inference in {elapsed:.1f}s ({elapsed/3200*1000:.1f} ms/image).")

        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 4: Compute Metrics for Mapped Labels
    print("\n=======================================================")
    print("--- 4. One-vs-Rest Transferability Ranking Metrics ---")
    print("=======================================================\n")

    out_dir = REPO_ROOT / "outputs" / "external_rfmid_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for mapping in LABEL_MAPPINGS:
        int_class = mapping["internal_class"]
        # Dynamic, validated resolution of class index against declared model class_names
        canonical_name = DISEASE_CANONICAL_MAP.get(int_class, int_class)
        assert canonical_name in declared_class_names, f"Class '{canonical_name}' not found in model class_names: {declared_class_names}"
        int_idx = declared_class_names.index(canonical_name)
        assert declared_class_names[int_idx] == canonical_name, f"Index resolution mismatch for {canonical_name}"
        col = mapping["rfmid_col"]
        status = mapping["status"]
        mtype = mapping["mapping_type"]

        if col is None or col not in full_rfmid_df.columns:
            print(f"❌ {int_class:40s} -> [{status}] (No direct mapping)")
            summary_rows.append({
                "internal_class": int_class,
                "rfmid_label": "N/A",
                "mapping_type": mtype,
                "status": status,
                "positives": 0,
                "negatives": 3200,
                "prevalence_pct": 0.0,
                "convnext_auroc": None,
                "convnext_auroc_95ci": None,
                "convnext_ap": None,
                "convnext_ap_95ci": None,
                "swin_auroc": None,
                "swin_auroc_95ci": None,
                "swin_ap": None,
                "swin_ap_95ci": None,
                "notes": mapping["notes"],
            })
            continue

        y_true = full_rfmid_df[col].values
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        prev = n_pos / len(y_true) * 100.0

        # ConvNeXt Scores
        s_cnext = model_probs["convnext_small"][:, int_idx]
        c_auc, c_auc_l, c_auc_u, c_ap, c_ap_l, c_ap_u = compute_stratified_bootstrap_ci(y_true, s_cnext, n_boot=1000)

        # Swin-Base Scores
        s_swin = model_probs["swin_base"][:, int_idx]
        s_auc, s_auc_l, s_auc_u, s_ap, s_ap_l, s_ap_u = compute_stratified_bootstrap_ci(y_true, s_swin, n_boot=1000)

        print(f"[{status}] {int_class} ↔ {col} (Pos={n_pos}, Neg={n_neg}, Prev={prev:.2f}%):")
        print(f"  ConvNeXt-Small : AUROC = {c_auc:.4f} [95% CI: {c_auc_l:.4f}-{c_auc_u:.4f}] | AP = {c_ap:.4f} [95% CI: {c_ap_l:.4f}-{c_ap_u:.4f}]")
        print(f"  Swin-Base      : AUROC = {s_auc:.4f} [95% CI: {s_auc_l:.4f}-{s_auc_u:.4f}] | AP = {s_ap:.4f} [95% CI: {s_ap_l:.4f}-{s_ap_u:.4f}]")

        summary_rows.append({
            "internal_class": int_class,
            "rfmid_label": col,
            "mapping_type": mtype,
            "status": status,
            "positives": n_pos,
            "negatives": n_neg,
            "prevalence_pct": prev,
            "convnext_auroc": c_auc,
            "convnext_auroc_95ci": [c_auc_l, c_auc_u],
            "convnext_ap": c_ap,
            "convnext_ap_95ci": [c_ap_l, c_ap_u],
            "swin_auroc": s_auc,
            "swin_auroc_95ci": [s_auc_l, s_auc_u],
            "swin_ap": s_ap,
            "swin_ap_95ci": [s_ap_l, s_ap_u],
            "notes": mapping["notes"],
        })

    # Step 5: Save Deliverables
    print("\n--- 5. Saving Deliverables ---")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "rfmid_evaluation_summary.csv", index=False)

    # Save per-image predictions
    df_preds_swin = pd.DataFrame(model_probs["swin_base"], columns=cfg["data"]["class_names"])
    df_preds_swin.insert(0, "ID", full_rfmid_df["ID"])
    df_preds_swin.insert(1, "split", full_rfmid_df["split"])
    df_preds_swin.insert(2, "image_path", full_rfmid_df["image_path"])
    df_preds_swin.to_csv(out_dir / "rfmid_predictions_swin_base.csv", index=False)

    df_preds_cnext = pd.DataFrame(model_probs["convnext_small"], columns=cfg["data"]["class_names"])
    df_preds_cnext.insert(0, "ID", full_rfmid_df["ID"])
    df_preds_cnext.insert(1, "split", full_rfmid_df["split"])
    df_preds_cnext.insert(2, "image_path", full_rfmid_df["image_path"])
    df_preds_cnext.to_csv(out_dir / "rfmid_predictions_convnext_small.csv", index=False)

    report_payload = {
        "dataset": "RFMiD v1.0",
        "cohort_size": 3200,
        "evaluation_protocol": "One-vs-Rest Binary AUROC & Average Precision via pure FP32 inference",
        "checkpoints_verified": CHECKPOINTS,
        "results": summary_rows,
        "methodological_limitations": [
            "RFMiD is multi-label (23.41% multi-disease prevalence); models were trained with 10-class mutually exclusive softmax.",
            "Softmax output scores represent exploratory ranking discrimination, not calibrated multi-label posterior probabilities.",
            "ODC and MYA are surrogate morphological signs, not confirmed clinical diagnostic labels for Glaucoma or pathological Myopia.",
            "Pterygium is absent (anterior segment); Retinal Detachment is absent (only tractional RT exists).",
            "Image-level bootstrap 95% CIs may underestimate variance if undetected patient-level clustering exists in RFMiD.",
            "High AUROC does not establish clinical safety or deployability."
        ]
    }

    with open(out_dir / "rfmid_preflight_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"✅ Deliverables successfully written to: {out_dir}")


if __name__ == "__main__":
    main()
