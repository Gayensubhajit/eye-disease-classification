#!/usr/bin/env python3
"""
Pre-Registered Multi-Label External Validation Runner for RFMiD v1.0.

Evaluates the frozen three-seed 9-class EfficientNet-B0 fundus model on the
Retinal Fundus Multi-Disease Image Dataset (RFMiD v1.0; N=3,200).

Methodological Specifications:
- Internal Test Set Quarantine: data_clean/test is NEVER accessed.
- Frozen Checkpoints: Seed 42, Seed 1337, Seed 2026 (weights unchanged).
- Preprocessing: crop_fundus_area(img, thresh=10), Resize(224, 224), ImageNet Normalization.
- Multi-Label Semantics: One-vs-Rest independent binary classification.
- Threshold Protocol: Frozen operating point tau=0.5; threshold-independent AUROC & PR-AUC prioritized.
- Statistical Rigor: B=2,000 stratified bootstrap iterations for 95% percentile CIs.
- Aggregate Metrics: Mean +/- sample SD across seeds 42, 1337, 2026.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Root path configuration
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbone import build_model
from src.data.preprocessing import crop_fundus_area, get_val_transforms

# Quarantine Assertion: Ensure data_clean/test is NEVER referenced or imported
assert not any("data_clean/test" in p for p in sys.path), "Quarantine violation!"

SEEDS = [42, 1337, 2026]
OUTPUT_BASE = ROOT / "outputs/external_evaluation_rfmid"
RFMID_DIR = ROOT / "data_external/rfmid_v1/download_raw"

CLASS_NAMES = [
    "Central Serous Chorioretinopathy [Color Fundus]",  # 0
    "Diabetic Retinopathy",                            # 1
    "Disc Edema",                                      # 2
    "Glaucoma",                                        # 3
    "Healthy",                                         # 4
    "Macular Scar",                                    # 5
    "Myopia",                                          # 6
    "Retinal Detachment",                              # 7
    "Retinitis Pigmentosa",                            # 8
]

MAPPING_SPECS = [
    {
        "internal_class_idx": 1,
        "internal_class_name": "Diabetic Retinopathy",
        "external_label": "DR",
        "ground_truth_key": "DR",
        "mapping_status": "Exact",
        "statistical_power": "Robust (N_pos=632 >= 200)",
        "is_primary": True,
        "notes": "Direct correspondence to non-proliferative and proliferative DR vascular lesions.",
    },
    {
        "internal_class_idx": 4,
        "internal_class_name": "Healthy",
        "external_label": "Disease_Risk == 0 / NL",
        "ground_truth_key": "Normal",
        "mapping_status": "Exact",
        "statistical_power": "Robust (N_pos=669 >= 200)",
        "is_primary": True,
        "notes": "Disease_Risk == 0 indicates absence of any detectable retinal risk/lesions.",
    },
    {
        "internal_class_idx": 5,
        "internal_class_name": "Macular Scar",
        "external_label": "MS",
        "ground_truth_key": "MS",
        "mapping_status": "Exact",
        "statistical_power": "Borderline / Limited (Small N: 20 <= N_pos < 50; Wide CIs)",
        "is_primary": True,
        "notes": "RFMiD MS denotes Macular Scars (27 cases total). Wide bootstrap CIs required.",
    },
    {
        "internal_class_idx": 0,
        "internal_class_name": "Central Serous Chorioretinopathy [Color Fundus]",
        "external_label": "CSR",
        "ground_truth_key": "CSR",
        "mapping_status": "Exact",
        "statistical_power": "Moderate (Adequate Power: 50 <= N_pos < 200)",
        "is_primary": True,
        "notes": "Central Serous Retinopathy / Chorioretinopathy (61 cases total).",
    },
    {
        "internal_class_idx": 2,
        "internal_class_name": "Disc Edema",
        "external_label": "ODE",
        "ground_truth_key": "ODE",
        "mapping_status": "Exact",
        "statistical_power": "Moderate (Adequate Power: 50 <= N_pos < 200)",
        "is_primary": True,
        "notes": "Optic Disc Edema (96 cases total; distinct from Drusen DN).",
    },
    {
        "internal_class_idx": 8,
        "internal_class_name": "Retinitis Pigmentosa",
        "external_label": "RP",
        "ground_truth_key": "RP",
        "mapping_status": "Exact (Underpowered)",
        "statistical_power": "Severely Inadequate (N_pos=10 < 20; Exploratory Only)",
        "is_primary": False,
        "notes": "Severely inadequate sample size (N=10 in 3,200). Descriptive analysis only.",
    },
    {
        "internal_class_idx": 6,
        "internal_class_name": "Myopia",
        "external_label": "MYA",
        "ground_truth_key": "MYA",
        "mapping_status": "Approximate",
        "statistical_power": "Moderate (Adequate Power: 50 <= N_pos < 200)",
        "is_primary": False,
        "notes": "RFMiD MYA denotes Myopic Atrophy (167 cases). Feature-level morphological evaluation.",
    },
    {
        "internal_class_idx": 3,
        "internal_class_name": "Glaucoma",
        "external_label": "ODC",
        "ground_truth_key": "ODC",
        "mapping_status": "Approximate",
        "statistical_power": "Robust (N_pos=445 >= 200; Sign-level only)",
        "is_primary": False,
        "notes": "RFMiD ODC denotes Optic Disc Cupping (an anatomic sign, not clinical glaucoma diagnosis).",
    },
    {
        "internal_class_idx": 7,
        "internal_class_name": "Retinal Detachment",
        "external_label": "None (RT for exploratory only)",
        "ground_truth_key": "RT",
        "mapping_status": "Unavailable",
        "statistical_power": "Non-Equivalent & Underpowered (N_pos=25; Direct Evaluation Unavailable)",
        "is_primary": False,
        "notes": "Direct evaluation UNAVAILABLE. RFMiD RT denotes Retinal Traction Detachment (25 cases).",
    },
]


class RFMiDDataset(Dataset):
    """Dataset for RFMiD v1.0 across all partitions."""

    def __init__(self, samples: List[Dict[str, Any]], transform=None, crop_fundus: bool = True):
        self.samples = samples
        self.transform = transform
        self.crop_fundus = crop_fundus

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        img_path = item["image_path"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_fundus:
            image = crop_fundus_area(image, threshold=10)

        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image_tensor, idx


def collect_rfmid_manifest() -> List[Dict[str, Any]]:
    """Assemble verified manifest of all 3,200 RFMiD images and labels."""
    partitions = [
        ("Training_Set", RFMID_DIR / "Training_Set/Training_Set/RFMiD_Training_Labels.csv", RFMID_DIR / "Training_Set/Training_Set/Training"),
        ("Evaluation_Set", RFMID_DIR / "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv", RFMID_DIR / "Evaluation_Set/Evaluation_Set/Validation"),
        ("Test_Set", RFMID_DIR / "Test_Set/Test_Set/RFMiD_Testing_Labels.csv", RFMID_DIR / "Test_Set/Test_Set/Test"),
    ]

    manifest = []
    for partition_name, csv_path, img_dir in partitions:
        assert csv_path.exists(), f"Missing CSV: {csv_path}"
        assert img_dir.exists(), f"Missing Image Dir: {img_dir}"

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        all_disease_cols = [c for c in df.columns if c not in ["ID", "Disease_Risk"]]

        for _, row in df.iterrows():
            img_id = int(row["ID"])
            img_path = img_dir / f"{img_id}.png"
            assert img_path.exists(), f"Missing image: {img_path}"

            disease_risk = int(row["Disease_Risk"])
            is_normal = 1 if disease_risk == 0 else 0
            num_conditions = int(sum(row[col] for col in all_disease_cols))

            sample = {
                "sample_id": len(manifest),
                "image_id": img_id,
                "partition": partition_name,
                "image_path": img_path,
                "image_rel_path": str(img_path.relative_to(ROOT)),
                "Disease_Risk": disease_risk,
                "Normal": is_normal,
                "DR": int(row["DR"]),
                "ODC": int(row["ODC"]),
                "MYA": int(row["MYA"]),
                "ODE": int(row["ODE"]),
                "CSR": int(row["CSR"]),
                "MS": int(row["MS"]),
                "RT": int(row["RT"]),
                "RP": int(row["RP"]),
                "num_conditions": num_conditions,
            }
            manifest.append(sample)

    assert len(manifest) == 3200, f"Expected 3,200 images, got {len(manifest)}"
    return manifest


def run_inference_all_seeds(
    manifest: List[Dict[str, Any]],
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Execute forward passes for Seed 42, 1337, and 2026 checkpoints in one data pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFERENCE] Using execution device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    transform = get_val_transforms(image_size=224)
    dataset = RFMiDDataset(manifest, transform=transform, crop_fundus=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load all 3 models
    models: Dict[int, torch.nn.Module] = {}
    for seed in SEEDS:
        ckpt_path = ROOT / f"outputs/clean_9class_baseline_effnet/seed_{seed}/best_model.pth"
        assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
        with open(ckpt_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        print(f"  Loaded Seed {seed} checkpoint: {ckpt_path.name} (SHA-256: {h[:16]}...)")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[seed] = model

    results = {
        s: {
            "logits": np.zeros((len(manifest), 9), dtype=np.float32),
            "probs": np.zeros((len(manifest), 9), dtype=np.float32),
            "sig_probs": np.zeros((len(manifest), 9), dtype=np.float32),
        }
        for s in SEEDS
    }

    t0 = time.time()
    processed_count = 0
    print(f"[INFERENCE] Starting forward passes across all {len(manifest)} RFMiD images...")

    with torch.no_grad():
        for batch_imgs, batch_indices in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_idx_arr = batch_indices.numpy()

            for seed in SEEDS:
                logits = models[seed](batch_imgs)
                probs = torch.softmax(logits, dim=1)
                sig_probs = torch.sigmoid(logits)

                results[seed]["logits"][batch_idx_arr] = logits.cpu().numpy()
                results[seed]["probs"][batch_idx_arr] = probs.cpu().numpy()
                results[seed]["sig_probs"][batch_idx_arr] = sig_probs.cpu().numpy()

            processed_count += len(batch_idx_arr)
            if processed_count % 640 == 0 or processed_count == len(manifest):
                elapsed = time.time() - t0
                rate = processed_count / elapsed
                print(f"  Processed {processed_count}/{len(manifest)} images ({rate:.1f} img/s)")

    print(f"[INFERENCE] Inference complete in {time.time() - t0:.2f}s.")
    return results


def compute_binary_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
    tau: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics and 95% stratified bootstrap confidence intervals."""
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    total_count = len(y_true)

    # Operating point at tau=0.5
    y_pred_bin = (y_prob >= tau).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float((2 * prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0
    bal_acc = float((sens + spec) / 2.0)

    # Threshold-independent metrics
    if pos_count > 0 and neg_count > 0:
        auroc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = float("nan")
        pr_auc = float("nan")

    # Exploratory optimal Youden's J
    best_j = -1.0
    opt_tau = 0.5
    opt_sens = sens
    opt_spec = spec
    if pos_count > 0 and neg_count > 0:
        threshold_candidates = np.linspace(0.01, 0.99, 99)
        for cand_tau in threshold_candidates:
            c_pred = (y_prob >= cand_tau).astype(int)
            c_tp = np.sum((y_true == 1) & (c_pred == 1))
            c_fp = np.sum((y_true == 0) & (c_pred == 1))
            c_fn = np.sum((y_true == 1) & (c_pred == 0))
            c_tn = np.sum((y_true == 0) & (c_pred == 0))
            c_sens = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_spec = c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else 0.0
            j = c_sens + c_spec - 1.0
            if j > best_j:
                best_j = j
                opt_tau = float(cand_tau)
                opt_sens = float(c_sens)
                opt_spec = float(c_spec)

    # Stratified Bootstrap
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_auroc = []
    boot_pr_auc = []
    boot_sens = []
    boot_spec = []
    boot_prec = []
    boot_f1 = []
    boot_bal_acc = []

    if pos_count >= 2 and neg_count >= 2 and n_bootstraps > 0:
        for _ in range(n_bootstraps):
            b_pos = rng.choice(pos_idx, size=pos_count, replace=True)
            b_neg = rng.choice(neg_idx, size=neg_count, replace=True)
            b_idx = np.concatenate([b_pos, b_neg])

            b_y_true = y_true[b_idx]
            b_y_prob = y_prob[b_idx]
            b_y_pred = (b_y_prob >= tau).astype(int)

            b_tp = np.sum((b_y_true == 1) & (b_y_pred == 1))
            b_fp = np.sum((b_y_true == 0) & (b_y_pred == 1))
            b_fn = np.sum((b_y_true == 1) & (b_y_pred == 0))
            b_tn = np.sum((b_y_true == 0) & (b_y_pred == 0))

            b_s = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
            b_sp = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0
            b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
            b_f = (2 * b_p * b_s) / (b_p + b_s) if (b_p + b_s) > 0 else 0.0

            boot_sens.append(b_s)
            boot_spec.append(b_sp)
            boot_prec.append(b_p)
            boot_f1.append(b_f)
            boot_bal_acc.append((b_s + b_sp) / 2.0)

            try:
                boot_auroc.append(roc_auc_score(b_y_true, b_y_prob))
            except Exception:
                pass
            try:
                boot_pr_auc.append(average_precision_score(b_y_true, b_y_prob))
            except Exception:
                pass

    def get_ci(vals, default_val):
        if len(vals) < 50:
            return [default_val, default_val]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "n_total": total_count,
        "n_pos": pos_count,
        "n_neg": neg_count,
        "prevalence_pct": round((pos_count / total_count) * 100.0, 3) if total_count > 0 else 0.0,
        "auroc": auroc,
        "auroc_ci95": get_ci(boot_auroc, auroc),
        "pr_auc": pr_auc,
        "pr_auc_ci95": get_ci(boot_pr_auc, pr_auc),
        "frozen_tau": tau,
        "sensitivity": sens,
        "sensitivity_ci95": get_ci(boot_sens, sens),
        "specificity": spec,
        "specificity_ci95": get_ci(boot_spec, spec),
        "precision": prec,
        "precision_ci95": get_ci(boot_prec, prec),
        "f1": f1,
        "f1_ci95": get_ci(boot_f1, f1),
        "balanced_accuracy": bal_acc,
        "balanced_accuracy_ci95": get_ci(boot_bal_acc, bal_acc),
        "confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "exploratory_youden": {
            "optimal_tau": opt_tau,
            "optimal_sensitivity": opt_sens,
            "optimal_specificity": opt_spec,
            "optimal_j": float(best_j),
        },
    }


def main():
    print("=" * 80)
    print("PRE-REGISTERED EXTERNAL VALIDATION: RFMiD v1.0 (N=3,200)")
    print("Three-Seed 9-Class EfficientNet-B0 Baseline")
    print("=" * 80)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    manifest = collect_rfmid_manifest()
    df_manifest = pd.DataFrame(manifest)

    # 1. Run Model Inferences
    inference_results = run_inference_all_seeds(manifest, batch_size=32, num_workers=4)

    # 2. Build and save per-seed prediction ledgers
    per_seed_metrics: Dict[int, Any] = {}

    partitions_to_eval = [
        ("Full_Cohort", list(range(len(manifest)))),
        ("Training_Set", df_manifest[df_manifest["partition"] == "Training_Set"].index.tolist()),
        ("Evaluation_Set", df_manifest[df_manifest["partition"] == "Evaluation_Set"].index.tolist()),
        ("Test_Set", df_manifest[df_manifest["partition"] == "Test_Set"].index.tolist()),
    ]

    for seed in SEEDS:
        seed_dir = OUTPUT_BASE / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n" + "-" * 60)
        print(f"PROCESSING AND EVALUATING SEED {seed}")
        print("-" * 60)

        seed_logits = inference_results[seed]["logits"]
        seed_probs = inference_results[seed]["probs"]
        seed_sig_probs = inference_results[seed]["sig_probs"]

        # Prediction Ledger
        ledger_rows = []
        for i, item in enumerate(manifest):
            row = {
                "sample_id": item["sample_id"],
                "image_id": item["image_id"],
                "partition": item["partition"],
                "image_rel_path": item["image_rel_path"],
                "Disease_Risk": item["Disease_Risk"],
                "Normal": item["Normal"],
                "DR": item["DR"],
                "ODC": item["ODC"],
                "MYA": item["MYA"],
                "ODE": item["ODE"],
                "CSR": item["CSR"],
                "MS": item["MS"],
                "RT": item["RT"],
                "RP": item["RP"],
                "num_conditions": item["num_conditions"],
                "top1_pred_idx": int(np.argmax(seed_probs[i])),
                "top1_pred_class": CLASS_NAMES[np.argmax(seed_probs[i])],
                "top1_max_prob": float(np.max(seed_probs[i])),
            }
            for c_idx, c_name in enumerate(CLASS_NAMES):
                row[f"logit_{c_name}"] = float(seed_logits[i, c_idx])
                row[f"prob_{c_name}"] = float(seed_probs[i, c_idx])
                row[f"sig_prob_{c_name}"] = float(seed_sig_probs[i, c_idx])
            ledger_rows.append(row)

        df_ledger = pd.DataFrame(ledger_rows)
        ledger_csv_path = seed_dir / "rfmid_predictions.csv"
        df_ledger.to_csv(ledger_csv_path, index=False)
        print(f"Saved prediction ledger to {ledger_csv_path} ({len(df_ledger)} rows)")

        # Compute Metrics across Partitions
        seed_partition_metrics = {}
        for part_name, indices in partitions_to_eval:
            print(f"  Evaluating partition: {part_name} (N = {len(indices)})...")
            sub_df = df_manifest.iloc[indices]
            sub_probs = seed_probs[indices]

            class_metrics = {}
            primary_aucs = []
            primary_pr_aucs = []
            primary_f1s = []
            primary_sens = []
            primary_spec = []

            all_aucs = []
            all_pr_aucs = []
            all_f1s = []
            all_sens = []
            all_spec = []

            for spec in MAPPING_SPECS:
                c_idx = spec["internal_class_idx"]
                c_name = spec["internal_class_name"]
                gt_key = spec["ground_truth_key"]
                is_prim = spec["is_primary"]

                y_true = sub_df[gt_key].to_numpy(dtype=int)
                y_prob = sub_probs[:, c_idx]

                m_res = compute_binary_metrics_with_bootstrap(
                    y_true, y_prob, n_bootstraps=2000, seed=seed, tau=0.5
                )
                m_res["mapping_status"] = spec["mapping_status"]
                m_res["statistical_power"] = spec["statistical_power"]
                m_res["is_primary"] = is_prim
                m_res["notes"] = spec["notes"]
                class_metrics[c_name] = m_res

                # Aggregate lists
                if not np.isnan(m_res["auroc"]):
                    all_aucs.append(m_res["auroc"])
                    if is_prim:
                        primary_aucs.append(m_res["auroc"])
                if not np.isnan(m_res["pr_auc"]):
                    all_pr_aucs.append(m_res["pr_auc"])
                    if is_prim:
                        primary_pr_aucs.append(m_res["pr_auc"])
                all_f1s.append(m_res["f1"])
                all_sens.append(m_res["sensitivity"])
                all_spec.append(m_res["specificity"])
                if is_prim:
                    primary_f1s.append(m_res["f1"])
                    primary_sens.append(m_res["sensitivity"])
                    primary_spec.append(m_res["specificity"])

            seed_partition_metrics[part_name] = {
                "partition": part_name,
                "n_samples": len(indices),
                "primary_exact_macro": {
                    "macro_auroc": float(np.mean(primary_aucs)) if primary_aucs else float("nan"),
                    "macro_pr_auc": float(np.mean(primary_pr_aucs)) if primary_pr_aucs else float("nan"),
                    "macro_f1": float(np.mean(primary_f1s)) if primary_f1s else float("nan"),
                    "macro_sensitivity": float(np.mean(primary_sens)) if primary_sens else float("nan"),
                    "macro_specificity": float(np.mean(primary_spec)) if primary_spec else float("nan"),
                },
                "all_mappings_macro": {
                    "macro_auroc": float(np.mean(all_aucs)) if all_aucs else float("nan"),
                    "macro_pr_auc": float(np.mean(all_pr_aucs)) if all_pr_aucs else float("nan"),
                    "macro_f1": float(np.mean(all_f1s)) if all_f1s else float("nan"),
                    "macro_sensitivity": float(np.mean(all_sens)) if all_sens else float("nan"),
                    "macro_specificity": float(np.mean(all_spec)) if all_spec else float("nan"),
                },
                "per_class": class_metrics,
            }

        per_seed_metrics[seed] = seed_partition_metrics
        with open(seed_dir / "external_metrics.json", "w", encoding="utf-8") as f:
            json.dump(seed_partition_metrics, f, indent=2)
        print(f"Saved Seed {seed} metrics to {seed_dir / 'external_metrics.json'}")

    # 3. Multi-Seed Aggregation & Baseline Comparison
    print("\n" + "=" * 80)
    print("COMPUTING MULTI-SEED AGGREGATES & GENERALIZATION GAPS")
    print("=" * 80)

    # Load Internal Validation Baseline
    internal_baselines: Dict[int, Any] = {}
    for seed in SEEDS:
        with open(ROOT / f"outputs/clean_9class_baseline_effnet/seed_{seed}/val_metrics.json", "r") as f:
            internal_baselines[seed] = json.load(f)

    aggregate_summary = {
        "dataset": "RFMiD v1.0",
        "cohort_size": 3200,
        "seeds_evaluated": SEEDS,
        "partitions": {},
    }

    for part_name, _ in partitions_to_eval:
        part_agg: Dict[str, Any] = {
            "partition": part_name,
            "classes": {},
            "primary_exact_macro": {},
            "all_mappings_macro": {},
        }

        # Per-class aggregate across seeds
        for spec in MAPPING_SPECS:
            c_name = spec["internal_class_name"]
            stat_power = spec["statistical_power"]
            mapping_status = spec["mapping_status"]
            is_prim = spec["is_primary"]

            aurocs = [per_seed_metrics[s][part_name]["per_class"][c_name]["auroc"] for s in SEEDS]
            pr_aucs = [per_seed_metrics[s][part_name]["per_class"][c_name]["pr_auc"] for s in SEEDS]
            senss = [per_seed_metrics[s][part_name]["per_class"][c_name]["sensitivity"] for s in SEEDS]
            specs = [per_seed_metrics[s][part_name]["per_class"][c_name]["specificity"] for s in SEEDS]
            f1s = [per_seed_metrics[s][part_name]["per_class"][c_name]["f1"] for s in SEEDS]
            bal_accs = [per_seed_metrics[s][part_name]["per_class"][c_name]["balanced_accuracy"] for s in SEEDS]

            # Internal baseline comparison (from internal val split)
            int_f1s = [internal_baselines[s]["per_class"][c_name]["f1"] for s in SEEDS]
            int_senss = [internal_baselines[s]["per_class"][c_name]["sensitivity"] for s in SEEDS]
            int_specs = [internal_baselines[s]["per_class"][c_name]["specificity"] for s in SEEDS]

            # Average 95% CIs across seeds
            ci_auroc_low = float(np.mean([per_seed_metrics[s][part_name]["per_class"][c_name]["auroc_ci95"][0] for s in SEEDS]))
            ci_auroc_high = float(np.mean([per_seed_metrics[s][part_name]["per_class"][c_name]["auroc_ci95"][1] for s in SEEDS]))
            ci_pr_low = float(np.mean([per_seed_metrics[s][part_name]["per_class"][c_name]["pr_auc_ci95"][0] for s in SEEDS]))
            ci_pr_high = float(np.mean([per_seed_metrics[s][part_name]["per_class"][c_name]["pr_auc_ci95"][1] for s in SEEDS]))

            part_agg["classes"][c_name] = {
                "mapping_status": mapping_status,
                "statistical_power": stat_power,
                "is_primary": is_prim,
                "rfmid_label": spec["external_label"],
                "n_pos": per_seed_metrics[42][part_name]["per_class"][c_name]["n_pos"],
                "n_neg": per_seed_metrics[42][part_name]["per_class"][c_name]["n_neg"],
                "prevalence_pct": per_seed_metrics[42][part_name]["per_class"][c_name]["prevalence_pct"],
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs, ddof=1)),
                "auroc_ci95_mean": [ci_auroc_low, ci_auroc_high],
                "pr_auc_mean": float(np.mean(pr_aucs)),
                "pr_auc_std": float(np.std(pr_aucs, ddof=1)),
                "pr_auc_ci95_mean": [ci_pr_low, ci_pr_high],
                "frozen_tau_0_5": {
                    "sensitivity_mean": float(np.mean(senss)),
                    "sensitivity_std": float(np.std(senss, ddof=1)),
                    "specificity_mean": float(np.mean(specs)),
                    "specificity_std": float(np.std(specs, ddof=1)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s, ddof=1)),
                    "balanced_accuracy_mean": float(np.mean(bal_accs)),
                    "balanced_accuracy_std": float(np.std(bal_accs, ddof=1)),
                },
                "internal_val_baseline": {
                    "f1_mean": float(np.mean(int_f1s)),
                    "f1_std": float(np.std(int_f1s, ddof=1)),
                    "sensitivity_mean": float(np.mean(int_senss)),
                    "sensitivity_std": float(np.std(int_senss, ddof=1)),
                    "specificity_mean": float(np.mean(int_specs)),
                    "specificity_std": float(np.std(int_specs, ddof=1)),
                },
                "generalization_gap": {
                    "delta_f1": float(np.mean(f1s) - np.mean(int_f1s)),
                    "delta_sensitivity": float(np.mean(senss) - np.mean(int_senss)),
                    "delta_specificity": float(np.mean(specs) - np.mean(int_specs)),
                },
            }

        # Macro metrics across seeds
        for macro_key in ["primary_exact_macro", "all_mappings_macro"]:
            for m_field in ["macro_auroc", "macro_pr_auc", "macro_f1", "macro_sensitivity", "macro_specificity"]:
                vals = [per_seed_metrics[s][part_name][macro_key][m_field] for s in SEEDS]
                part_agg[macro_key][f"{m_field}_mean"] = float(np.mean(vals))
                part_agg[macro_key][f"{m_field}_std"] = float(np.std(vals, ddof=1))

        aggregate_summary["partitions"][part_name] = part_agg

    # Save Aggregate JSON
    agg_json_path = OUTPUT_BASE / "aggregate_results.json"
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)
    print(f"\nSaved multi-seed aggregate results to {agg_json_path}")

    # Build CSV summary for the Full Cohort
    summary_rows = []
    fc_classes = aggregate_summary["partitions"]["Full_Cohort"]["classes"]
    for c_name, c_data in fc_classes.items():
        summary_rows.append({
            "Disease Class": c_name,
            "Mapping Status": c_data["mapping_status"],
            "External Label": c_data["rfmid_label"],
            "Support (Pos/Total)": f"{c_data['n_pos']}/{c_data['n_pos'] + c_data['n_neg']}",
            "Prevalence (%)": f"{c_data['prevalence_pct']:.2f}%",
            "AUROC (Mean +/- SD)": f"{c_data['auroc_mean']*100:.2f}% +/- {c_data['auroc_std']*100:.2f}%",
            "AUROC 95% CI": f"[{c_data['auroc_ci95_mean'][0]*100:.2f}%, {c_data['auroc_ci95_mean'][1]*100:.2f}%]",
            "PR-AUC (Mean +/- SD)": f"{c_data['pr_auc_mean']*100:.2f}% +/- {c_data['pr_auc_std']*100:.2f}%",
            "PR-AUC 95% CI": f"[{c_data['pr_auc_ci95_mean'][0]*100:.2f}%, {c_data['pr_auc_ci95_mean'][1]*100:.2f}%]",
            "Ext Sensitivity (tau=0.5)": f"{c_data['frozen_tau_0_5']['sensitivity_mean']*100:.2f}% +/- {c_data['frozen_tau_0_5']['sensitivity_std']*100:.2f}%",
            "Ext Specificity (tau=0.5)": f"{c_data['frozen_tau_0_5']['specificity_mean']*100:.2f}% +/- {c_data['frozen_tau_0_5']['specificity_std']*100:.2f}%",
            "Ext F1 (tau=0.5)": f"{c_data['frozen_tau_0_5']['f1_mean']*100:.2f}% +/- {c_data['frozen_tau_0_5']['f1_std']*100:.2f}%",
            "Int Val F1 (Internal Baseline)": f"{c_data['internal_val_baseline']['f1_mean']*100:.2f}% +/- {c_data['internal_val_baseline']['f1_std']*100:.2f}%",
            "F1 Generalization Gap": f"{c_data['generalization_gap']['delta_f1']*100:+.2f}%",
            "Statistical Power": c_data["statistical_power"],
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_BASE / "external_evaluation_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved CSV tabular summary to {summary_csv_path}")

    # Print Full Cohort Aggregate Report to Terminal
    print("\n" + "=" * 90)
    print("RFMiD v1.0 FULL COHORT (N=3,200) EXTERNAL VALIDATION SUMMARY")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # Macro averages
    fc_prim = aggregate_summary["partitions"]["Full_Cohort"]["primary_exact_macro"]
    fc_all = aggregate_summary["partitions"]["Full_Cohort"]["all_mappings_macro"]
    print("\nPRIMARY EXACT MAPPINGS MACRO METRICS (DR, Healthy, MS, CSR, ODE):")
    print(f"  Macro-AUROC:        {fc_prim['macro_auroc_mean']*100:.2f}% +/- {fc_prim['macro_auroc_std']*100:.2f}%")
    print(f"  Macro-PR-AUC:       {fc_prim['macro_pr_auc_mean']*100:.2f}% +/- {fc_prim['macro_pr_auc_std']*100:.2f}%")
    print(f"  Macro-F1 (tau=0.5): {fc_prim['macro_f1_mean']*100:.2f}% +/- {fc_prim['macro_f1_std']*100:.2f}%")
    print(f"  Macro-Sensitivity:  {fc_prim['macro_sensitivity_mean']*100:.2f}% +/- {fc_prim['macro_sensitivity_std']*100:.2f}%")
    print(f"  Macro-Specificity:  {fc_prim['macro_specificity_mean']*100:.2f}% +/- {fc_prim['macro_specificity_std']*100:.2f}%")

    print("\nALL EVALUABLE MAPPINGS MACRO METRICS (Including Approximate/Exploratory):")
    print(f"  Macro-AUROC:        {fc_all['macro_auroc_mean']*100:.2f}% +/- {fc_all['macro_auroc_std']*100:.2f}%")
    print(f"  Macro-PR-AUC:       {fc_all['macro_pr_auc_mean']*100:.2f}% +/- {fc_all['macro_pr_auc_std']*100:.2f}%")
    print(f"  Macro-F1 (tau=0.5): {fc_all['macro_f1_mean']*100:.2f}% +/- {fc_all['macro_f1_std']*100:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
