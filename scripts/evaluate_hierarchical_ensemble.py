"""
Hierarchical Cascaded Classifier:
Routes ambiguous samples in the Glaucoma-Healthy-Myopia triangle to a dedicated fine-grained specialist.
"""
import os
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FundusDataset, get_val_transforms
from src.models.backbone import build_model
from src.metrics import compute_metrics, plot_confusion_matrix
from scripts.evaluate_ms_tta import predict_multiscale_tta

def run_hierarchical_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load 5-model cached probabilities for 10 classes
    cache_file = "outputs/cached_10class_5model_probs.npz"
    if not os.path.exists(cache_file):
        raise FileNotFoundError("Missing cached 5-model probabilities.")
    
    data = np.load(cache_file)
    test_targets = data["test_targets"] # (600,)
    class_names = [str(c) for c in data["class_names"]]

    # Best 4-model linear blend weights
    w = np.array([0.143, 0.429, 0.286, 0.143])
    m_keys = ["convnext", "effnet", "resnet", "biomed"]
    base_fused = sum(w[i] * data[f"test_{m_keys[i]}"] for i in range(4)) # (600, 10)
    base_preds = base_fused.argmax(axis=1)

    base_acc = np.mean(base_preds == test_targets)
    print(f"Base 10-Class Ensemble Test Accuracy: {base_acc*100:.2f}% ({int(round(base_acc*600))}/600)")

    # 2. Check specialist checkpoint
    spec_cfg_path = "configs/specialist_glaucoma_healthy_myopia.yaml"
    spec_ckpt_path = "outputs/specialist_glaucoma_healthy_myopia/best_model.pth"
    if not os.path.exists(spec_ckpt_path):
        print(f"\n[Notice] Specialist checkpoint not found at {spec_ckpt_path}.")
        print("Please train the specialist model first.")
        return

    with open(spec_cfg_path) as f:
        spec_cfg = yaml.safe_load(f)

    spec_classes = spec_cfg["data"]["class_names"] # ['Glaucoma', 'Healthy', 'Myopia']
    spec_indices = [class_names.index(c) for c in spec_classes]
    g_idx, h_idx, m_idx = spec_indices

    print(f"\nLoading Specialist Model from {spec_ckpt_path}...")
    ckpt = torch.load(spec_ckpt_path, map_location=device, weights_only=False)
    spec_model = build_model(spec_cfg).to(device)
    spec_model.load_state_dict(ckpt["model_state_dict"])
    spec_model.eval()

    # 3. Evaluate specialist with MS-TTA on all 600 test images
    val_transforms = get_val_transforms(spec_cfg["data"]["image_size"])
    test_dataset = FundusDataset(
        "data/test",
        class_names=class_names, # load all 600 images in same order
        transform=val_transforms,
        crop_fundus=spec_cfg["data"].get("crop_fundus", True),
        apply_clahe_flag=spec_cfg["data"].get("apply_clahe", True),
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    spec_probs_list = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            p = predict_multiscale_tta(spec_model, images, scales=[1.0, 1.15], use_flips=True, is_vit=False)
            spec_probs_list.append(p.cpu().numpy())

    spec_probs = np.concatenate(spec_probs_list) # (600, 3)

    # 4. Hierarchical Cascading:
    # If base ensemble predicted Glaucoma, Healthy, or Myopia, let specialist refine among the three!
    final_preds = base_preds.copy()
    rerouted_count = 0
    improved_count = 0

    for i in range(len(test_targets)):
        pred_c = base_preds[i]
        if pred_c in spec_indices:
            rerouted_count += 1
            # Specialist votes on [Glaucoma, Healthy, Myopia]
            spec_local_pred = np.argmax(spec_probs[i])
            spec_global_pred = spec_indices[spec_local_pred]
            
            # If specialist is confident (prob > 0.45), reassign
            if spec_probs[i][spec_local_pred] > 0.45:
                if final_preds[i] != test_targets[i] and spec_global_pred == test_targets[i]:
                    improved_count += 1
                final_preds[i] = spec_global_pred

    final_acc = np.mean(final_preds == test_targets)
    print("\n" + "="*70)
    print("           HIERARCHICAL CASCADED ENSEMBLE RESULTS           ")
    print("="*70)
    print(f"Total test images rerouted to specialist: {rerouted_count}")
    print(f"Total error cases corrected by specialist: {improved_count}")
    print(f"Base Ensemble Accuracy:         {base_acc*100:.2f}% ({int(round(base_acc*600))}/600)")
    print(f"Hierarchical Ensemble Accuracy: {final_acc*100:.2f}% ({int(round(final_acc*600))}/600)")

    # Compute full metrics
    metrics = compute_metrics(test_targets, final_preds, class_names=class_names)
    print(f"Macro F1-Score:                 {metrics['macro_f1']*100:.2f}%")
    print(f"Macro Sensitivity:              {metrics['macro_sensitivity']*100:.2f}%")
    print(f"Macro Specificity:              {metrics['macro_specificity']*100:.2f}%")
    print(f"Cohen's Kappa:                  {metrics['kappa']:.4f}")

    # Literature benchmark check
    target_sota = 0.9386
    print("\nBenchmark Comparison against IEEE Access 2026 (93.86%):")
    if final_acc > target_sota:
        print(f"  >>> ✅ BEATEN! New Accuracy: {final_acc*100:.2f}% vs Benchmark: 93.86% (+{final_acc*100 - 93.86:+.2f}%)")
    else:
        print(f"  >>> Current Accuracy: {final_acc*100:.2f}% vs Benchmark: 93.86% ({final_acc*100 - 93.86:+.2f}%)")
    print("="*70)

    # Save output
    out_dir = Path("outputs/hierarchical_cascaded_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(
        test_targets, final_preds, class_names,
        out_dir / "hierarchical_confusion_matrix.png",
        title=f"Hierarchical Cascaded 10-Class Confusion Matrix (Acc: {final_acc*100:.2f}%)"
    )
    print(f"Saved artifacts to {out_dir}/")

if __name__ == "__main__":
    run_hierarchical_eval()
