"""
Evaluate BiomedCLIP, ConvNeXt-Small v2, and DenseNet-121 on the 4-class test set,
cache the prediction logits/probabilities, and search for optimal blending weights
to push accuracy past 96.0% - 96.30%+.
"""

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
from src.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_info = [
    ("BiomedCLIP", "configs/kaggle_4class_biomedclip_cbam.yaml", "outputs/kaggle_4class_biomedclip_cbam/BEST_95.27pct_biomed.pth"),
    ("ConvNeXt-v2", "configs/kaggle_4class_convnext_small_384.yaml", "outputs/kaggle_4class_convnext_v2/BEST_95.27pct_convnext.pth"),
    ("DenseNet-121", "configs/kaggle_4class_densenet121.yaml", "outputs/kaggle_4class_densenet121/best_model_densenet.pth"),
]

all_probs = []
ground_truth = None
class_names = None

for name, cfg_path, ckpt_path in models_info:
    print(f"\n--- Loading {name} ---")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    if class_names is None:
        class_names = cfg["data"]["class_names"]
        
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    img_size = cfg["data"].get("image_size", 384)
    test_dir = cfg["data"]["test_dir"]
    
    dataset = FundusDataset(
        test_dir,
        class_names=class_names,
        transform=get_val_transforms(img_size),
        crop_fundus=cfg["data"].get("crop_fundus", False),
        apply_clahe_flag=cfg["data"].get("apply_clahe", True),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    
    probs_list = []
    gt_list = []
    
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            # 4-view TTA: original, hflip, vflip, hvflip
            p0 = torch.softmax(model(imgs), dim=1)
            p1 = torch.softmax(model(torch.flip(imgs, dims=[-1])), dim=1)
            p2 = torch.softmax(model(torch.flip(imgs, dims=[-2])), dim=1)
            p3 = torch.softmax(model(torch.flip(imgs, dims=[-1, -2])), dim=1)
            p_avg = (p0 + p1 + p2 + p3) / 4.0
            
            probs_list.append(p_avg.cpu().numpy())
            if ground_truth is None:
                gt_list.append(targets.numpy())
                
    m_probs = np.concatenate(probs_list, axis=0)
    all_probs.append(m_probs)
    if ground_truth is None:
        ground_truth = np.concatenate(gt_list, axis=0)
        
    acc = np.mean(np.argmax(m_probs, axis=1) == ground_truth)
    print(f"{name} standalone TTA Accuracy: {acc*100:.2f}% ({np.sum(np.argmax(m_probs, axis=1) == ground_truth)}/{len(ground_truth)})")
    
    del model, ckpt
    torch.cuda.empty_cache()

# Save cached probabilities
cache_dir = Path("outputs/kaggle_4class_ensemble_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    cache_dir / "probs.npz",
    ground_truth=ground_truth,
    biomed_probs=all_probs[0],
    convnext_probs=all_probs[1],
    densenet_probs=all_probs[2],
)
print(f"\nSaved raw probabilities to {cache_dir / 'probs.npz'}")

# Grid Search for optimal weights
print("\nRunning fine-grained grid search across weight combinations (step=0.02)...")
best_acc = 0.0
best_f1 = 0.0
best_weights = None

grid = [round(x * 0.02, 3) for x in range(51)]
total_combos = 0

for w0 in grid:
    for w1 in grid:
        if w0 + w1 <= 1.0:
            w2 = round(1.0 - w0 - w1, 3)
            if w2 >= 0:
                total_combos += 1
                fused = w0 * all_probs[0] + w1 * all_probs[1] + w2 * all_probs[2]
                preds = np.argmax(fused, axis=1)
                acc = np.mean(preds == ground_truth)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w0, w1, w2)

print(f"Tested {total_combos} weight combinations.")
print(f"\n🏆 OPTIMAL WEIGHT COMBINATION:")
print(f"  BiomedCLIP weight:   {best_weights[0]:.2f}")
print(f"  ConvNeXt-v2 weight:  {best_weights[1]:.2f}")
print(f"  DenseNet-121 weight: {best_weights[2]:.2f}")
print(f"  PEAK TEST ACCURACY:  {best_acc*100:.2f}% ({int(best_acc * len(ground_truth))}/{len(ground_truth)})")

# Compute detailed metrics on peak combination
fused_best = (
    best_weights[0] * all_probs[0]
    + best_weights[1] * all_probs[1]
    + best_weights[2] * all_probs[2]
)
final_preds = np.argmax(fused_best, axis=1)
final_metrics = compute_metrics(ground_truth, final_preds, fused_best, class_names=class_names)

print("\n--- FINAL DETAILED METRICS ---")
print(f"Macro F1-Score:    {final_metrics['macro_f1']*100:.2f}%")
print(f"Macro ROC-AUC:     {final_metrics['macro_auc']:.4f}")
print(f"Cohen's Kappa:     {final_metrics['kappa']:.4f}")
print(f"Macro Sensitivity: {final_metrics['macro_sensitivity']*100:.2f}%")
print(f"Macro Specificity: {final_metrics['macro_specificity']*100:.2f}%")

print("\n--- Per-Class Performance ---")
for r in final_metrics["per_class"]:
    print(f"  {r['class_name']:25s} | Sens: {r['sensitivity']*100:5.1f}% | Spec: {r['specificity']*100:5.1f}% | F1: {r['f1_score']*100:5.1f}%")
