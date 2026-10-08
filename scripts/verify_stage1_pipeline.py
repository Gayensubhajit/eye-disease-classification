#!/usr/bin/env python3
"""
Stage 1 Pipeline Sanity Verification Script

Validates:
1. PyTorch Dataset loading on data_clean/ (2622 train, 562 val, 563 test).
2. Deterministic class mapping across all splits.
3. Metric computation sanity test with unambiguous signature and explicit assertions.
4. GPU forward, loss, backward, and optimizer execution with explicit torch.isfinite() checks.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data.dataset import FundusDataset
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.metrics import compute_metrics

def main():
    print("=== Running Stage 1 Pipeline Sanity Verification ===")

    # 1. Dataset Split Counts & Deterministic Class Mapping
    class_names = [
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
    ]

    train_dir = os.path.join(REPO_ROOT, "data_clean", "train")
    val_dir = os.path.join(REPO_ROOT, "data_clean", "val")
    test_dir = os.path.join(REPO_ROOT, "data_clean", "test")

    train_ds = FundusDataset(train_dir, class_names=class_names, transform=get_train_transforms(224))
    val_ds = FundusDataset(val_dir, class_names=class_names, transform=get_val_transforms(224))
    test_ds = FundusDataset(test_dir, class_names=class_names, transform=get_val_transforms(224))

    print(f"1. Dataset Split Counts:")
    print(f"   Train samples: {len(train_ds)} (Expected: 2622)")
    print(f"   Val samples:   {len(val_ds)} (Expected: 562)")
    print(f"   Test samples:  {len(test_ds)} (Expected: 563)")
    print(f"   Total samples: {len(train_ds) + len(val_ds) + len(test_ds)} (Expected: 3747)")

    assert len(train_ds) == 2622, f"Train count mismatch: {len(train_ds)}"
    assert len(val_ds) == 562, f"Val count mismatch: {len(val_ds)}"
    assert len(test_ds) == 563, f"Test count mismatch: {len(test_ds)}"

    expected_mapping = {name: idx for idx, name in enumerate(class_names)}
    assert train_ds.class_to_idx == expected_mapping, "Class mapping mismatch in train_ds!"
    assert val_ds.class_to_idx == expected_mapping, "Class mapping mismatch in val_ds!"
    assert test_ds.class_to_idx == expected_mapping, "Class mapping mismatch in test_ds!"
    print("   Class-to-index mapping: EXACT 10-CLASS ALIGNMENT VERIFIED.")

    # 2. Metric Sanity Test with Unambiguous Signature & Explicit Assertions
    print("\n2. Metric Sanity Test (compute_metrics):")
    sample_targets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sample_preds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sample_probs = np.eye(10, dtype=np.float32)

    # Calling compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics = compute_metrics(
        y_true=sample_targets,
        y_pred=sample_preds,
        y_prob=sample_probs,
        class_names=class_names
    )

    print(f"   Calculated accuracy: {metrics['accuracy']}")
    print(f"   Calculated macro_f1: {metrics['macro_f1']}")
    print(f"   Calculated balanced_acc: {metrics['balanced_accuracy']}")

    assert metrics['accuracy'] == 1.0, f"Expected accuracy 1.0, got {metrics['accuracy']}"
    assert metrics['macro_f1'] == 1.0, f"Expected macro_f1 1.0, got {metrics['macro_f1']}"
    assert metrics['balanced_accuracy'] == 1.0, f"Expected balanced_accuracy 1.0, got {metrics['balanced_accuracy']}"
    print("   Metric sanity assertions: ALL PASSED (Accuracy=1.0, Macro-F1=1.0).")

    # 3. GPU Forward, Loss, Backward & Explicit isfinite Checks
    print("\n3. GPU Forward, Loss, Backward & isfinite Assertions:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    assert torch.cuda.is_available(), "CUDA device required for training verification!"

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        assert torch.isfinite(outputs).all(), "Non-finite outputs (NaN/Inf) detected in logits!"

        # Loss computation
        loss = criterion(outputs, targets)
        assert torch.isfinite(loss), f"Non-finite loss detected: {loss.item()}"

        # Backward pass
        loss.backward()

        # Gradient finiteness assertions
        finite_grad_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in parameter: {name}"
                finite_grad_count += 1

        assert finite_grad_count > 0, "No gradients computed during backward pass!"

        # Optimizer step
        optimizer.step()
        print(f"   Sanity Step Forward/Backward OK:")
        print(f"     - Batch Size: {len(targets)}")
        print(f"     - Output Logits Shape: {list(outputs.shape)}")
        print(f"     - Loss Value: {loss.item():.4f}")
        print(f"     - Finite Parameter Gradients Verified: {finite_grad_count}/{finite_grad_count}")
        break

    print("\n=== Stage 1 Sanity Verification: ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
