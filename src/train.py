"""Training entry point for fundus eye disease classification models."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml

from src.data.dataset import create_dataloaders
from src.models.backbone import build_model
from src.losses.focal_loss import get_loss_function
from src.metrics import compute_metrics, plot_confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Set global random seed for full experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(targets)
        num_samples += len(targets)

        if dry_run and step >= 2:
            break

    return total_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
    dry_run: bool = False,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation over a dataset loader."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * len(targets)
        num_samples += len(targets)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if dry_run and step >= 2:
            break

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / max(num_samples, 1)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=class_names)

    return avg_loss, metrics, y_true, y_pred, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fundus classification model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Perform fast sanity check run")
    parser.add_argument("--evaluate-test", action="store_true", default=None, help="Force evaluation on test set")
    parser.add_argument("--no-test-eval", action="store_true", default=False, help="Disable evaluation on test set")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select compute device
    device_setting = config["training"].get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    print(f"Using device: {device}")

    # Build DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    class_names = config["data"]["class_names"]

    print(f"Loaded folder-based dataset: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples across {len(class_names)} classes.")

    # Compute class counts for loss
    train_labels = train_loader.dataset.labels
    class_counts = np.bincount(train_labels, minlength=len(class_names))

    # Build Model, Loss, Optimizer
    model = build_model(config).to(device)
    criterion = get_loss_function(config, class_counts=class_counts).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    epochs = 2 if args.dry_run else int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1 = -1.0
    best_model_path = output_dir / "best_model.pth"
    history = []

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, dry_run=args.dry_run
        )

        val_loss, val_metrics, val_true, val_pred, _ = evaluate_epoch(
            model, val_loader, criterion, device, class_names, dry_run=args.dry_run
        )

        scheduler.step()

        elapsed = time.time() - epoch_start
        macro_f1 = val_metrics["macro_f1"]
        acc = val_metrics["accuracy"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {acc:.4f} | Val Macro F1: {macro_f1:.4f}"
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        history.append(log_entry)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )
            plot_confusion_matrix(
                val_true,
                val_pred,
                class_names,
                str(output_dir / "best_val_confusion_matrix.png"),
                title=f"Best Val Confusion Matrix (Epoch {epoch})",
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    print(f"Best Val Macro F1: {best_macro_f1:.4f} (Saved to {best_model_path})")

    # Save history log
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    import hashlib
    sha256 = hashlib.sha256()
    with open(best_model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest()
    print(f"Best Checkpoint SHA-256: {ckpt_hash}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    # Determine whether test set evaluation is enabled
    if args.no_test_eval:
        evaluate_test = False
    elif args.evaluate_test is not None:
        evaluate_test = args.evaluate_test
    else:
        evaluate_test = config.get("training", {}).get("evaluate_test", False)

    if evaluate_test:
        print("\n--- Running Final Independent Test Evaluation (Best Checkpoint) ---")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_metrics, test_true, test_pred, test_probs = evaluate_epoch(
            model, test_loader, criterion, device, class_names, dry_run=args.dry_run
        )
        print(
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}"
        )
        test_results = {
            "best_epoch": checkpoint["epoch"],
            "checkpoint_path": str(best_model_path),
            "checkpoint_sha256": ckpt_hash,
            "val_macro_f1_at_best_epoch": float(best_macro_f1),
            "test_loss": float(test_loss),
            "test_metrics": test_metrics,
            "device": str(device),
            "training_time_minutes": float(total_time / 60.0),
        }
        with open(output_dir / "test_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

        plot_confusion_matrix(
            test_true,
            test_pred,
            class_names,
            str(output_dir / "test_confusion_matrix.png"),
            title=f"Test Confusion Matrix (Best Model from Epoch {checkpoint['epoch']})",
        )
    else:
        print("\n[NOTE] Test set evaluation is DISABLED for this experiment to preserve test partition integrity.")
        val_summary = {
            "best_epoch": checkpoint["epoch"],
            "checkpoint_path": str(best_model_path),
            "checkpoint_sha256": ckpt_hash,
            "val_macro_f1_at_best_epoch": float(best_macro_f1),
            "val_metrics_at_best_epoch": checkpoint["val_metrics"],
            "device": str(device),
            "training_time_minutes": float(total_time / 60.0),
            "test_evaluated": False,
        }
        with open(output_dir / "val_evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(val_summary, f, indent=2)


if __name__ == "__main__":
    main()
