"""Deterministic preflight verification script for the 9-class fundus cohort.

Verifies:
1. Exact row counts and split partitions (2,612 train, 560 val, 561 test = 3,733 total).
2. Preservation of source-group IDs, archive paths, and hashes against canonical V4 inventory.
3. Class-to-index consistency across train, val, and test loaders.
4. Total absence of Pterygium records and samples.
5. Strict quarantine: Test DataLoader is NEVER iterated.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from src.data.dataset import create_dataloaders


def main() -> int:
    print("=== Step 1: Cross-Checking Manifest Against Canonical V4 Inventory ===")
    inv_path = ROOT / "outputs/audit/data_clean_v4_inventory.csv"
    manifest_path = ROOT / "outputs/audit/fundus_9class_split_manifest.csv"
    pte_path = ROOT / "outputs/audit/excluded_pterygium_anterior_images.csv"

    if not inv_path.exists():
        print(f"ERROR: Inventory file missing: {inv_path}")
        return 1
    if not manifest_path.exists() or not pte_path.exists():
        print(f"ERROR: 9-class or Pterygium manifest missing!")
        return 1

    inv = pd.read_csv(inv_path)
    m9 = pd.read_csv(manifest_path)
    pte = pd.read_csv(pte_path)

    assert len(inv) == 3747, f"Expected 3747 in canonical inventory, got {len(inv)}"
    assert len(m9) == 3733, f"Expected 3733 in 9-class manifest, got {len(m9)}"
    assert len(pte) == 14, f"Expected 14 in Pterygium manifest, got {len(pte)}"
    assert len(m9) + len(pte) == 3747, "Sum of cohorts does not equal 3,747!"

    # Verify split distributions
    splits_m9 = m9["split"].value_counts().to_dict()
    assert splits_m9.get("train") == 2612, f"Train split count error: {splits_m9}"
    assert splits_m9.get("val") == 560, f"Val split count error: {splits_m9}"
    assert splits_m9.get("test") == 561, f"Test split count error: {splits_m9}"

    # Verify zero Pterygium in 9-class manifest
    assert (m9["class"] == "Pterygium").sum() == 0, "Pterygium found in 9-class manifest!"
    assert m9["clean_image_path"].nunique() == 3733, "Duplicate clean_image_path in 9-class manifest!"

    # Verify metadata preservation against inventory
    m9_inv_merged = pd.merge(m9, inv, on=["clean_image_path", "split", "class", "source_group_id", "md5", "sha256"], how="inner")
    assert len(m9_inv_merged) == 3733, f"Metadata discrepancy between 9-class manifest and canonical inventory! Matched: {len(m9_inv_merged)}"
    print("PASS: Manifest cross-check against canonical V4 inventory passed with zero discrepancies.")

    print("\n=== Step 2: Dataloader Configuration-Only Preflight ===")
    config_path = ROOT / "configs/clean_9class_fundus_baseline_efficientnet_b0.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset

    assert len(train_ds) == 2612, f"Train DataLoader length mismatch: {len(train_ds)}"
    assert len(val_ds) == 560, f"Val DataLoader length mismatch: {len(val_ds)}"
    assert len(test_ds) == 561, f"Test DataLoader length mismatch: {len(test_ds)}"

    # Check class mappings
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, "Class index mappings differ across splits!"
    assert "Pterygium" not in train_ds.class_to_idx, "Pterygium present in class index map!"
    assert len(train_ds.class_names) == 9, f"Class names count is not 9: {len(train_ds.class_names)}"

    # Check individual samples in memory
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pte_count = sum(1 for s in ds.samples if "pterygium" in str(s[0]).lower())
        assert pte_count == 0, f"Found {pte_count} Pterygium samples in {name} dataset!"

    print("PASS: DataLoader preflight confirmed 2612 / 560 / 561 samples, identical class mappings, and zero Pterygium images.")
    print("QUARANTINE ENFORCED: Test DataLoader was constructed but NOT iterated or evaluated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
