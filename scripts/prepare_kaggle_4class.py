"""
Prepare stratified 70/20/10 train/val/test splits for the Kaggle 4-class Eye Disease dataset.
"""

import os
import shutil
import random
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_splits(
    src_dir: str = "/home/silentbyte/.cache/kagglehub/datasets/gunavenkatdoddi/eye-diseases-classification/versions/1/dataset",
    dest_dir: str = "data_kaggle_4class",
    seed: int = 42
):
    random.seed(seed)
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    classes = sorted([d.name for d in src_path.iterdir() if d.is_dir()])
    print(f"Discovered {len(classes)} classes: {classes}")

    for split in ["train", "val", "test"]:
        for c in classes:
            (dest_path / split / c).mkdir(parents=True, exist_ok=True)

    summary = {}
    for c in classes:
        all_imgs = sorted([
            p for p in (src_path / c).iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        # 70% train, 30% temp
        train_imgs, temp_imgs = train_test_split(all_imgs, test_size=0.30, random_state=seed, shuffle=True)
        # 20% val, 10% test of total (2/3 of 30% = 20%, 1/3 of 30% = 10%)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(1/3), random_state=seed, shuffle=True)

        summary[c] = {
            "total": len(all_imgs),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        for img in train_imgs:
            shutil.copy2(img, dest_path / "train" / c / img.name)
        for img in val_imgs:
            shutil.copy2(img, dest_path / "val" / c / img.name)
        for img in test_imgs:
            shutil.copy2(img, dest_path / "test" / c / img.name)

    print("\nSplit Summary:")
    print(f"{'Class':25s} | {'Train (70%)':12s} | {'Val (20%)':10s} | {'Test (10%)':10s} | {'Total':6s}")
    print("-" * 75)
    tot_tr, tot_v, tot_ts = 0, 0, 0
    for c, counts in summary.items():
        print(f"{c:25s} | {counts['train']:12d} | {counts['val']:10d} | {counts['test']:10d} | {counts['total']:6d}")
        tot_tr += counts['train']
        tot_v += counts['val']
        tot_ts += counts['test']
    print("-" * 75)
    print(f"{'TOTAL':25s} | {tot_tr:12d} | {tot_v:10d} | {tot_ts:10d} | {tot_tr+tot_v+tot_ts:6d}")

if __name__ == "__main__":
    prepare_splits()
