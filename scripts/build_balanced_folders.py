"""Script to build physical, equally balanced train/val/test folders (no CSV files).

Creates:
  data/train/<class_name>/ -> 400 images per class (4,000 total)
  data/val/<class_name>/   -> 50 images per class  (500 total)
  data/test/<class_name>/  -> 50 images per class  (500 total)
"""

import shutil
import random
from pathlib import Path
from typing import List
import cv2
import numpy as np
import albumentations as A


def get_augmenter() -> A.Compose:
    """Return augmentation pipeline for synthesizing images for minority classes."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.7),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.5),
    ])


def build_balanced_dataset(
    original_dataset_dir: str = "Eye Disease Image Dataset/Original Dataset/Original Dataset",
    augmented_dataset_dir: str = "Eye Disease Image Dataset/Augmented Dataset/Augmented Dataset",
    output_base_dir: str = "data",
    train_per_class: int = 400,
    val_per_class: int = 50,
    test_per_class: int = 50,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    orig_path = Path(original_dataset_dir)
    aug_path = Path(augmented_dataset_dir)
    out_path = Path(output_base_dir)

    total_target = train_per_class + val_per_class + test_per_class  # 500

    class_dirs = sorted([d for d in orig_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes to balance.")

    augmenter = get_augmenter()

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # Collect original images
        orig_images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))

        # Collect pre-augmented images if available
        aug_class_dir = aug_path / class_name
        aug_images = []
        if aug_class_dir.exists() and aug_class_dir.is_dir():
            aug_images = sorted(list(aug_class_dir.glob("*.jpg")) + list(aug_class_dir.glob("*.png")))

        print(f"  Found {len(orig_images)} original, {len(aug_images)} pre-augmented.")

        all_pool: List[Path] = orig_images.copy()
        # Add pre-augmented images that aren't already duplicates
        all_pool.extend([img for img in aug_images if img.name not in {o.name for o in orig_images}])

        random.shuffle(all_pool)

        selected_images: List[np.ndarray] = []

        # Load available images up to target
        for img_p in all_pool[:total_target]:
            img = cv2.imread(str(img_p))
            if img is not None:
                selected_images.append(img)

        # If still fewer than total_target, synthesize using Albumentations
        if len(selected_images) < total_target:
            needed = total_target - len(selected_images)
            print(f"  Synthesizing {needed} additional augmented images for {class_name}...")
            source_images = selected_images.copy()
            idx = 0
            while len(selected_images) < total_target:
                base_img = source_images[idx % len(source_images)]
                aug_img = augmenter(image=base_img)["image"]
                selected_images.append(aug_img)
                idx += 1

        random.shuffle(selected_images)

        # Split into train, val, test
        train_imgs = selected_images[:train_per_class]
        val_imgs = selected_images[train_per_class : train_per_class + val_per_class]
        test_imgs = selected_images[train_per_class + val_per_class : total_target]

        splits_data = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        for split_name, imgs in splits_data.items():
            split_class_dir = out_path / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(imgs):
                out_file = split_class_dir / f"img_{i:04d}.jpg"
                cv2.imwrite(str(out_file), img)

            print(f"  Saved {len(imgs)} images to {split_name}/{class_name}")

    print("\n--- Balanced Folder Dataset Created Successfully! ---")
    print(f"Train: {train_per_class * len(class_dirs)} images ({train_per_class} per class)")
    print(f"Val:   {val_per_class * len(class_dirs)} images ({val_per_class} per class)")
    print(f"Test:  {test_per_class * len(class_dirs)} images ({test_per_class} per class)")


if __name__ == "__main__":
    build_balanced_dataset()
