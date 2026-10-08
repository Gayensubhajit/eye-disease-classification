"""PyTorch Dataset and DataLoader module for folder-based fundus eye disease classification."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.data.preprocessing import (
    crop_fundus_area,
    apply_clahe,
    get_train_transforms,
    get_val_transforms,
)


class FundusDataset(Dataset):
    """PyTorch Dataset loading fundus images directly from class subfolders (ImageFolder layout)."""

    def __init__(
        self,
        root_dir: str | Path,
        class_names: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        crop_fundus: bool = True,
        apply_clahe_flag: bool = False,
    ):
        """
        Args:
            root_dir: Path to directory containing class subdirectories (e.g. data/train).
            class_names: Optional list of class names to ensure consistent label indexing.
            transform: Albumentations transformation pipeline.
            crop_fundus: Whether to crop black background borders.
            apply_clahe_flag: Whether to apply CLAHE enhancement.
        """
        self.root_path = Path(root_dir)
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_path.resolve()}")

        # Discover class folders
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted([d.name for d in self.root_path.iterdir() if d.is_dir()])

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.crop_fundus = crop_fundus
        self.apply_clahe_flag = apply_clahe_flag

        # Collect (image_path, label) samples
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_path / class_name
            if not class_dir.exists():
                continue
            class_idx = self.class_to_idx[class_name]
            image_files = sorted(
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.png"))
            )
            for img_file in image_files:
                self.samples.append((img_file, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.root_path.resolve()}. Check dataset directory path.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable at path: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop_fundus:
            image = crop_fundus_area(image)

        if self.apply_clahe_flag:
            image = apply_clahe(image)

        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image_tensor, label

    @property
    def labels(self) -> np.ndarray:
        """Return numpy array of all labels in dataset."""
        return np.array([s[1] for s in self.samples])


def create_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders directly from folder paths.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    image_size = config["data"]["image_size"]
    crop_fundus = config["data"].get("crop_fundus", True)
    apply_clahe_flag = config["data"].get("apply_clahe", False)
    class_names = config["data"].get("class_names", None)

    conservative_aug = (
        config['data'].get('augmentation_mode') == 'conservative'
        or config['data'].get('conservative_augmentation', False)
    )
    train_transforms = get_train_transforms(image_size, conservative=conservative_aug)
    val_transforms = get_val_transforms(image_size)

    train_dataset = FundusDataset(
        config["data"]["train_dir"],
        class_names=class_names,
        transform=train_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    val_dataset = FundusDataset(
        config["data"]["val_dir"],
        class_names=class_names,
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    test_dataset = FundusDataset(
        config["data"]["test_dir"],
        class_names=class_names,
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
