"""PyTorch Dataset and DataLoader creation for fundus eye disease classification."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

from src.data.preprocessing import (
    crop_fundus_area,
    apply_clahe,
    get_train_transforms,
    get_val_transforms,
)


class FundusDataset(Dataset):
    """PyTorch Dataset loading fundus images from split CSV manifests."""

    def __init__(
        self,
        df_or_csv: Any,
        transform: Optional[Any] = None,
        crop_fundus: bool = True,
        apply_clahe_flag: bool = False,
    ):
        """
        Args:
            df_or_csv: Path to CSV file or a pandas DataFrame.
            transform: Albumentations transformation pipeline.
            crop_fundus: Whether to crop black background borders.
            apply_clahe_flag: Whether to apply CLAHE enhancement.
        """
        if isinstance(df_or_csv, (str, Path)):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.transform = transform
        self.crop_fundus = crop_fundus
        self.apply_clahe_flag = apply_clahe_flag

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])

        image = cv2.imread(image_path)
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


def create_dataloaders(
    config: Dict[str, Any],
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders.

    Args:
        config: Configuration dictionary loaded from config.yaml.
        use_weighted_sampler: If True, uses WeightedRandomSampler for train dataloader.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    image_size = config["data"]["image_size"]
    crop_fundus = config["data"].get("crop_fundus", True)
    apply_clahe_flag = config["data"].get("apply_clahe", False)

    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    train_dataset = FundusDataset(
        config["data"]["train_csv"],
        transform=train_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    val_dataset = FundusDataset(
        config["data"]["val_csv"],
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )
    test_dataset = FundusDataset(
        config["data"]["test_csv"],
        transform=val_transforms,
        crop_fundus=crop_fundus,
        apply_clahe_flag=apply_clahe_flag,
    )

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        labels = train_dataset.df["label"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
