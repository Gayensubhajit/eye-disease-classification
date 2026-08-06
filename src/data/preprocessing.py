"""Preprocessing routines and Albumentations augmentations for colour fundus images."""

from typing import Tuple, Optional
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def crop_fundus_area(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop uninformative dark margins around the fundus circle.

    Args:
        image: RGB numpy image array.
        threshold: Intensity threshold below which pixels are considered background.

    Returns:
        Cropped RGB image array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return image

    row_sum = np.sum(mask, axis=1)
    col_sum = np.sum(mask, axis=0)

    y_min, y_max = np.where(row_sum > 0)[0][[0, -1]]
    x_min, x_max = np.where(col_sum > 0)[0][[0, -1]]

    # Ensure valid bounding box
    if y_max > y_min and x_max > x_min:
        return image[y_min : y_max + 1, x_min : x_max + 1]
    return image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on LAB color space.

    Args:
        image: RGB numpy image array.
        clip_limit: CLAHE contrast threshold limit.
        tile_grid_size: Grid size for histogram equalization.

    Returns:
        CLAHE-enhanced RGB image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Return training data augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Return validation and testing data transformation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
