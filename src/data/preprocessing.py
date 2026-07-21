"""Fundus-image preprocessing utilities."""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the luminance channel of an RGB image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
