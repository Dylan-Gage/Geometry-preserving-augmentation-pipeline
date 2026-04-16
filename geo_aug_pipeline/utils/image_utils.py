"""
utils/image_utils.py
Low-level image helpers shared across modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_image_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str | Path, rgb: bool = True) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img
    cv2.imwrite(str(path), out)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Ensure binary mask is dtype uint8 with values 0/255."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def crop_with_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    padding: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop the tight bounding box of the masked object (+ padding).

    Returns
    -------
    crop        : RGB crop of the bounding region
    mask_crop   : corresponding binary mask crop (uint8, 0/255)
    bbox        : (x1, y1, x2, y2) in original image coords
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask — no foreground pixels found.")

    h, w = image.shape[:2]
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, w)
    y2 = min(int(ys.max()) + padding, h)

    crop = image[y1:y2, x1:x2]
    mask_crop = binary_mask[y1:y2, x1:x2]
    return crop, mask_crop, (x1, y1, x2, y2)


def composite_foreground(
    original: np.ndarray,
    augmented: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Paste the original foreground pixels (from `original`) on top of
    `augmented` using `binary_mask`. This enforces pixel-identical geometry.

    All three arrays must have the same spatial dimensions.
    """
    assert original.shape == augmented.shape, "Shape mismatch in composite."
    mask_3ch = np.stack([binary_mask > 0] * 3, axis=-1)
    result = augmented.copy()
    result[mask_3ch] = original[mask_3ch]
    return result


def polygon_to_binary_mask(
    polygons: list[list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert COCO-style polygon segmentation to a binary mask.

    Parameters
    ----------
    polygons : list of flat [x1,y1,x2,y2,...] coordinate lists
    height, width : output mask dimensions
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    """Simple RLE encoding for mask storage (not pycocotools format)."""
    flat = binary_mask.flatten()
    changes = np.concatenate(([0], np.where(np.diff(flat))[0] + 1, [len(flat)]))
    lengths = np.diff(changes)
    values = flat[changes[:-1]]
    return {"size": list(binary_mask.shape), "counts": lengths.tolist(), "values": values.tolist()}
