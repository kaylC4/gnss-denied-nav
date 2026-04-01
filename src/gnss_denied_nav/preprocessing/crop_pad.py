"""
crop_pad.py — Stage 5: crop / pad a dimensione fissa.

Porta l'immagine (e la maschera opzionale) esattamente a target_size × target_size
pixel, richiesti dall'encoder di embedding.

Strategia (in ordine):
  1. Center crop  — se una delle dimensioni supera target_size
  2. Zero pad     — se una delle dimensioni è ancora inferiore a target_size

La maschera viene trattata con la stessa trasformazione geometrica:
  - crop: stesse coordinate dell'immagine
  - pad:  bordo aggiunto con False (pixel non reali)

Riferimento: spec drone_satellite_pipeline_spec, §6.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


class CropPadResult(NamedTuple):
    """Output dello Stage 5."""

    image: np.ndarray  # (target_size, target_size, 3) uint8 BGR
    mask: np.ndarray | None  # (target_size, target_size) bool | None


def crop_pad(
    img: np.ndarray,
    target_size: int = 512,
    pad_value: int = 0,
    mask: np.ndarray | None = None,
) -> CropPadResult:
    """
    Porta l'immagine a target_size × target_size pixel.

    Parameters
    ----------
    img :
        Immagine rescalata (H, W, 3) uint8 BGR — output Stage 4.
    target_size :
        Dimensione target in pixel (quadrata). Default: 512 per DINOv2.
    pad_value :
        Valore usato per il padding (default 0 = nero).
    mask :
        Maschera booleana (H, W) opzionale da Stage 3/4. Viene trasformata
        con le stesse coordinate. Il padding aggiunge False.

    Returns
    -------
    CropPadResult
        image : (target_size, target_size, 3) uint8
        mask  : (target_size, target_size) bool | None
    """
    if target_size <= 0:
        raise ValueError(f"target_size deve essere > 0, ricevuto: {target_size}")
    if not 0 <= pad_value <= 255:
        raise ValueError(f"pad_value deve essere in [0, 255], ricevuto: {pad_value}")

    img_out, mask_out = _crop(img, mask, target_size)
    img_out, mask_out = _pad(img_out, mask_out, target_size, pad_value)

    return CropPadResult(image=img_out, mask=mask_out)


# ── helpers privati ───────────────────────────────────────────────────────────


def _crop(
    img: np.ndarray,
    mask: np.ndarray | None,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Center crop se almeno una dimensione supera target_size."""
    h, w = img.shape[:2]
    if h <= target_size and w <= target_size:
        return img, mask

    y_start = max(0, (h - target_size) // 2)
    x_start = max(0, (w - target_size) // 2)
    y_end = y_start + min(h, target_size)
    x_end = x_start + min(w, target_size)

    img_cropped = img[y_start:y_end, x_start:x_end]
    mask_cropped = mask[y_start:y_end, x_start:x_end] if mask is not None else None
    return img_cropped, mask_cropped


def _pad(
    img: np.ndarray,
    mask: np.ndarray | None,
    target_size: int,
    pad_value: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Padding simmetrico se almeno una dimensione è inferiore a target_size."""
    h, w = img.shape[:2]
    if h >= target_size and w >= target_size:
        return img, mask

    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded: np.ndarray = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[pad_value] * 3,
    )

    mask_padded: np.ndarray | None = None
    if mask is not None:
        # Padding con False: i pixel aggiunti non sono reali
        mask_u8 = mask.astype(np.uint8)
        mask_padded_u8: np.ndarray = cv2.copyMakeBorder(
            mask_u8,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        mask_padded = mask_padded_u8 > 0

    return img_padded, mask_padded
