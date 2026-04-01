"""
north_align.py — Stage 3: allineamento al nord geografico.

Ruota l'immagine in modo che il nord sia in alto, compensando l'heading
del drone. L'output ha dimensioni espanse (bounding box della rotazione)
per non tagliare i bordi: angoli neri sono inevitabili per heading ≠ 0°/90°.

La maschera dei pixel validi (True = pixel reale, False = bordo nero) viene
restituita nel risultato e può essere salvata su disco con `save_mask`.

Riferimento: spec drone_satellite_pipeline_spec, §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class NorthAlignResult(NamedTuple):
    """Output dello Stage 3."""

    image: np.ndarray  # (H', W', 3) uint8 BGR — immagine ruotata, bounding box espanso
    mask: np.ndarray  # (H', W')   bool  — True = pixel valido, False = bordo nero
    M: np.ndarray  # (2, 3)   float64 — matrice affine applicata (per tracciabilità)


def north_align(img: np.ndarray, heading_deg: float) -> NorthAlignResult:
    """
    Ruota l'immagine in modo che il nord sia in alto.

    Parameters
    ----------
    img :
        Immagine in vista nadir (H, W, 3) uint8 BGR — output Stage 2.
    heading_deg :
        Heading magnetico/GPS in gradi (0 = nord, CW positivo).
        Viene applicata una rotazione di -heading_deg per compensare.

    Returns
    -------
    NorthAlignResult
        image : immagine ruotata con bounding box espanso
        mask  : maschera booleana (H', W') — True = pixel valido
        M     : matrice affine 2×3 applicata
    """
    h, w = img.shape[:2]
    angle = -heading_deg  # heading CW → rotazione CCW nel piano immagine

    M = _rotation_matrix(w, h, angle)
    new_w, new_h = _expanded_size(w, h, M)

    # Aggiusta la traslazione per centrare l'immagine nel bounding box espanso
    M = _adjust_translation(M, w, new_w, h, new_h)

    rotated: np.ndarray = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = _compute_valid_mask(w, h, M, new_w, new_h)

    return NorthAlignResult(image=rotated, mask=mask, M=M)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """
    Salva la maschera su disco come PNG in scala di grigi.

    255 = pixel valido, 0 = bordo nero introdotto dalla rotazione.

    Parameters
    ----------
    mask :
        Array booleano (H', W') restituito da NorthAlignResult.mask.
    path :
        Percorso di destinazione (es. "outputs/frame_001_mask.png").
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    cv2.imwrite(str(path), mask_uint8)


# ── helpers privati ───────────────────────────────────────────────────────────


def _rotation_matrix(w: int, h: int, angle_deg: float) -> np.ndarray:
    """Matrice affine di rotazione centrata sull'immagine."""
    center = (w / 2.0, h / 2.0)
    M: np.ndarray = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return M


def _expanded_size(w: int, h: int, M: np.ndarray) -> tuple[int, int]:
    """
    Calcola le dimensioni del bounding box dopo la rotazione,
    in modo che l'intera immagine originale sia contenuta senza tagli.
    """
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    return new_w, new_h


def _adjust_translation(M: np.ndarray, w: int, new_w: int, h: int, new_h: int) -> np.ndarray:
    """Aggiunge la traslazione necessaria a centrare l'output nel bounding box."""
    M = M.copy()
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return M


def _compute_valid_mask(w: int, h: int, M: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Genera la maschera booleana dei pixel validi applicando la stessa
    trasformazione affine a un'immagine tutta-255.

    Usa INTER_NEAREST per evitare artefatti di interpolazione ai bordi
    della maschera.
    """
    src_mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask: np.ndarray = cv2.warpAffine(
        src_mask,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_mask > 0
