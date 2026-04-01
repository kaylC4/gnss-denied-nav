"""
gsd_match.py — Stage 4: allineamento della scala spaziale (GSD match).

Ridimensiona l'immagine drone in modo che il suo GSD (Ground Sample Distance)
coincida con quello del satellite target.

  - gsd_drone < gsd_satellite → drone più dettagliato → downsample (INTER_AREA)
  - gsd_drone > gsd_satellite → drone meno dettagliato → upsample (INTER_CUBIC)

Se viene fornita la maschera prodotta dallo Stage 3 (north_align), viene
ridimensionata con la stessa scala usando INTER_NEAREST per mantenere i
valori binari.

Riferimento: spec drone_satellite_pipeline_spec, §5.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from gnss_denied_nav.config import CameraConfig


class GSDMatchResult(NamedTuple):
    """Output dello Stage 4."""

    image: np.ndarray  # (H', W', 3) uint8 BGR — immagine rescalata
    mask: np.ndarray | None  # (H', W') bool | None — maschera rescalata (se fornita)
    scale: float  # fattore di scala applicato = gsd_drone / gsd_satellite
    gsd_drone_m: float  # GSD calcolato dell'immagine drone [m/px]


def gsd_match(
    img: np.ndarray,
    cfg: CameraConfig,
    alt_agl_m: float,
    satellite_gsd_m: float,
    mask: np.ndarray | None = None,
) -> GSDMatchResult:
    """
    Ridimensiona l'immagine drone per matchare il GSD del satellite.

    Parameters
    ----------
    img :
        Immagine north-aligned (H, W, 3) uint8 BGR — output Stage 3.
    cfg :
        Configurazione camera (usa pixel_pitch_um e K[0,0] per il GSD).
    alt_agl_m :
        Altitudine Above Ground Level [m].
    satellite_gsd_m :
        GSD target del satellite [m/px].
    mask :
        Maschera booleana (H, W) da Stage 3. Se fornita viene rescalata
        con la stessa scala dell'immagine.

    Returns
    -------
    GSDMatchResult
    """
    if alt_agl_m <= 0.0:
        raise ValueError(f"alt_agl_m deve essere > 0, ricevuto: {alt_agl_m}")
    if satellite_gsd_m <= 0.0:
        raise ValueError(f"satellite_gsd_m deve essere > 0, ricevuto: {satellite_gsd_m}")

    gsd_drone = compute_gsd(cfg.pixel_pitch_um, alt_agl_m, cfg.K)
    scale = gsd_drone / satellite_gsd_m

    resized_img = _resize(img, scale, binary=False)
    resized_mask = _resize(mask, scale, binary=True) if mask is not None else None

    return GSDMatchResult(
        image=resized_img,
        mask=resized_mask,
        scale=scale,
        gsd_drone_m=gsd_drone,
    )


def compute_gsd(pixel_pitch_um: float, alt_agl_m: float, K: np.ndarray) -> float:
    """
    Calcola il GSD (Ground Sample Distance) dell'immagine drone [m/px].

        GSD = (pixel_pitch [m] * altitude [m]) / focal_length [m]

    La focal length in metri si ricava da:
        fx [px] = focal_length [m] / pixel_pitch [m]
        → focal_length [m] = fx * pixel_pitch_um * 1e-6

    Parameters
    ----------
    pixel_pitch_um :
        Dimensione fisica del pixel del sensore [µm].
    alt_agl_m :
        Altitudine Above Ground Level [m].
    K :
        Matrice intrinseca (3, 3) — usa K[0, 0] come focal length in pixel.
    """
    pixel_pitch_m = pixel_pitch_um * 1e-6
    focal_m = float(K[0, 0]) * pixel_pitch_m
    return (pixel_pitch_m * alt_agl_m) / focal_m


# ── helpers privati ───────────────────────────────────────────────────────────


def _resize(arr: np.ndarray, scale: float, binary: bool) -> np.ndarray:
    """
    Ridimensiona un array 2D o 3D con il fattore scale.

    Parameters
    ----------
    arr :
        Array da ridimensionare.
    scale :
        Fattore di scala (< 1 = rimpicciolisce, > 1 = ingrandisce).
    binary :
        Se True usa INTER_NEAREST (per maschere booleane).
        Se False usa INTER_AREA (downsample) o INTER_CUBIC (upsample).
    """
    h, w = arr.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if binary:
        # cv2.resize non accetta bool — convertiamo in uint8 e ritorniamo bool
        src = arr.astype(np.uint8) * 255
        resized_u8: np.ndarray = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized_u8 > 0

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized: np.ndarray = cv2.resize(arr, (new_w, new_h), interpolation=interp)
    return resized
