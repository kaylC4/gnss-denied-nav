"""
undistort.py — Stage 1: rimozione distorsione lente.

Deve essere applicato sull'immagine raw prima di qualsiasi altra
trasformazione geometrica. Supporta modelli pinhole (radiale/tangenziale)
e fisheye (equidistante, 4 parametri).

Riferimento: spec drone_satellite_pipeline_spec, §2.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from gnss_denied_nav.config import CameraConfig


class UndistortResult(NamedTuple):
    """Output dello Stage 1."""

    image: np.ndarray  # (H', W', 3) uint8 BGR — immagine rettificata
    K_new: np.ndarray  # (3, 3) float64 — matrice intrinseca aggiornata (input Stage 2)


def undistort(img: np.ndarray, cfg: CameraConfig, balance: float = 0.0) -> UndistortResult:
    """
    Rimuove la distorsione della lente dall'immagine raw.

    Parameters
    ----------
    img :
        Immagine raw (H, W, 3) uint8 BGR.
    cfg :
        Configurazione camera con K, dist_coeffs e camera_type.
    balance :
        Trade-off FOV / bordi neri in [0.0, 1.0].
        0.0 → nessun bordo nero, FOV ridotto (default, consigliato per matching).
        1.0 → FOV completo, bordi neri presenti.

    Returns
    -------
    UndistortResult
        image : immagine rettificata
        K_new : nuova matrice intrinseca da passare allo Stage 2
    """
    if not 0.0 <= balance <= 1.0:
        raise ValueError(f"balance deve essere in [0.0, 1.0], ricevuto: {balance}")

    if cfg.camera_type == "fisheye":
        return _undistort_fisheye(img, cfg, balance)
    return _undistort_pinhole(img, cfg, balance)


# ── implementazioni private ───────────────────────────────────────────────────


def _undistort_pinhole(img: np.ndarray, cfg: CameraConfig, balance: float) -> UndistortResult:
    h, w = img.shape[:2]
    K = cfg.K
    D = cfg.dist_coeffs

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=balance)

    undist: np.ndarray = cv2.undistort(img, K, D, None, new_K)

    # Con balance < 1.0 il ROI contiene solo pixel validi — croppiamo ed
    # aggiorniamo il punto principale di conseguenza.
    if balance < 1.0:
        x, y, rw, rh = roi
        undist = undist[y : y + rh, x : x + rw]
        new_K = new_K.copy()
        new_K[0, 2] -= x
        new_K[1, 2] -= y

    return UndistortResult(image=undist, K_new=new_K)


def _undistort_fisheye(img: np.ndarray, cfg: CameraConfig, balance: float) -> UndistortResult:
    h, w = img.shape[:2]
    K = cfg.K
    D = cfg.dist_coeffs.reshape(4, 1)  # fisheye API richiede shape (4, 1)

    new_K: np.ndarray = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    map1: np.ndarray
    map2: np.ndarray
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

    undist: np.ndarray = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return UndistortResult(image=undist, K_new=new_K)
