"""
warp_nadir.py — Stage 2: proiezione top-down (warp to nadir).

Calcola l'homography che porta l'asse ottico della camera a puntare
verticalmente verso il basso, compensando il tilt IMU.

Saltato automaticamente se camera_orientation == "downward".

Riferimento: spec drone_satellite_pipeline_spec, §3.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gnss_denied_nav.config import CameraConfig

# Asse z nadir: verso il basso nel frame mondo (convenzione NED / OpenCV)
_Z_NADIR = np.array([0.0, 0.0, -1.0])


class WarpResult(NamedTuple):
    """Output dello Stage 2."""

    image: np.ndarray  # (H, W, 3) uint8 BGR — immagine in vista top-down
    H: np.ndarray  # (3, 3) float64 — homography applicata (identità se saltato)


def warp_to_nadir(
    img: np.ndarray,
    K: np.ndarray,
    R_cam: np.ndarray,
    cfg: CameraConfig,
    output_size: tuple[int, int] | None = None,
) -> WarpResult:
    """
    Proietta l'immagine in vista nadir (top-down).

    Parameters
    ----------
    img :
        Immagine rettificata (H, W, 3) uint8 BGR — output Stage 1.
    K :
        Matrice intrinseca (3, 3) aggiornata — K_new da Stage 1.
    R_cam :
        Matrice di rotazione camera-to-world (3, 3) dall'IMU.
    cfg :
        Configurazione camera. Se camera_orientation == "downward" lo stage
        è saltato e viene restituita l'immagine invariata con H = I.
    output_size :
        (width, height) dell'immagine di output. Se None usa le dimensioni
        dell'immagine in ingresso.

    Returns
    -------
    WarpResult
        image : immagine in vista top-down
        H     : homography applicata (da passare per tracciabilità)
    """
    if cfg.camera_orientation == "downward":
        h, w = img.shape[:2]
        return WarpResult(image=img.copy(), H=np.eye(3, dtype=np.float64))

    H = _compute_nadir_homography(K, R_cam)
    h, w = img.shape[:2]
    size = output_size if output_size is not None else (w, h)

    warped: np.ndarray = cv2.warpPerspective(
        img,
        H,
        size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return WarpResult(image=warped, H=H)


# ── helpers privati ───────────────────────────────────────────────────────────


def _compute_nadir_homography(K: np.ndarray, R_cam: np.ndarray) -> np.ndarray:
    """H = K @ R_corr @ K^{-1}, dove R_corr porta z_cam → z_nadir."""
    z_cam = R_cam[:, 2]
    R_corr = _rotation_between_vectors(z_cam, _Z_NADIR)
    H: np.ndarray = K @ R_corr @ np.linalg.inv(K)
    return H


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrice di rotazione minima che porta il vettore a nel vettore b
    (formula di Rodrigues).

    Gestisce il caso degenere di vettori opposti (rotazione di π attorno
    a un asse perpendicolare arbitrario).
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    c = float(np.dot(a, b))

    # Caso degenere: vettori opposti — ruota di π attorno a un asse perp.
    if c < -0.999999:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        rot: np.ndarray = Rotation.from_rotvec(np.pi * axis).as_matrix()
        return rot

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    R: np.ndarray = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s + 1e-12)
    return R
