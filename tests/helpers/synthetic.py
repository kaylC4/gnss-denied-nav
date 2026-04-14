"""
synthetic.py — Generatori di test data sintetici per i test della pipeline.

Fornisce:
  make_checkerboard   — scacchiera anti-aliasata via supersampling 4×
  apply_brown_conrady — distorsione forward (clean → distorted)
  psnr                — Peak Signal-to-Noise Ratio in dB tra due immagini uint8
"""

from __future__ import annotations

import cv2
import numpy as np


def make_checkerboard(h: int, w: int, square_size: int) -> np.ndarray:
    """
    Genera una scacchiera anti-aliasata via supersampling 4× + INTER_AREA.

    Il board è asimmetrico quando h/square_size ≠ w/square_size (es. 12×16
    quadrati per 768×1024 px con square_size=64) — questo permette a
    cv2.findChessboardCorners di disambiguare l'orientamento.

    Parameters
    ----------
    h, w        : dimensioni dell'immagine output in pixel
    square_size : lato del singolo quadrato in pixel nell'output finale

    Returns
    -------
    np.ndarray  : (h, w, 3) uint8 BGR, quadrati bianchi e neri alternati
    """
    scale = 4
    h_big, w_big = h * scale, w * scale
    sq_big = square_size * scale

    img_big = np.zeros((h_big, w_big), dtype=np.uint8)
    n_rows = (h_big + sq_big - 1) // sq_big
    n_cols = (w_big + sq_big - 1) // sq_big

    for r in range(n_rows):
        for c in range(n_cols):
            if (r + c) % 2 == 0:
                r0 = r * sq_big
                r1 = min((r + 1) * sq_big, h_big)
                c0 = c * sq_big
                c1 = min((c + 1) * sq_big, w_big)
                img_big[r0:r1, c0:c1] = 255

    img_gray: np.ndarray = cv2.resize(img_big, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # type: ignore[no-any-return]


def apply_brown_conrady(
    clean_img: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    balance: float = 0.0,
) -> np.ndarray:
    """
    Applica la distorsione forward Brown-Conrady a un'immagine pulita.

    Costruisce la backward map via cv2.undistortPoints con P=K_new, dove K_new è
    calcolato con lo stesso ``balance`` (alpha) che verrà usato da undistort().
    Questo rende la mappa di proiezione coerente con quella di _undistort_pinhole,
    garantendo che undistort(apply_brown_conrady(img, K, D, b), K, D, b) ≈ img.

    Meccanismo:
      cv2.undistort usa K_new = getOptimalNewCameraMatrix(K, D, size, alpha=balance).
      La backward map di undistort per pixel (u_r, v_r) nell'output è:
        normalizza con K_new → applica distorsione → proietta con K → (u_src, v_src)
      L'inversa di questa trasformazione per pixel (u_d, v_d) nel distorto è:
        normalizza con K → applica distorsione inversa → proietta con K_new → (u_c, v_c)
      Che è esattamente cv2.undistortPoints(pts, K, D, P=K_new).

    Parameters
    ----------
    clean_img   : (H, W, 3) uint8 BGR — immagine pulita (ground truth)
    K           : (3, 3) float64 — matrice intrinseca
    dist_coeffs : 4 o 5 elementi float64 — [k1, k2, p1, p2 (, k3)]
    balance     : stesso valore passato a undistort() — determina K_new

    Returns
    -------
    np.ndarray  : (H, W, 3) uint8 BGR — immagine distorta, stessa shape dell'input
    """
    h, w = clean_img.shape[:2]

    # Calcola K_new con lo stesso alpha usato da _undistort_pinhole.
    # Usare P=K_new (invece di P=K) garantisce la coerenza del round-trip.
    K_new: np.ndarray
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=balance)

    xs, ys = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    # pts_dist: coordinate pixel nell'immagine distorta (output) per cui
    # cerchiamo la corrispondente posizione nell'immagine pulita (input)
    pts_dist = np.stack([xs.ravel(), ys.ravel()], axis=-1).reshape(-1, 1, 2)

    # P=K_new specchia la proiezione output usata da _undistort_pinhole
    pts_clean: np.ndarray = cv2.undistortPoints(pts_dist, K, dist_coeffs, P=K_new)

    map_x = pts_clean[:, 0, 0].reshape(h, w)
    map_y = pts_clean[:, 0, 1].reshape(h, w)

    distorted: np.ndarray = cv2.remap(
        clean_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return distorted


def psnr(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB tra due immagini uint8.

    Returns float('inf') se le immagini sono pixel-identiche.
    """
    mse = float(np.mean((img_ref.astype(np.float64) - img_test.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))
