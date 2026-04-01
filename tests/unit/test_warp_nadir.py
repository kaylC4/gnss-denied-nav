"""
test_warp_nadir.py — Unit test per lo Stage 2 (warp to nadir).

Criteri di successo (spec §3.5):
  - Camera downward: H è identità, immagine invariata
  - Camera forward + R=I: asse z = [0,0,1] → warp verso [0,0,-1], H non identità
  - R_corr porta effettivamente z_cam in z_nadir
  - Output shape coerente
  - output_size override funziona
  - Caso degenere vettori opposti: rotazione valida (det=1, ortogonale)
"""

from __future__ import annotations

import numpy as np

from gnss_denied_nav.config import CameraConfig
from gnss_denied_nav.preprocessing.warp_nadir import (
    WarpResult,
    _compute_nadir_homography,
    _rotation_between_vectors,
    warp_to_nadir,
)

# ── fixture ───────────────────────────────────────────────────────────────────

W, H = 640, 480
K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
R_IDENTITY = np.eye(3, dtype=np.float64)


def _make_cfg(orientation: str) -> CameraConfig:
    return CameraConfig(
        camera_type="pinhole",
        camera_orientation=orientation,  # type: ignore[arg-type]
        pixel_pitch_um=1.5,
        K=K,
        dist_coeffs=np.zeros(4),
        image_size=(W, H),
    )


def _blank() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _gradient() -> np.ndarray:
    """Immagine con gradiente — non costante, utile per verificare il warp."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(W, dtype=np.uint8), (H, 1))
    return img


CFG_DOWN = _make_cfg("downward")
CFG_FORWARD = _make_cfg("forward")

# ── test downward (skip) ──────────────────────────────────────────────────────


def test_downward_returns_warp_result() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_DOWN)
    assert isinstance(result, WarpResult)


def test_downward_H_is_identity() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_DOWN)
    np.testing.assert_array_equal(result.H, np.eye(3))


def test_downward_image_unchanged() -> None:
    img = _gradient()
    result = warp_to_nadir(img, K, R_IDENTITY, CFG_DOWN)
    np.testing.assert_array_equal(result.image, img)


def test_downward_output_shape_preserved() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_DOWN)
    assert result.image.shape == (H, W, 3)


# ── test forward ──────────────────────────────────────────────────────────────


def test_forward_R_identity_H_not_identity() -> None:
    """R=I → z_cam=[0,0,1], opposto a z_nadir=[0,0,-1] → warp non banale."""
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_FORWARD)
    assert not np.allclose(result.H, np.eye(3))


def test_forward_output_shape_preserved() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_FORWARD)
    assert result.image.shape == (H, W, 3)


def test_forward_output_size_override() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_FORWARD, output_size=(320, 240))
    assert result.image.shape == (240, 320, 3)


def test_forward_already_nadir_H_near_identity() -> None:
    """Camera che punta già verso il basso: R tale che z_cam = [0,0,-1]."""
    # Rotazione di 180° attorno all'asse x: z → -z
    R_nadir = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    result = warp_to_nadir(_blank(), K, R_nadir, CFG_FORWARD)
    np.testing.assert_allclose(result.H, np.eye(3), atol=1e-6)


def test_forward_H_shape() -> None:
    result = warp_to_nadir(_blank(), K, R_IDENTITY, CFG_FORWARD)
    assert result.H.shape == (3, 3)


def test_forward_output_dtype_uint8() -> None:
    result = warp_to_nadir(_gradient(), K, R_IDENTITY, CFG_FORWARD)
    assert result.image.dtype == np.uint8


# ── test _rotation_between_vectors ───────────────────────────────────────────


def test_rot_between_same_vectors_is_identity() -> None:
    v = np.array([0.0, 0.0, 1.0])
    R = _rotation_between_vectors(v, v)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rot_between_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    R = _rotation_between_vectors(a, b)
    # R @ a deve dare b
    np.testing.assert_allclose(R @ a, b, atol=1e-10)


def test_rot_between_vectors_is_rotation_matrix() -> None:
    """det=1 e R @ R.T = I."""
    a = np.array([1.0, 0.5, 0.3])
    b = np.array([-0.2, 0.8, 0.6])
    R = _rotation_between_vectors(a, b)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_rot_between_opposite_vectors_is_valid_rotation() -> None:
    """Caso degenere: a e b antiparalleli."""
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    R = _rotation_between_vectors(a, b)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    # R @ a deve dare b
    np.testing.assert_allclose(R @ a, b, atol=1e-10)


def test_rot_maps_source_to_target() -> None:
    """R @ a (normalizzato) deve coincidere con b (normalizzato)."""
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    R = _rotation_between_vectors(a, b)
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    np.testing.assert_allclose(R @ a_n, b_n, atol=1e-10)


# ── test _compute_nadir_homography ────────────────────────────────────────────


def test_nadir_homography_shape() -> None:
    H_mat = _compute_nadir_homography(K, R_IDENTITY)
    assert H_mat.shape == (3, 3)


def test_nadir_homography_already_nadir_is_identity() -> None:
    R_nadir = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    H_mat = _compute_nadir_homography(K, R_nadir)
    np.testing.assert_allclose(H_mat, np.eye(3), atol=1e-6)
