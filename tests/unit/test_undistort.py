"""
test_undistort.py — Unit test per lo Stage 1 (rimozione distorsione lente).

Criteri di successo (spec §2.4):
  - Output shape coerente con balance e camera_type
  - K_new aggiornato correttamente (cx/cy shiftati dopo crop)
  - Immagine senza distorsione su input sintetico a distorsione nulla
  - Fisheye: reshape dist_coeffs (4,) → (4, 1) trasparente al chiamante
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_denied_nav.config import CameraConfig
from gnss_denied_nav.preprocessing.undistort import UndistortResult, undistort

# ── fixture ───────────────────────────────────────────────────────────────────

W, H = 640, 480
FX = FY = 500.0
CX, CY = W / 2, H / 2


def _make_camera(camera_type: str, dist: list[float]) -> CameraConfig:
    return CameraConfig(
        camera_type=camera_type,  # type: ignore[arg-type]
        camera_orientation="downward",
        pixel_pitch_um=1.5,
        K=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array(dist, dtype=np.float64),
        image_size=(W, H),
    )


def _blank_image() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    """Pattern a scacchiera 8×6 — linee rette nel mondo."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    sq = H // 8
    for r in range(8):
        for c in range(W // sq):
            if (r + c) % 2 == 0:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 255
    return img


PINHOLE_ZERO = _make_camera("pinhole", [0.0, 0.0, 0.0, 0.0])
PINHOLE_DIST = _make_camera("pinhole", [-0.3, 0.1, 0.0, 0.0])
FISHEYE_ZERO = _make_camera("fisheye", [0.0, 0.0, 0.0, 0.0])

# ── test output type ──────────────────────────────────────────────────────────


def test_returns_undistort_result() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO)
    assert isinstance(result, UndistortResult)
    assert isinstance(result.image, np.ndarray)
    assert isinstance(result.K_new, np.ndarray)


# ── test pinhole ──────────────────────────────────────────────────────────────


def test_pinhole_zero_dist_balance1_preserves_size() -> None:
    """Distorsione nulla + balance=1 → dimensioni invariate."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    assert result.image.shape == (H, W, 3)


def test_pinhole_zero_dist_balance0_output_is_3channel() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_pinhole_zero_dist_Knew_shape() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    assert result.K_new.shape == (3, 3)


def test_pinhole_zero_dist_Knew_last_row() -> None:
    """La terza riga di K_new deve essere [0, 0, 1]."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    np.testing.assert_array_equal(result.K_new[2], [0.0, 0.0, 1.0])


def test_pinhole_zero_dist_balance1_Knew_close_to_K() -> None:
    """Con distorsione nulla e balance=1, K_new ≈ K."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    np.testing.assert_allclose(result.K_new, PINHOLE_ZERO.K, atol=1.0)


def test_pinhole_balance0_crop_shifts_principal_point() -> None:
    """
    Con balance=0.0 e distorsione non nulla, il crop deve shiftare cx/cy
    in modo che il punto principale rimanga coerente con l'immagine croppata.
    """
    result = undistort(_checkerboard(), PINHOLE_DIST, balance=0.0)
    h_out, w_out = result.image.shape[:2]
    cx_new = result.K_new[0, 2]
    cy_new = result.K_new[1, 2]
    # Il punto principale deve stare dentro l'immagine croppata
    assert 0 <= cx_new <= w_out, f"cx_new={cx_new} fuori da [0, {w_out}]"
    assert 0 <= cy_new <= h_out, f"cy_new={cy_new} fuori da [0, {h_out}]"


def test_pinhole_balance1_no_crop_cx_cy_unchanged_approx() -> None:
    """Con balance=1 non c'è crop → cx/cy non vengono shiftati."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    # Con distorsione nulla, K_new ≈ K originale
    assert abs(result.K_new[0, 2] - CX) < 2.0
    assert abs(result.K_new[1, 2] - CY) < 2.0


# ── test fisheye ──────────────────────────────────────────────────────────────


def test_fisheye_zero_dist_output_shape() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_fisheye_Knew_shape() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    assert result.K_new.shape == (3, 3)


def test_fisheye_Knew_last_row() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    np.testing.assert_array_equal(result.K_new[2], [0.0, 0.0, 1.0])


def test_fisheye_dtype_uint8() -> None:
    """L'immagine output deve rimanere uint8."""
    result = undistort(_checkerboard(), FISHEYE_ZERO, balance=0.0)
    assert result.image.dtype == np.uint8


# ── test validazione input ────────────────────────────────────────────────────


def test_invalid_balance_raises() -> None:
    with pytest.raises(ValueError, match="balance"):
        undistort(_blank_image(), PINHOLE_ZERO, balance=1.5)


def test_negative_balance_raises() -> None:
    with pytest.raises(ValueError, match="balance"):
        undistort(_blank_image(), PINHOLE_ZERO, balance=-0.1)


# ── test dtype preservato ─────────────────────────────────────────────────────


def test_pinhole_output_dtype_uint8() -> None:
    result = undistort(_checkerboard(), PINHOLE_DIST, balance=0.0)
    assert result.image.dtype == np.uint8
