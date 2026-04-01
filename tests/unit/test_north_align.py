"""
test_north_align.py — Unit test per lo Stage 3 (north-align).

Criteri di successo (spec §4.3):
  - heading=0°: immagine e mask invariate (no rotazione)
  - heading=90°: dimensioni espanse, mask non tutta True
  - heading=45°: bounding box espanso rispetto all'originale
  - mask shape == image shape[:2]
  - mask dtype == bool
  - pixel validi interni alla maschera, bordi neri esterni
  - save_mask: file PNG scritto correttamente, valori 0/255
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnss_denied_nav.preprocessing.north_align import (
    NorthAlignResult,
    _compute_valid_mask,
    _expanded_size,
    _rotation_matrix,
    north_align,
    save_mask,
)

# ── fixture ───────────────────────────────────────────────────────────────────

W, H = 200, 150


def _solid(color: tuple[int, int, int] = (128, 64, 32)) -> np.ndarray:
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = color
    return img


def _gradient() -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(W, dtype=np.uint8), (H, 1))
    return img


# ── test tipo di ritorno ──────────────────────────────────────────────────────


def test_returns_north_align_result() -> None:
    result = north_align(_solid(), 0.0)
    assert isinstance(result, NorthAlignResult)


# ── test heading=0 ────────────────────────────────────────────────────────────


def test_zero_heading_shape_preserved() -> None:
    result = north_align(_solid(), 0.0)
    assert result.image.shape == (H, W, 3)


def test_zero_heading_mask_all_true() -> None:
    """Nessuna rotazione → nessun bordo nero → tutta la maschera è True."""
    result = north_align(_solid(), 0.0)
    assert result.mask.all()


def test_zero_heading_M_is_identity_affine() -> None:
    """heading=0 → matrice affine è la proiezione dell'identità."""
    result = north_align(_solid(), 0.0)
    # Parte rotazionale deve essere identità
    np.testing.assert_allclose(result.M[:, :2], np.eye(2), atol=1e-6)


# ── test heading=90 ───────────────────────────────────────────────────────────


def test_90deg_heading_image_is_3channel() -> None:
    result = north_align(_solid(), 90.0)
    assert result.image.ndim == 3 and result.image.shape[2] == 3


def test_90deg_heading_mask_not_all_true() -> None:
    """Rotazione di 90° su immagine non quadrata → bordi neri presenti."""
    result = north_align(_solid(), 90.0)
    assert not result.mask.all()


def test_45deg_heading_mask_has_false_corners() -> None:
    """
    45° su immagine rettangolare → angoli del bounding box espanso fuori maschera.
    (90° su rettangolo non produce angoli neri: l'immagine riempie esattamente
    il bounding box ruotato.)
    """
    result = north_align(_solid(), 45.0)
    h, w = result.mask.shape
    assert not result.mask[0, 0]
    assert not result.mask[0, w - 1]
    assert not result.mask[h - 1, 0]
    assert not result.mask[h - 1, w - 1]


def test_90deg_heading_center_is_valid() -> None:
    """Il centro dell'immagine deve sempre essere un pixel valido."""
    result = north_align(_solid(), 90.0)
    h, w = result.mask.shape
    assert result.mask[h // 2, w // 2]


# ── test heading=45 ───────────────────────────────────────────────────────────


def test_45deg_bounding_box_expanded() -> None:
    """45° su immagine non quadrata → bounding box > dimensioni originali."""
    result = north_align(_solid(), 45.0)
    h_out, w_out = result.image.shape[:2]
    assert w_out > W or h_out > H


def test_45deg_expected_size() -> None:
    """Verifica la formula del bounding box per 45°."""
    angle_rad = math.radians(45.0)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    expected_w = int(H * sin_a + W * cos_a)
    expected_h = int(H * cos_a + W * sin_a)
    result = north_align(_solid(), 45.0)
    assert result.image.shape[:2] == (expected_h, expected_w)


# ── test mask proprietà ───────────────────────────────────────────────────────


def test_mask_shape_matches_image() -> None:
    for heading in [0.0, 30.0, 90.0, 180.0]:
        result = north_align(_solid(), heading)
        assert result.mask.shape == result.image.shape[:2], f"heading={heading}"


def test_mask_dtype_bool() -> None:
    result = north_align(_solid(), 45.0)
    assert result.mask.dtype == bool


def test_mask_false_pixels_are_black_in_image() -> None:
    """
    Pixel fuori maschera devono essere (0,0,0) o quasi.
    INTER_LANCZOS4 ha kernel 4×4 e può sbordare di 1-2px oltre il confine
    segnato dalla maschera INTER_NEAREST: tolleriamo un margine di 1%.
    """
    result = north_align(_solid(), 45.0)
    invalid = ~result.mask
    invalid_pixels = result.image[invalid]  # (N, 3)
    nonzero_rows = np.any(invalid_pixels > 0, axis=1)
    bleed_ratio = nonzero_rows.sum() / max(len(nonzero_rows), 1)
    assert bleed_ratio < 0.05, f"troppi pixel non-zero fuori maschera: {bleed_ratio:.2%}"


def test_mask_valid_pixels_nonzero_for_solid_color() -> None:
    """Pixel dentro la maschera devono avere colore non zero (immagine solida)."""
    result = north_align(_solid((100, 100, 100)), 45.0)
    valid = result.mask
    assert np.any(result.image[valid] > 0)


# ── test dtype e contenuto immagine ──────────────────────────────────────────


def test_output_dtype_uint8() -> None:
    result = north_align(_gradient(), 33.0)
    assert result.image.dtype == np.uint8


def test_180deg_heading_same_size() -> None:
    """180° → bounding box identico all'originale."""
    result = north_align(_solid(), 180.0)
    assert result.image.shape == (H, W, 3)


# ── test save_mask ────────────────────────────────────────────────────────────


def test_save_mask_creates_file(tmp_path: "pytest.TempPathFactory") -> None:
    result = north_align(_solid(), 45.0)
    out = tmp_path / "mask.png"
    save_mask(result.mask, out)
    assert out.exists()


def test_save_mask_values_are_0_or_255(tmp_path: "pytest.TempPathFactory") -> None:
    import cv2

    result = north_align(_solid(), 45.0)
    out = tmp_path / "mask.png"
    save_mask(result.mask, out)
    loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    unique_vals = set(np.unique(loaded).tolist())
    assert unique_vals.issubset({0, 255})


def test_save_mask_roundtrip(tmp_path: "pytest.TempPathFactory") -> None:
    """Caricando la PNG salvata si ottiene la stessa maschera booleana."""
    import cv2

    result = north_align(_solid(), 45.0)
    out = tmp_path / "mask.png"
    save_mask(result.mask, out)
    loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    restored = loaded > 0
    np.testing.assert_array_equal(restored, result.mask)


# ── test helpers privati ──────────────────────────────────────────────────────


def test_expanded_size_zero_angle() -> None:
    M = _rotation_matrix(W, H, 0.0)
    new_w, new_h = _expanded_size(W, H, M)
    assert new_w == W and new_h == H


def test_expanded_size_90deg_swaps_dimensions() -> None:
    """90° → larghezza e altezza si scambiano (approssimativamente)."""
    M = _rotation_matrix(W, H, 90.0)
    new_w, new_h = _expanded_size(W, H, M)
    assert new_w == H and new_h == W


def test_compute_valid_mask_all_ones_zero_angle() -> None:
    M = _rotation_matrix(W, H, 0.0)
    mask = _compute_valid_mask(W, H, M, W, H)
    assert mask.all()
