"""
test_gsd_match.py — Unit test per lo Stage 4 (GSD match).

Criteri di successo (spec §5):
  - GSD calcolato corretto (formula pixel_pitch * alt / focal)
  - Downsample quando drone più dettagliato del satellite (scale < 1)
  - Upsample quando drone meno dettagliato (scale > 1)
  - Scale = 1 → dimensioni invariate
  - Maschera rescalata coerentemente con l'immagine
  - Maschera None → GSDMatchResult.mask è None
  - Output dtype uint8, maschera bool
  - Validazione input (alt <= 0, satellite_gsd <= 0)
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_denied_nav.config import CameraConfig
from gnss_denied_nav.preprocessing.gsd_match import (
    GSDMatchResult,
    _resize,
    compute_gsd,
    gsd_match,
)

# ── fixture ───────────────────────────────────────────────────────────────────

W, H = 320, 240
PIXEL_PITCH_UM = 1.5
FX = FY = 500.0  # pixel


def _make_cfg() -> CameraConfig:
    return CameraConfig(
        camera_type="pinhole",
        camera_orientation="downward",
        pixel_pitch_um=PIXEL_PITCH_UM,
        K=np.array([[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(4),
        image_size=(W, H),
    )


CFG = _make_cfg()


def _blank() -> np.ndarray:
    return np.full((H, W, 3), 100, dtype=np.uint8)


def _blank_mask() -> np.ndarray:
    return np.ones((H, W), dtype=bool)


# ── test compute_gsd ──────────────────────────────────────────────────────────


def test_compute_gsd_formula() -> None:
    """
    GSD = pixel_pitch_m * alt / (fx * pixel_pitch_m) = alt / fx.
    Con alt=50m, fx=500px → GSD = 0.1 m/px.
    """
    gsd = compute_gsd(PIXEL_PITCH_UM, alt_agl_m=50.0, K=CFG.K)
    expected = 50.0 / FX  # alt / fx
    assert abs(gsd - expected) < 1e-10


def test_compute_gsd_increases_with_altitude() -> None:
    gsd_low = compute_gsd(PIXEL_PITCH_UM, 30.0, CFG.K)
    gsd_high = compute_gsd(PIXEL_PITCH_UM, 100.0, CFG.K)
    assert gsd_high > gsd_low


def test_compute_gsd_proportional_to_altitude() -> None:
    gsd_50 = compute_gsd(PIXEL_PITCH_UM, 50.0, CFG.K)
    gsd_100 = compute_gsd(PIXEL_PITCH_UM, 100.0, CFG.K)
    assert abs(gsd_100 / gsd_50 - 2.0) < 1e-10


# ── test tipo di ritorno ──────────────────────────────────────────────────────


def test_returns_gsd_match_result() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    assert isinstance(result, GSDMatchResult)


# ── test downsample (drone più dettagliato del satellite) ─────────────────────
# alt=50m, fx=500 → gsd_drone=0.1 m/px < sat=0.3 → scale=0.1/0.3 ≈ 0.33


def test_downsample_scale_less_than_1() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    assert result.scale < 1.0


def test_downsample_image_smaller() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    h_out, w_out = result.image.shape[:2]
    assert w_out < W and h_out < H


def test_downsample_scale_value() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    expected_scale = (50.0 / FX) / 0.3
    assert abs(result.scale - expected_scale) < 1e-10


# ── test upsample (drone meno dettagliato del satellite) ──────────────────────
# alt=1000m, fx=500 → gsd_drone=2.0 m/px > sat=0.3 → scale=2.0/0.3 ≈ 6.67


def test_upsample_scale_greater_than_1() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=1000.0, satellite_gsd_m=0.3)
    assert result.scale > 1.0


def test_upsample_image_larger() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=1000.0, satellite_gsd_m=0.3)
    h_out, w_out = result.image.shape[:2]
    assert w_out > W and h_out > H


# ── test scale = 1 ───────────────────────────────────────────────────────────
# alt tale che gsd_drone == satellite_gsd → scale = 1 → stesse dimensioni


def test_scale_one_preserves_size() -> None:
    # gsd_drone = alt / fx → alt = gsd_sat * fx
    alt = 0.3 * FX  # = 150m → gsd_drone = 0.3 m/px = satellite_gsd
    result = gsd_match(_blank(), CFG, alt_agl_m=alt, satellite_gsd_m=0.3)
    assert abs(result.scale - 1.0) < 1e-10
    assert result.image.shape[:2] == (H, W)


# ── test gsd_drone_m nel risultato ───────────────────────────────────────────


def test_gsd_drone_m_matches_compute_gsd() -> None:
    alt = 50.0
    sat_gsd = 0.3
    result = gsd_match(_blank(), CFG, alt_agl_m=alt, satellite_gsd_m=sat_gsd)
    expected = compute_gsd(PIXEL_PITCH_UM, alt, CFG.K)
    assert abs(result.gsd_drone_m - expected) < 1e-10


# ── test maschera ─────────────────────────────────────────────────────────────


def test_mask_none_when_not_provided() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    assert result.mask is None


def test_mask_rescaled_same_shape_as_image() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3, mask=_blank_mask())
    assert result.mask is not None
    assert result.mask.shape == result.image.shape[:2]


def test_mask_dtype_bool_after_resize() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3, mask=_blank_mask())
    assert result.mask is not None
    assert result.mask.dtype == bool


def test_all_true_mask_stays_all_true_after_resize() -> None:
    """Maschera tutta True → dopo rescale è ancora tutta True."""
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3, mask=_blank_mask())
    assert result.mask is not None
    assert result.mask.all()


# ── test dtype output ─────────────────────────────────────────────────────────


def test_output_image_dtype_uint8() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    assert result.image.dtype == np.uint8


def test_output_image_3_channels() -> None:
    result = gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.3)
    assert result.image.ndim == 3 and result.image.shape[2] == 3


# ── test validazione input ────────────────────────────────────────────────────


def test_negative_altitude_raises() -> None:
    with pytest.raises(ValueError, match="alt_agl_m"):
        gsd_match(_blank(), CFG, alt_agl_m=-1.0, satellite_gsd_m=0.3)


def test_zero_altitude_raises() -> None:
    with pytest.raises(ValueError, match="alt_agl_m"):
        gsd_match(_blank(), CFG, alt_agl_m=0.0, satellite_gsd_m=0.3)


def test_zero_satellite_gsd_raises() -> None:
    with pytest.raises(ValueError, match="satellite_gsd_m"):
        gsd_match(_blank(), CFG, alt_agl_m=50.0, satellite_gsd_m=0.0)


# ── test _resize helper ───────────────────────────────────────────────────────


def test_resize_scale_half() -> None:
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 128
    out = _resize(arr, 0.5, binary=False)
    assert out.shape == (50, 100, 3)


def test_resize_scale_2x() -> None:
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 128
    out = _resize(arr, 2.0, binary=False)
    assert out.shape == (200, 400, 3)


def test_resize_minimum_size_1px() -> None:
    """Scale molto piccolo → non scende sotto 1px."""
    arr = np.ones((10, 10, 3), dtype=np.uint8)
    out = _resize(arr, 0.001, binary=False)
    assert out.shape[0] >= 1 and out.shape[1] >= 1
