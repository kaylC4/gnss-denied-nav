"""
test_domain_norm.py — Unit test per lo Stage 6 (domain normalization).

Criteri di successo (spec §7):
  - method='none': immagine invariata
  - method='clahe': shape e dtype preservati, contrasto modificato
  - method='hist': shape e dtype preservati, distribuzione avvicinata al ref
  - hist senza reference_tile: ValueError
  - hist con reference_tile mancante: FileNotFoundError
  - DomainNormConfig: validazione clip_limit, tile_grid_size, method
  - Compatibilità backward: YAML con stringa piatta (legacy)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gnss_denied_nav.config import DomainNormConfig, PipelineConfig
from gnss_denied_nav.preprocessing.domain_norm import domain_normalize

# ── fixture ───────────────────────────────────────────────────────────────────

H, W = 64, 64


def _cfg(
    method: str = "none",
    reference_tile: Path | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> DomainNormConfig:
    return DomainNormConfig(
        method=method,  # type: ignore[arg-type]
        reference_tile=reference_tile,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def _dark() -> np.ndarray:
    """Immagine scura uniforme."""
    return np.full((H, W, 3), 30, dtype=np.uint8)


def _bright() -> np.ndarray:
    """Immagine chiara uniforme."""
    return np.full((H, W, 3), 220, dtype=np.uint8)


def _gradient() -> np.ndarray:
    """Immagine con gradiente 0→255 sull'asse x."""
    row = np.linspace(0, 255, W, dtype=np.uint8)
    channel = np.tile(row, (H, 1))  # (H, W)
    return np.stack([channel, channel, channel], axis=-1)  # (H, W, 3)


# ── test method='none' ────────────────────────────────────────────────────────


def test_none_returns_same_array_content() -> None:
    img = _gradient()
    result = domain_normalize(img, _cfg("none"))
    np.testing.assert_array_equal(result, img)


def test_none_shape_preserved() -> None:
    result = domain_normalize(_dark(), _cfg("none"))
    assert result.shape == (H, W, 3)


def test_none_dtype_uint8() -> None:
    result = domain_normalize(_dark(), _cfg("none"))
    assert result.dtype == np.uint8


# ── test method='clahe' ───────────────────────────────────────────────────────


def test_clahe_shape_preserved() -> None:
    result = domain_normalize(_dark(), _cfg("clahe"))
    assert result.shape == (H, W, 3)


def test_clahe_dtype_uint8() -> None:
    result = domain_normalize(_dark(), _cfg("clahe"))
    assert result.dtype == np.uint8


def test_clahe_modifies_image() -> None:
    """CLAHE su immagine scura deve modificare i valori di pixel."""
    img = _dark()
    result = domain_normalize(img, _cfg("clahe"))
    assert not np.array_equal(result, img)


def test_clahe_increases_contrast_on_low_contrast_image() -> None:
    """CLAHE deve aumentare la std dei valori del canale L*."""
    img = np.full((H, W, 3), 80, dtype=np.uint8)
    # Aggiungi minima variazione
    img[H // 2 :, :] = 90
    result = domain_normalize(img, _cfg("clahe"))
    lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    assert lab_out[:, :, 0].std() >= lab_in[:, :, 0].std()


def test_clahe_custom_clip_limit() -> None:
    """clip_limit diverso non deve causare errori."""
    result = domain_normalize(_gradient(), _cfg("clahe", clip_limit=4.0))
    assert result.shape == (H, W, 3)


def test_clahe_custom_tile_grid() -> None:
    result = domain_normalize(_gradient(), _cfg("clahe", tile_grid_size=(4, 4)))
    assert result.shape == (H, W, 3)


# ── test method='hist' ────────────────────────────────────────────────────────


def test_hist_shape_preserved(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), _bright())
    result = domain_normalize(_dark(), _cfg("hist", reference_tile=ref_path))
    assert result.shape == (H, W, 3)


def test_hist_dtype_uint8(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), _bright())
    result = domain_normalize(_dark(), _cfg("hist", reference_tile=ref_path))
    assert result.dtype == np.uint8


def test_hist_shifts_distribution_toward_reference(tmp_path: Path) -> None:
    """
    Immagine scura → histogram matching verso immagine chiara:
    la media dell'output deve essere più alta dell'input.
    """
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), _bright())
    src = _dark()
    result = domain_normalize(src, _cfg("hist", reference_tile=ref_path))
    assert result.mean() > src.mean()


def test_hist_identical_src_and_ref_unchanged(tmp_path: Path) -> None:
    """Se src e ref hanno la stessa distribuzione, l'output è uguale all'input."""
    img = _gradient()
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), img)
    result = domain_normalize(img, _cfg("hist", reference_tile=ref_path))
    # Le distribuzioni coincidono → l'immagine non deve cambiare significativamente
    assert np.abs(result.astype(int) - img.astype(int)).mean() < 5.0


def test_hist_no_reference_tile_raises() -> None:
    with pytest.raises(ValueError, match="reference_tile"):
        domain_normalize(_dark(), _cfg("hist", reference_tile=None))


def test_hist_missing_reference_tile_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        domain_normalize(_dark(), _cfg("hist", reference_tile=tmp_path / "nonexistent.png"))


# ── test DomainNormConfig — validazione ───────────────────────────────────────


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        DomainNormConfig._from_dict({"method": "unknown"})


def test_invalid_clip_limit_raises() -> None:
    with pytest.raises(ValueError, match="clip_limit"):
        DomainNormConfig._from_dict({"method": "clahe", "clip_limit": 0.0})


def test_invalid_tile_grid_size_raises() -> None:
    with pytest.raises(ValueError, match="tile_grid_size"):
        DomainNormConfig._from_dict({"method": "clahe", "tile_grid_size": [0, 8]})


def test_valid_config_defaults() -> None:
    cfg = DomainNormConfig._from_dict({"method": "none"})
    assert cfg.method == "none"
    assert cfg.reference_tile is None
    assert cfg.clip_limit == 2.0
    assert cfg.tile_grid_size == (8, 8)


# ── test compatibilità backward YAML (stringa piatta → dict) ──────────────────


def test_legacy_string_yaml_still_loads(tmp_path: Path) -> None:
    """
    YAML con domain_normalization: none (stringa, non dict) deve ancora
    caricarsi correttamente grazie al fallback in PreprocessingConfig.
    """
    cfg_yaml = tmp_path / "test.yaml"
    cfg_yaml.write_text(
        """
sensors:
  camera:
    camera_type: pinhole
    camera_orientation: downward
    pixel_pitch: 1.5
    fx: 500.0
    fy: 500.0
    cx: 320.0
    cy: 240.0
    dist_coeffs: [0.0, 0.0, 0.0, 0.0]
    width: 640
    height: 480
    model: pinhole_radtan
  flight:
    heading: 0.0
pipeline:
  preprocessing:
    satellite_gsd: 0.3
    domain_normalization: none
"""
    )
    cfg = PipelineConfig.from_yaml(cfg_yaml)
    assert cfg.preprocessing.domain_norm.method == "none"
