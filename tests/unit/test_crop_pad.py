"""
test_crop_pad.py — Unit test per lo Stage 5 (Crop / Pad).

Criteri di successo (spec §6):
  - Output esattamente target_size × target_size per tutti i casi
  - Immagine > target: center crop
  - Immagine < target: padding simmetrico
  - Immagine == target: nessuna modifica
  - Immagine più grande in una sola dimensione: crop + pad
  - Maschera crop/pad coerente con l'immagine
  - Padding maschera aggiunge False
  - dtype uint8 preservato, maschera bool
  - Validazione input
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_denied_nav.preprocessing.crop_pad import (
    CropPadResult,
    _crop,
    _pad,
    crop_pad,
)

TARGET = 64


def _img(h: int, w: int, fill: int = 128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _mask(h: int, w: int, val: bool = True) -> np.ndarray:
    return np.full((h, w), val, dtype=bool)


# ── test tipo di ritorno ──────────────────────────────────────────────────────


def test_returns_crop_pad_result() -> None:
    result = crop_pad(_img(TARGET, TARGET), target_size=TARGET)
    assert isinstance(result, CropPadResult)


# ── test dimensioni esatte ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "h, w",
    [
        (TARGET, TARGET),       # esatto
        (TARGET * 2, TARGET * 2),  # troppo grande
        (TARGET // 2, TARGET // 2),  # troppo piccolo
        (TARGET * 2, TARGET // 2),  # grande in H, piccolo in W
        (TARGET // 2, TARGET * 2),  # piccolo in H, grande in W
        (TARGET + 1, TARGET - 1),  # asimmetrico di 1px
    ],
)
def test_output_is_always_target_size(h: int, w: int) -> None:
    result = crop_pad(_img(h, w), target_size=TARGET)
    assert result.image.shape == (TARGET, TARGET, 3), f"input ({h},{w}) → {result.image.shape}"


# ── test center crop ──────────────────────────────────────────────────────────


def test_crop_preserves_center_content() -> None:
    """Il center crop deve mantenere il contenuto centrale."""
    img = np.zeros((TARGET * 2, TARGET * 2, 3), dtype=np.uint8)
    # Rettangolo bianco al centro
    img[TARGET // 2 : TARGET + TARGET // 2, TARGET // 2 : TARGET + TARGET // 2] = 255
    result = crop_pad(img, target_size=TARGET)
    # Il centro dell'output deve essere bianco
    cy, cx = TARGET // 2, TARGET // 2
    assert result.image[cy, cx, 0] == 255


def test_crop_symmetric() -> None:
    """Il crop deve essere simmetrico: stessa quantità rimossa a sinistra/destra."""
    img = np.zeros((TARGET * 3, TARGET * 3, 3), dtype=np.uint8)
    # Riga centrale orizzontale bianca
    img[TARGET * 3 // 2, :] = 255
    result = crop_pad(img, target_size=TARGET)
    # La riga bianca deve essere al centro del risultato
    center_row = result.image[TARGET // 2]
    assert np.any(center_row > 0)


# ── test padding ──────────────────────────────────────────────────────────────


def test_pad_value_default_zero() -> None:
    """Il padding di default deve essere nero."""
    img = _img(TARGET // 2, TARGET // 2, fill=200)
    result = crop_pad(img, target_size=TARGET)
    # Angolo in alto a sinistra deve essere nero (zona di padding)
    assert result.image[0, 0, 0] == 0


def test_pad_value_custom() -> None:
    """Il padding con valore personalizzato deve usare quel valore."""
    img = _img(TARGET // 2, TARGET // 2, fill=100)
    result = crop_pad(img, target_size=TARGET, pad_value=127)
    assert result.image[0, 0, 0] == 127


def test_pad_symmetric_even() -> None:
    """Padding pari: stessa quantità sopra/sotto e sinistra/destra."""
    # Immagine 32×32 in target 64 → pad 16 su tutti i lati
    img = _img(TARGET // 2, TARGET // 2, fill=255)
    result = crop_pad(img, target_size=TARGET)
    pad = TARGET // 4  # = 16
    # Riga pad-1 deve essere nera (padding), riga pad deve essere bianca (originale)
    assert result.image[pad - 1, TARGET // 2, 0] == 0
    assert result.image[pad, TARGET // 2, 0] == 255


# ── test immagine già a target_size ──────────────────────────────────────────


def test_exact_size_image_unchanged() -> None:
    img = _img(TARGET, TARGET, fill=77)
    result = crop_pad(img, target_size=TARGET)
    np.testing.assert_array_equal(result.image, img)


# ── test dtype ────────────────────────────────────────────────────────────────


def test_output_dtype_uint8() -> None:
    result = crop_pad(_img(TARGET * 2, TARGET // 2), target_size=TARGET)
    assert result.image.dtype == np.uint8


# ── test maschera — shape e dtype ─────────────────────────────────────────────


@pytest.mark.parametrize("h, w", [(TARGET * 2, TARGET * 2), (TARGET // 2, TARGET // 2)])
def test_mask_shape_matches_image(h: int, w: int) -> None:
    result = crop_pad(_img(h, w), target_size=TARGET, mask=_mask(h, w))
    assert result.mask is not None
    assert result.mask.shape == result.image.shape[:2]


def test_mask_dtype_bool() -> None:
    h = w = TARGET // 2
    result = crop_pad(_img(h, w), target_size=TARGET, mask=_mask(h, w))
    assert result.mask is not None
    assert result.mask.dtype == bool


def test_mask_none_when_not_provided() -> None:
    result = crop_pad(_img(TARGET, TARGET), target_size=TARGET)
    assert result.mask is None


# ── test maschera — padding aggiunge False ────────────────────────────────────


def test_mask_padding_adds_false() -> None:
    """Le regioni aggiunte dal padding devono essere False nella maschera."""
    h = w = TARGET // 2
    result = crop_pad(_img(h, w), target_size=TARGET, mask=_mask(h, w, val=True))
    assert result.mask is not None
    # Angoli dell'immagine espansa = zona di padding = False
    assert not result.mask[0, 0]
    assert not result.mask[0, TARGET - 1]
    assert not result.mask[TARGET - 1, 0]
    assert not result.mask[TARGET - 1, TARGET - 1]


def test_mask_original_region_stays_true() -> None:
    """La regione originale deve rimanere True dopo il padding."""
    h = w = TARGET // 2
    result = crop_pad(_img(h, w), target_size=TARGET, mask=_mask(h, w, val=True))
    assert result.mask is not None
    # Il centro deve essere True
    assert result.mask[TARGET // 2, TARGET // 2]


def test_mask_crop_preserves_values() -> None:
    """Dopo il crop la maschera deve contenere i valori della zona centrale."""
    h = w = TARGET * 2
    mask = np.zeros((h, w), dtype=bool)
    # Solo il centro è True
    mask[TARGET // 2 : TARGET + TARGET // 2, TARGET // 2 : TARGET + TARGET // 2] = True
    result = crop_pad(_img(h, w), target_size=TARGET, mask=mask)
    assert result.mask is not None
    assert result.mask[TARGET // 2, TARGET // 2]


# ── test validazione input ────────────────────────────────────────────────────


def test_zero_target_size_raises() -> None:
    with pytest.raises(ValueError, match="target_size"):
        crop_pad(_img(TARGET, TARGET), target_size=0)


def test_negative_target_size_raises() -> None:
    with pytest.raises(ValueError, match="target_size"):
        crop_pad(_img(TARGET, TARGET), target_size=-1)


def test_invalid_pad_value_raises() -> None:
    with pytest.raises(ValueError, match="pad_value"):
        crop_pad(_img(TARGET // 2, TARGET // 2), target_size=TARGET, pad_value=300)


# ── test helpers privati ──────────────────────────────────────────────────────


def test_crop_no_op_when_smaller() -> None:
    img = _img(TARGET // 2, TARGET // 2)
    out, _ = _crop(img, None, TARGET)
    np.testing.assert_array_equal(out, img)


def test_pad_no_op_when_larger() -> None:
    img = _img(TARGET * 2, TARGET * 2)
    out, _ = _pad(img, None, TARGET, 0)
    np.testing.assert_array_equal(out, img)
