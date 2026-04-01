"""
domain_norm.py — Stage 6: normalizzazione radiometrica (domain normalization).

Riduce il gap di dominio tra immagini drone e tile satellitari che differiscono
sistematicamente per illuminazione, bilanciamento del bianco e compressione.

Metodi disponibili (configurati in pipeline.preprocessing.domain_normalization):

  none  — nessuna modifica (default); consigliato con encoder robusti (DINOv2)
  hist  — histogram matching rispetto a un tile satellite di riferimento;
           richiede domain_normalization.reference_tile nel YAML
  clahe — CLAHE (Contrast Limited Adaptive Histogram Equalization) sul canale L*;
           utile per scene a basso contrasto o nebbiose

Riferimento: spec drone_satellite_pipeline_spec, §7.
"""

from __future__ import annotations

import cv2
import numpy as np

from gnss_denied_nav.config import DomainNormConfig


def domain_normalize(img: np.ndarray, cfg: DomainNormConfig) -> np.ndarray:
    """
    Applica la normalizzazione radiometrica configurata.

    Parameters
    ----------
    img :
        Immagine (H, W, 3) uint8 BGR — output Stage 5.
    cfg :
        Configurazione della normalizzazione (method, parametri CLAHE, ecc.).

    Returns
    -------
    np.ndarray
        Immagine normalizzata (H, W, 3) uint8 BGR, stessa shape dell'input.
    """
    if cfg.method == "none":
        return img

    if cfg.method == "hist":
        return _hist_match(img, cfg)

    # method == "clahe"
    return _clahe(img, cfg)


# ── implementazioni private ───────────────────────────────────────────────────


def _hist_match(img: np.ndarray, cfg: DomainNormConfig) -> np.ndarray:
    """
    Histogram matching: adatta la distribuzione dei colori dell'immagine drone
    a quella del tile satellite di riferimento (channel-wise).
    """
    if cfg.reference_tile is None:
        raise ValueError(
            "domain_normalization.method='hist' richiede reference_tile nel YAML, "
            "ma il valore è null."
        )
    if not cfg.reference_tile.exists():
        raise FileNotFoundError(f"reference_tile non trovato: {cfg.reference_tile}")

    ref_raw = cv2.imread(str(cfg.reference_tile))
    if ref_raw is None:
        raise ValueError(f"Impossibile leggere reference_tile: {cfg.reference_tile}")
    ref: np.ndarray = ref_raw

    result = img.copy().astype(np.float32)
    ref_f = ref.astype(np.float32)

    for ch in range(3):
        result[:, :, ch] = _match_channel(result[:, :, ch], ref_f[:, :, ch])

    out: np.ndarray = np.array(np.clip(result, 0, 255), dtype=np.uint8)
    return out


def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Histogram matching su un singolo canale usando la CDF.

    Mappa i valori di src in modo che la loro distribuzione cumulativa
    approssimi quella di ref.
    """
    src_flat = src.ravel()
    ref_flat = ref.ravel()

    src_hist, _ = np.histogram(src_flat, bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref_flat, bins=256, range=(0, 256))

    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    # Normalizza le CDF
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # Look-up table: per ogni livello di src trova il livello di ref più vicino
    lut: np.ndarray = np.searchsorted(ref_cdf, src_cdf).astype(np.float32)

    indices: np.ndarray = src.astype(np.int32)
    return np.take(lut, indices)


def _clahe(img: np.ndarray, cfg: DomainNormConfig) -> np.ndarray:
    """
    CLAHE sul canale L* (spazio CIE L*a*b*).

    Migliora il contrasto locale senza amplificare eccessivamente il rumore
    (il clip_limit limita l'amplificazione massima).
    """
    lab: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clip_limit,
        tileGridSize=cfg.tile_grid_size,
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result: np.ndarray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result
