"""
sampler.py — Selezione dei frame da ispezionare.

Supporta due modalità:
  - percent : campione casuale di X% del dataset
  - count   : numero fisso di N immagini

Il seed random è fissato per garantire riproducibilità tra run diverse.
"""

from __future__ import annotations

import numpy as np

from gnss_denied_nav.config import InspectionConfig


def select_indices(n_frames: int, cfg: InspectionConfig) -> list[int]:
    """
    Restituisce gli indici dei frame da ispezionare, ordinati.

    Parameters
    ----------
    n_frames :
        Numero totale di frame nel dataset.
    cfg :
        Configurazione di ispezione (mode, value, seed).

    Returns
    -------
    list[int]
        Indici (0-based) dei frame selezionati, ordinati in modo crescente.
    """
    if n_frames == 0:
        return []

    rng = np.random.default_rng(cfg.seed)

    if cfg.sampling_mode == "percent":
        k = max(1, int(n_frames * cfg.sampling_value / 100))
    else:
        k = min(cfg.sampling_value, n_frames)

    k = min(k, n_frames)
    indices: np.ndarray = rng.choice(n_frames, size=k, replace=False)
    return sorted(int(i) for i in indices)
