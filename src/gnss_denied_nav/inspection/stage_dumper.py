"""
stage_dumper.py — Salvataggio su disco di immagini e parametri per ogni stage.

Struttura su disco:
    <output_dir>/
        index.parquet
        frame_<timestamp_ns>/
            s1_undistort.png
            s1_params.json
            s2_warp_nadir.png
            s2_params.json
            s3_north_align.png
            s3_mask.png
            s3_params.json
            s4_gsd_match.png
            s4_params.json
            s5_crop_pad.png
            s5_params.json
            s6_domain_norm.png
            s6_params.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from gnss_denied_nav.config import InspectionConfig

_STAGE_NAMES: dict[int, str] = {
    1: "undistort",
    2: "warp_nadir",
    3: "north_align",
    4: "gsd_match",
    5: "crop_pad",
    6: "domain_norm",
}


class StageDumper:
    """Gestisce il salvataggio su disco degli output intermedi della pipeline."""

    def __init__(self, cfg: InspectionConfig) -> None:
        self._cfg = cfg
        self._output_dir = cfg.output_dir
        self._stages = set(cfg.stages)
        self._index_rows: list[dict[str, Any]] = []

    def frame_dir(self, timestamp_ns: int) -> Path:
        """Restituisce il percorso della cartella per un frame specifico."""
        return self._output_dir / f"frame_{timestamp_ns}"

    def dump_stage(
        self,
        timestamp_ns: int,
        stage: int,
        image: np.ndarray,
        params: dict[str, Any],
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Salva immagine + parametri JSON per uno stage specifico.

        Parameters
        ----------
        timestamp_ns :
            Timestamp del frame in nanosecondi.
        stage :
            Numero dello stage (1-6).
        image :
            Immagine (H, W, 3) uint8 BGR da salvare come PNG.
        params :
            Dizionario dei parametri e metriche da salvare come JSON.
        mask :
            Maschera opzionale (H, W) bool — salvata solo per stage 3.
        """
        if stage not in self._stages:
            return

        fdir = self.frame_dir(timestamp_ns)
        fdir.mkdir(parents=True, exist_ok=True)

        name = _STAGE_NAMES[stage]

        # Salva immagine
        img_path = fdir / f"s{stage}_{name}.png"
        cv2.imwrite(str(img_path), image)

        # Salva maschera (stage 3)
        if mask is not None:
            mask_path = fdir / f"s{stage}_mask.png"
            cv2.imwrite(str(mask_path), (mask.astype(np.uint8)) * 255)

        # Salva parametri JSON
        params_path = fdir / f"s{stage}_params.json"
        serializable = _make_serializable(params)
        params_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))

    def register_frame(self, timestamp_ns: int, filename: str) -> None:
        """Registra un frame nell'indice per il parquet finale."""
        self._index_rows.append({
            "timestamp_ns": timestamp_ns,
            "filename": filename,
            "frame_dir": str(self.frame_dir(timestamp_ns)),
        })

    def write_index(self) -> Path:
        """Scrive index.parquet con la lista dei frame ispezionati."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._output_dir / "index.parquet"
        df = pd.DataFrame(self._index_rows)
        df.to_parquet(index_path, index=False)
        return index_path


def _make_serializable(obj: Any) -> Any:
    """Converte ndarray e tipi numpy in tipi JSON-serializzabili."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
