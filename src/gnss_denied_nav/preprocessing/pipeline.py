"""
pipeline.py — Orchestratore dei 6 stage di preprocessing drone-to-satellite.

Esegue i 6 stage in sequenza e, se l'inspection è abilitata, salva su disco
immagini e parametri per i frame campionati.

Uso:
    from gnss_denied_nav.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(cfg)
    result = pipeline.run(img, timestamp_ns, filename, alt_agl_m)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from gnss_denied_nav.config import PipelineConfig
from gnss_denied_nav.inspection.sampler import select_indices
from gnss_denied_nav.inspection.stage_dumper import StageDumper
from gnss_denied_nav.preprocessing.crop_pad import crop_pad
from gnss_denied_nav.preprocessing.domain_norm import domain_normalize
from gnss_denied_nav.preprocessing.gsd_match import gsd_match
from gnss_denied_nav.preprocessing.north_align import north_align
from gnss_denied_nav.preprocessing.undistort import undistort
from gnss_denied_nav.preprocessing.warp_nadir import warp_to_nadir


class PipelineResult(NamedTuple):
    """Output finale della pipeline di preprocessing."""

    image: np.ndarray  # (H, W, 3) uint8 BGR — immagine pronta per l'encoder
    mask: np.ndarray | None  # (H, W) bool | None — maschera pixel validi


class PreprocessingPipeline:
    """Orchestratore dei 6 stage con supporto opzionale per inspection."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._dumper: StageDumper | None = None
        self._inspected_indices: set[int] = set()

    def prepare_inspection(self, n_frames: int) -> None:
        """
        Pre-calcola quali frame ispezionare e prepara il dumper.

        Deve essere chiamato prima di run() se si vuole l'inspection attiva.
        Senza questa chiamata, l'inspection è silenziosamente disabilitata.
        """
        if not self._cfg.inspection.enabled:
            return
        indices = select_indices(n_frames, self._cfg.inspection)
        self._inspected_indices = set(indices)
        self._dumper = StageDumper(self._cfg.inspection)

    def run(
        self,
        img: np.ndarray,
        timestamp_ns: int,
        filename: str,
        alt_agl_m: float,
        frame_index: int = -1,
    ) -> PipelineResult:
        """
        Esegue i 6 stage di preprocessing su una singola immagine.

        Parameters
        ----------
        img :
            Immagine raw (H, W, 3) uint8 BGR.
        timestamp_ns :
            Timestamp del frame in nanosecondi.
        filename :
            Nome file del frame (per l'indice di ispezione).
        alt_agl_m :
            Altitudine AGL [m] per il calcolo GSD.
        frame_index :
            Indice del frame nel dataset (per decidere se ispezionare).

        Returns
        -------
        PipelineResult
        """
        inspect = self._should_inspect(frame_index)
        cfg = self._cfg
        cam = cfg.camera
        pre = cfg.preprocessing

        if inspect:
            assert self._dumper is not None
            self._dumper.register_frame(timestamp_ns, filename)

        # ── Stage 1: Undistort ──────────────────────────────────────────────
        r1 = undistort(img, cam, balance=pre.undistort_balance)
        if inspect:
            self._dump(timestamp_ns, 1, r1.image, {
                "K_new": r1.K_new,
                "balance": pre.undistort_balance,
                "camera_type": cam.camera_type,
                "size_in": list(img.shape[:2]),
                "size_out": list(r1.image.shape[:2]),
            })

        # ── Stage 2: Warp to Nadir ──────────────────────────────────────────
        r2 = warp_to_nadir(r1.image, r1.K_new, cfg.flight.imu_rotation, cam)
        skipped_nadir = cam.camera_orientation == "downward"
        if inspect:
            self._dump(timestamp_ns, 2, r2.image, {
                "H": r2.H,
                "skipped": skipped_nadir,
                "camera_orientation": cam.camera_orientation,
                "size_out": list(r2.image.shape[:2]),
            })

        # ── Stage 3: North Align ────────────────────────────────────────────
        r3 = north_align(r2.image, cfg.flight.heading_deg)
        valid_ratio = float(r3.mask.sum()) / r3.mask.size if r3.mask.size > 0 else 0.0
        if inspect:
            self._dump(timestamp_ns, 3, r3.image, {
                "M": r3.M,
                "heading_deg": cfg.flight.heading_deg,
                "size_out": list(r3.image.shape[:2]),
                "valid_pixel_ratio": round(valid_ratio, 4),
            }, mask=r3.mask)

        # ── Stage 4: GSD Match ──────────────────────────────────────────────
        r4 = gsd_match(
            r3.image, cam, alt_agl_m, pre.satellite_gsd_m, mask=r3.mask,
        )
        if inspect:
            self._dump(timestamp_ns, 4, r4.image, {
                "scale": r4.scale,
                "gsd_drone_m": r4.gsd_drone_m,
                "gsd_satellite_m": pre.satellite_gsd_m,
                "direction": "downsample" if r4.scale < 1.0 else "upsample",
                "size_out": list(r4.image.shape[:2]),
            })

        # ── Stage 5: Crop/Pad ──────────────────────────────────────────────
        r5 = crop_pad(r4.image, target_size=pre.tile_size_px, mask=r4.mask)
        coverage = 0.0
        if r5.mask is not None and r5.mask.size > 0:
            coverage = float(r5.mask.sum()) / r5.mask.size
        if inspect:
            self._dump(timestamp_ns, 5, r5.image, {
                "target_size": pre.tile_size_px,
                "size_in": list(r4.image.shape[:2]),
                "size_out": list(r5.image.shape[:2]),
                "coverage_ratio": round(coverage, 4),
            })

        # ── Stage 6: Domain Normalization ───────────────────────────────────
        r6 = domain_normalize(r5.image, pre.domain_norm)
        if inspect:
            self._dump(timestamp_ns, 6, r6, {
                "method": pre.domain_norm.method,
                "clip_limit": pre.domain_norm.clip_limit,
                "tile_grid_size": list(pre.domain_norm.tile_grid_size),
                "size_out": list(r6.shape[:2]),
            })

        return PipelineResult(image=r6, mask=r5.mask)

    def finalize_inspection(self) -> None:
        """Scrive index.parquet a fine run. Chiamare dopo tutti i run()."""
        if self._dumper is not None:
            self._dumper.write_index()

    def _should_inspect(self, frame_index: int) -> bool:
        if self._dumper is None:
            return False
        if frame_index < 0:
            return False
        return frame_index in self._inspected_indices

    def _dump(
        self,
        timestamp_ns: int,
        stage: int,
        image: np.ndarray,
        params: dict[str, Any],
        mask: np.ndarray | None = None,
    ) -> None:
        if self._dumper is not None:
            self._dumper.dump_stage(timestamp_ns, stage, image, params, mask=mask)
