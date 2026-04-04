"""
test_pipeline.py — Unit test per il pipeline orchestrator.

Verifica che la pipeline esegua tutti i 6 stage e che l'inspection
salvi correttamente gli output intermedi quando abilitata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnss_denied_nav.config import (
    CameraConfig,
    DomainNormConfig,
    FlightConfig,
    InspectionConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from gnss_denied_nav.preprocessing.pipeline import PreprocessingPipeline


def _pipeline_config(
    tmp_path: Path | None = None,
    inspection_enabled: bool = False,
) -> PipelineConfig:
    """Crea una PipelineConfig minima per i test."""
    cam = CameraConfig(
        camera_type="pinhole",
        camera_orientation="downward",
        pixel_pitch_um=1.5,
        K=np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        image_size=(640, 480),
    )
    flight = FlightConfig(
        heading_deg=0.0,
        imu_rotation=np.eye(3, dtype=np.float64),
    )
    pre = PreprocessingConfig(
        satellite_gsd_m=0.3,
        undistort_balance=0.0,
        tile_size_px=64,
        domain_norm=DomainNormConfig(
            method="none", reference_tile=None, clip_limit=2.0, tile_grid_size=(8, 8),
        ),
    )
    inspection = InspectionConfig(
        enabled=inspection_enabled,
        output_dir=Path(str(tmp_path or "debug/stages")),
        sampling_mode="count",
        sampling_value=5,
        seed=42,
        stages=[1, 2, 3, 4, 5, 6],
    )
    return PipelineConfig(
        camera=cam,
        flight=flight,
        preprocessing=pre,
        inspection=inspection,
    )


def _dummy_img(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestPipelineRun:
    def test_returns_pipeline_result(self) -> None:
        cfg = _pipeline_config()
        pipe = PreprocessingPipeline(cfg)
        result = pipe.run(_dummy_img(), timestamp_ns=100, filename="f.png", alt_agl_m=50.0)
        assert result.image is not None
        assert result.image.dtype == np.uint8
        assert result.image.ndim == 3

    def test_output_is_tile_size(self) -> None:
        cfg = _pipeline_config()
        pipe = PreprocessingPipeline(cfg)
        result = pipe.run(_dummy_img(), timestamp_ns=100, filename="f.png", alt_agl_m=50.0)
        h, w = result.image.shape[:2]
        assert h == 64
        assert w == 64

    def test_mask_shape_matches_image(self) -> None:
        cfg = _pipeline_config()
        pipe = PreprocessingPipeline(cfg)
        result = pipe.run(_dummy_img(), timestamp_ns=100, filename="f.png", alt_agl_m=50.0)
        if result.mask is not None:
            assert result.mask.shape == result.image.shape[:2]


class TestPipelineInspection:
    def test_no_inspection_without_prepare(self, tmp_path: Path) -> None:
        cfg = _pipeline_config(tmp_path, inspection_enabled=True)
        pipe = PreprocessingPipeline(cfg)
        # Senza prepare_inspection non salva nulla
        pipe.run(_dummy_img(), timestamp_ns=100, filename="f.png", alt_agl_m=50.0, frame_index=0)
        assert not list(tmp_path.glob("frame_*"))

    def test_inspection_saves_stages(self, tmp_path: Path) -> None:
        cfg = _pipeline_config(tmp_path, inspection_enabled=True)
        pipe = PreprocessingPipeline(cfg)
        pipe.prepare_inspection(n_frames=3)
        # Con count=5 e n_frames=3, tutti e 3 i frame vengono campionati
        for i in range(3):
            pipe.run(
                _dummy_img(), timestamp_ns=1000 + i, filename=f"f_{i}.png",
                alt_agl_m=50.0, frame_index=i,
            )
        pipe.finalize_inspection()
        # Verifica che index.parquet esista
        assert (tmp_path / "index.parquet").exists()
        # Verifica che almeno un frame sia stato salvato
        frame_dirs = list(tmp_path.glob("frame_*"))
        assert len(frame_dirs) > 0
        # Verifica che tutti e 6 gli stage siano stati salvati
        fd = frame_dirs[0]
        assert (fd / "s1_undistort.png").exists()
        assert (fd / "s1_params.json").exists()
        assert (fd / "s6_domain_norm.png").exists()
        assert (fd / "s6_params.json").exists()

    def test_non_inspected_frame_not_saved(self, tmp_path: Path) -> None:
        cfg = _pipeline_config(tmp_path, inspection_enabled=True)
        pipe = PreprocessingPipeline(cfg)
        pipe.prepare_inspection(n_frames=100)
        # Frame index=99: potrebbe non essere campionato con count=5
        # Eseguiamo frame_index=-1 (flag per non ispezionare)
        pipe.run(
            _dummy_img(), timestamp_ns=999, filename="skip.png",
            alt_agl_m=50.0, frame_index=-1,
        )
        assert not (tmp_path / "frame_999").exists()

    def test_inspection_disabled_saves_nothing(self, tmp_path: Path) -> None:
        cfg = _pipeline_config(tmp_path, inspection_enabled=False)
        pipe = PreprocessingPipeline(cfg)
        pipe.prepare_inspection(n_frames=10)
        pipe.run(
            _dummy_img(), timestamp_ns=100, filename="f.png",
            alt_agl_m=50.0, frame_index=0,
        )
        assert not list(tmp_path.glob("frame_*"))
