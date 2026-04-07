"""
test_inspection.py — Unit test per il modulo inspection (sampler + stage_dumper).

Criteri di successo:
  - Sampler: modalità percent e count producono il numero atteso di indici
  - Sampler: seed fisso → risultati riproducibili
  - Sampler: edge cases (0 frame, count > n_frames)
  - StageDumper: crea struttura cartelle corretta
  - StageDumper: salva PNG e JSON leggibili
  - StageDumper: index.parquet generato correttamente
  - StageDumper: rispetta il filtro stages
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnss_denied_nav.config import InspectionConfig
from gnss_denied_nav.inspection.sampler import select_indices
from gnss_denied_nav.inspection.stage_dumper import StageDumper

# ── fixtures ─────────────────────────────────────────────────────────────────


def _cfg(
    mode: str = "percent",
    value: int = 10,
    seed: int = 42,
    stages: list[int] | None = None,
    output_dir: str = "debug/stages",
) -> InspectionConfig:
    return InspectionConfig(
        enabled=True,
        output_dir=Path(output_dir),
        sampling_mode=mode,  # type: ignore[arg-type]
        sampling_value=value,
        seed=seed,
        stages=stages if stages is not None else [1, 2, 3, 4, 5, 6],
    )


def _img(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _mask(h: int = 64, w: int = 64) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    m[:10, :10] = False
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampler:
    def test_percent_mode_count(self) -> None:
        indices = select_indices(1000, _cfg(mode="percent", value=10))
        assert len(indices) == 100  # 10% di 1000

    def test_count_mode(self) -> None:
        indices = select_indices(1000, _cfg(mode="count", value=50))
        assert len(indices) == 50

    def test_count_exceeds_n_frames(self) -> None:
        indices = select_indices(5, _cfg(mode="count", value=100))
        assert len(indices) == 5

    def test_empty_dataset(self) -> None:
        indices = select_indices(0, _cfg())
        assert indices == []

    def test_reproducible_with_same_seed(self) -> None:
        a = select_indices(100, _cfg(seed=42))
        b = select_indices(100, _cfg(seed=42))
        assert a == b

    def test_different_seed_different_result(self) -> None:
        a = select_indices(1000, _cfg(seed=1))
        b = select_indices(1000, _cfg(seed=2))
        assert a != b

    def test_indices_sorted(self) -> None:
        indices = select_indices(500, _cfg(mode="count", value=50))
        assert indices == sorted(indices)

    def test_indices_unique(self) -> None:
        indices = select_indices(500, _cfg(mode="count", value=50))
        assert len(indices) == len(set(indices))

    def test_indices_in_range(self) -> None:
        indices = select_indices(100, _cfg(mode="count", value=20))
        assert all(0 <= i < 100 for i in indices)

    def test_percent_rounds_up_to_at_least_one(self) -> None:
        # 1% di 5 = 0.05 → arrotondato a 1
        indices = select_indices(5, _cfg(mode="percent", value=1))
        assert len(indices) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE DUMPER
# ═══════════════════════════════════════════════════════════════════════════════


class TestStageDumper:
    def test_creates_frame_directory(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        dumper.dump_stage(123456, 1, _img(), {"K_new": np.eye(3)})
        assert (tmp_path / "out" / "frame_123456" / "s1_undistort.png").exists()

    def test_saves_params_json(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        dumper.dump_stage(123456, 1, _img(), {"balance": 0.5, "size": [64, 64]})
        params_path = tmp_path / "out" / "frame_123456" / "s1_params.json"
        assert params_path.exists()
        data = json.loads(params_path.read_text())
        assert data["balance"] == 0.5
        assert data["size"] == [64, 64]

    def test_saves_mask_for_stage3(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        dumper.dump_stage(100, 3, _img(), {"M": np.eye(2, 3)}, mask=_mask())
        assert (tmp_path / "out" / "frame_100" / "s3_mask.png").exists()

    def test_skips_unselected_stages(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out"), stages=[1, 3]))
        dumper.dump_stage(100, 2, _img(), {"H": np.eye(3)})
        assert not (tmp_path / "out" / "frame_100").exists()

    def test_numpy_arrays_serialized_in_json(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        dumper.dump_stage(
            100, 1, _img(), {"K": np.array([[1.0, 0, 320], [0, 1.0, 240], [0, 0, 1]])}
        )
        data = json.loads((tmp_path / "out" / "frame_100" / "s1_params.json").read_text())
        assert isinstance(data["K"], list)
        assert data["K"][0][0] == 1.0

    def test_write_index_parquet(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        dumper.register_frame(100, "img_100.png")
        dumper.register_frame(200, "img_200.png")
        index_path = dumper.write_index()
        assert index_path.exists()
        df = pd.read_parquet(index_path)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp_ns", "filename", "frame_dir"]

    def test_all_stage_files_created(self, tmp_path: Path) -> None:
        dumper = StageDumper(_cfg(output_dir=str(tmp_path / "out")))
        for stage in range(1, 7):
            dumper.dump_stage(42, stage, _img(), {"stage": stage})
        frame_dir = tmp_path / "out" / "frame_42"
        assert (frame_dir / "s1_undistort.png").exists()
        assert (frame_dir / "s2_warp_nadir.png").exists()
        assert (frame_dir / "s3_north_align.png").exists()
        assert (frame_dir / "s4_gsd_match.png").exists()
        assert (frame_dir / "s5_crop_pad.png").exists()
        assert (frame_dir / "s6_domain_norm.png").exists()
