"""
test_inspection_config.py — Unit test per InspectionConfig in config.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnss_denied_nav.config import InspectionConfig


class TestInspectionConfig:
    def test_from_dict_defaults(self) -> None:
        cfg = InspectionConfig._from_dict({"enabled": True})
        assert cfg.enabled is True
        assert cfg.output_dir == Path("debug/stages")
        assert cfg.sampling_mode == "percent"
        assert cfg.sampling_value == 10
        assert cfg.seed == 42
        assert cfg.stages == [1, 2, 3, 4, 5, 6]

    def test_from_dict_count_mode(self) -> None:
        cfg = InspectionConfig._from_dict(
            {
                "enabled": True,
                "sampling": {"mode": "count", "value": 100},
            }
        )
        assert cfg.sampling_mode == "count"
        assert cfg.sampling_value == 100

    def test_from_dict_custom_seed(self) -> None:
        cfg = InspectionConfig._from_dict({"enabled": True, "seed": 123})
        assert cfg.seed == 123

    def test_from_dict_subset_stages(self) -> None:
        cfg = InspectionConfig._from_dict({"enabled": True, "stages": [1, 3, 6]})
        assert cfg.stages == [1, 3, 6]

    def test_invalid_sampling_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="sampling.mode"):
            InspectionConfig._from_dict(
                {
                    "enabled": True,
                    "sampling": {"mode": "invalid"},
                }
            )

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError, match="sampling.value"):
            InspectionConfig._from_dict(
                {
                    "enabled": True,
                    "sampling": {"mode": "count", "value": -1},
                }
            )

    def test_percent_over_100_raises(self) -> None:
        with pytest.raises(ValueError, match="sampling.value"):
            InspectionConfig._from_dict(
                {
                    "enabled": True,
                    "sampling": {"mode": "percent", "value": 150},
                }
            )

    def test_invalid_stage_number_raises(self) -> None:
        with pytest.raises(ValueError, match="stages"):
            InspectionConfig._from_dict({"enabled": True, "stages": [0, 7]})

    def test_disabled_factory(self) -> None:
        cfg = InspectionConfig.disabled()
        assert cfg.enabled is False
        assert cfg.stages == [1, 2, 3, 4, 5, 6]
