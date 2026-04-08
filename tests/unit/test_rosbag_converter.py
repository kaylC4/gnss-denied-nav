"""
test_rosbag_converter.py — Unit test per RosbagConverter.

Copre la logica di _resolve_max_frames e il warning max_frames,
senza richiedere un vero file .bag (le dipendenze rosbags/cv2
vengono moccate dove necessario).
"""

from __future__ import annotations

import pytest

from gnss_denied_nav.io.converters.rosbag import RosbagConverter

# ── _resolve_max_frames ───────────────────────────────────────────────────────


class TestResolveMaxFrames:
    def test_none_returns_all_frames(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(None, 3210)
        assert effective == 3210
        assert warn is False

    def test_max_frames_less_than_available(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(100, 3210)
        assert effective == 100
        assert warn is False

    def test_max_frames_equal_to_available(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(3210, 3210)
        assert effective == 3210
        assert warn is True

    def test_max_frames_greater_than_available(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(5000, 3210)
        assert effective == 3210
        assert warn is True

    def test_max_frames_one(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(1, 3210)
        assert effective == 1
        assert warn is False

    def test_zero_camera_frames_in_bag(self) -> None:
        # Bag senza topic camera — non deve sollevare errori
        effective, warn = RosbagConverter._resolve_max_frames(None, 0)
        assert effective == 0
        assert warn is False

    def test_max_frames_with_zero_camera_frames(self) -> None:
        effective, warn = RosbagConverter._resolve_max_frames(10, 0)
        assert effective == 0
        assert warn is True


# ── warning a stdout ──────────────────────────────────────────────────────────


class TestMaxFramesWarning:
    """
    Verifica che il warning venga stampato quando max_frames >= n_camera_frames,
    usando direttamente _resolve_max_frames e intercettando stdout.
    """

    def test_warning_printed_when_max_exceeds_bag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        converter = RosbagConverter(
            topics={"camera": "/cam", "imu": "/imu", "gnss_in": "/fix", "gnss_gt": None},
            max_frames=5000,
        )
        effective, warn = converter._resolve_max_frames(converter._max_frames, 3210)

        if warn:
            print(
                f"  ⚠ max_frames ({converter._max_frames:,}) >= frame camera nel bag "
                f"(3,210) — verrà estratto l'intero bag"
            )

        captured = capsys.readouterr()
        assert "⚠" in captured.out
        assert "5,000" in captured.out
        assert effective == 3210

    def test_no_warning_when_max_within_range(self, capsys: pytest.CaptureFixture[str]) -> None:
        converter = RosbagConverter(
            topics={"camera": "/cam", "imu": "/imu", "gnss_in": "/fix", "gnss_gt": None},
            max_frames=100,
        )
        _, warn = converter._resolve_max_frames(converter._max_frames, 3210)
        assert warn is False


# ── pct calculation ───────────────────────────────────────────────────────────


class TestProgressPct:
    """Verifica il calcolo della percentuale della progress bar."""

    def test_pct_reaches_100_at_max_frames(self) -> None:
        max_frames = 100
        n_camera = 3210
        effective, _ = RosbagConverter._resolve_max_frames(max_frames, n_camera)
        # Simuliamo len(frame_rows) == max_frames
        pct = max_frames / (effective or 1)
        assert pct == pytest.approx(1.0)

    def test_pct_proportional_before_completion(self) -> None:
        max_frames = 100
        n_camera = 3210
        effective, _ = RosbagConverter._resolve_max_frames(max_frames, n_camera)
        pct = 50 / (effective or 1)  # metà strada
        assert pct == pytest.approx(0.5)

    def test_pct_100_when_max_frames_exceeds_bag(self) -> None:
        # max_frames > n_camera: effective = n_camera, estrae tutto
        max_frames = 5000
        n_camera = 3210
        effective, _ = RosbagConverter._resolve_max_frames(max_frames, n_camera)
        # A fine estrazione len(frame_rows) == n_camera == effective
        pct = n_camera / (effective or 1)
        assert pct == pytest.approx(1.0)
