"""Test che i data contract siano immutabili e abbiano i campi attesi."""
import numpy as np
import pytest
from gnss_denied_nav.interfaces.contracts import (
    CameraPose, EmbeddingBatch, LatLon, MatchResult, NavState,
    PatchSet, TileMosaic, TransformedQuery,
)


def test_latlon_named_tuple():
    ll = LatLon(lat=45.0, lon=12.0)
    assert ll.lat == 45.0
    assert ll.lon == 12.0


def test_camera_pose_frozen():
    pose = CameraPose(
        R=np.eye(3), t=np.zeros(3), alt_agl_m=50.0, timestamp_ns=0
    )
    with pytest.raises((AttributeError, TypeError)):
        pose.alt_agl_m = 100.0  # type: ignore


def test_nav_state_fields():
    state = NavState(
        lat=45.0, lon=12.0,
        covariance_m=np.eye(2) * 25.0,
        timestamp_ns=12345,
        match_score=0.87,
        method_id="ekf_loosely_coupled",
    )
    assert state.match_score == pytest.approx(0.87)
    assert state.method_id == "ekf_loosely_coupled"
