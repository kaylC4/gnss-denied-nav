"""Stub — NavigationFilter EKF loosely-coupled INS + vision (RF-10/11/12)."""

from __future__ import annotations

import numpy as np

from gnss_denied_nav.interfaces.base import NavigationFilter
from gnss_denied_nav.interfaces.contracts import MatchResult, NavState


class EKFNavigationFilter(NavigationFilter):
    """
    Stato 15D: [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw,
                b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    TODO:
        - predict(): propagazione con equazioni INS discreta, aggiornamento P
        - update(): guadagno Kalman, test Mahalanobis (RF-11), aggiornamento stato
        - R adattiva da match_score e patch_size_m (RF-12)
        - Delayed measurement: compensazione latenza visione con velocità INS
    """

    def __init__(
        self,
        chi2_gate: float = 5.99,
        R_min_m2: float = 25.0,
        Q_accel: float = 0.005,
        Q_gyro: float = 0.001,
    ) -> None:
        self._chi2_gate = chi2_gate
        self._R_min = R_min_m2
        self._x = np.zeros(15)
        self._P = np.eye(15) * 1e4
        self._Q_accel = Q_accel
        self._Q_gyro = Q_gyro

    def predict(self, imu_measurement: np.ndarray) -> NavState:
        raise NotImplementedError("EKFNavigationFilter.predict() — da implementare")

    def update(self, match: MatchResult, R_measurement: np.ndarray) -> NavState:
        raise NotImplementedError("EKFNavigationFilter.update() — da implementare")

    def get_state(self) -> NavState:
        raise NotImplementedError("EKFNavigationFilter.get_state() — da implementare")

    @property
    def name(self) -> str:
        return "ekf_loosely_coupled"
