"""Stub — PoseEstimator basato su INS + quota AGL."""

from __future__ import annotations

import numpy as np

from gnss_denied_nav.interfaces.base import PoseEstimator
from gnss_denied_nav.interfaces.contracts import CameraPose


class InsRadAltPoseEstimator(PoseEstimator):
    """
    Deriva la posa camera dal quaternione IMU integrato e dalla quota AGL.

    TODO:
        - Integrare giroscopio con filtro complementare o Madgwick
        - Compensare offset fisico camera↔IMU (T_cam_imu da calibration.yaml)
        - Aggiungere correzione magnetometro per yaw assoluto
    """

    def __init__(self, T_cam_imu: list[list[float]] | None = None) -> None:
        self._T = np.array(T_cam_imu) if T_cam_imu else np.eye(4)

    def estimate(
        self,
        imu_window: np.ndarray,
        alt_agl_m: float,
    ) -> CameraPose:
        # Placeholder: usa l'ultimo campione IMU per derivare assetto
        # imu_window shape: (N, 7) [ts_ns, ax, ay, az, gx, gy, gz]
        raise NotImplementedError("InsRadAltPoseEstimator.estimate() — da implementare")

    @property
    def name(self) -> str:
        return "ins_radalt"
