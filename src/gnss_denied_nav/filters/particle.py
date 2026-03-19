"""Stub — NavigationFilter Particle Filter (alternativa non-lineare)."""
from __future__ import annotations
import numpy as np
from gnss_denied_nav.interfaces.base import NavigationFilter
from gnss_denied_nav.interfaces.contracts import MatchResult, NavState

class ParticleNavigationFilter(NavigationFilter):
    """
    TODO: inizializzare particelle attorno all'ultimo GPS fix,
    propagare con modello cinematico + rumore, pesare con match_score,
    resampling sistematico, stima da media pesata.
    """
    def __init__(self, n_particles: int = 500) -> None:
        self._n = n_particles

    def predict(self, imu_measurement: np.ndarray) -> NavState:
        raise NotImplementedError

    def update(self, match: MatchResult, R_measurement: np.ndarray) -> NavState:
        raise NotImplementedError

    def get_state(self) -> NavState:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "particle_filter"
