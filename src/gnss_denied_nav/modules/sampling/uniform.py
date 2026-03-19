"""Stub — PatchSampler a stride uniforme con allineamento GSD."""
from __future__ import annotations
import numpy as np
from gnss_denied_nav.interfaces.base import PatchSampler
from gnss_denied_nav.interfaces.contracts import PatchSet, TileMosaic

class UniformPatchSampler(PatchSampler):
    """
    TODO:
        - Rescaling bilineare del mosaico per allineare GSD (mosaic.gsd_m → target_gsd_m)
        - Sliding window con stride configurabile
        - Calcolo coordinate geografiche del centro di ogni patch tramite interpolazione bbox
    """
    def __init__(self, stride_px: int = 64) -> None:
        self._stride = stride_px

    def sample(self, mosaic: TileMosaic, patch_size_px: int, target_gsd_m: float) -> PatchSet:
        raise NotImplementedError("UniformPatchSampler.sample() — da implementare")

    @property
    def name(self) -> str:
        return "uniform_stride"
