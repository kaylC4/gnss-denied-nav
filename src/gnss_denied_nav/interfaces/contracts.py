"""
Contratti dati condivisi tra tutti i moduli.
Ogni struttura è frozen — i moduli downstream non possono modificarla in-place.
Le unità di misura sono codificate nel nome del campo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class LatLon(NamedTuple):
    lat: float  # gradi decimali WGS84
    lon: float  # gradi decimali WGS84


@dataclass(frozen=True)
class CameraPose:
    """Output di PoseEstimator."""

    R: np.ndarray  # (3, 3) float64 — matrice di rotazione camera→world
    t: np.ndarray  # (3,)   float64 — traslazione camera→world [m]
    alt_agl_m: float  # quota sopra il terreno [m]
    timestamp_ns: int  # timestamp hardware [ns]


@dataclass(frozen=True)
class TileMosaic:
    """Output di TileProvider."""

    pixels: np.ndarray  # (H, W, C) uint8
    bbox: tuple[LatLon, LatLon]  # (top_left, bottom_right)
    gsd_m: float  # ground sampling distance [m/pixel]
    timestamp_ns: int


@dataclass(frozen=True)
class PatchSet:
    """Output di PatchSampler."""

    patches: np.ndarray  # (N, H, W, C) uint8
    centers: list[LatLon]  # coordinate geografiche del centro di ogni patch
    gsd_m: float


@dataclass(frozen=True)
class TransformedQuery:
    """Output di ViewTransformer — immagine drone riproiettata in vista ortometrica."""

    pixels: np.ndarray  # (H, W, C) uint8 — stessa griglia del PatchSet
    valid_mask: np.ndarray  # (H, W)    bool  — pixel validi dopo omografia
    timestamp_ns: int


@dataclass(frozen=True)
class EmbeddingBatch:
    """Output di FeatureEncoder."""

    vectors: np.ndarray  # (N, x) float32 — N embedding di dimensione x
    l2_normed: bool  # True se già normalizzati a norma unitaria
    timestamp_ns: int


@dataclass(frozen=True)
class MatchResult:
    """Output di RetrievalEngine — top-1 match."""

    patch_idx: int
    score: float  # similarità coseno in [0, 1]
    center: LatLon  # posizione geografica del centro della patch
    timestamp_ns: int


@dataclass(frozen=True)
class NavState:
    """Output di NavigationFilter — pseudo-misura GPS sintetico."""

    lat: float
    lon: float
    covariance_m: np.ndarray  # (2, 2) float64 — covarianza posizione [m²]
    timestamp_ns: int
    match_score: float
    method_id: str  # identifica quale backend ha prodotto la stima


@dataclass(frozen=True)
class SensorFrame:
    """
    Un tick di camera con i dati di supporto sincronizzati.

    Prodotto dal DataLoader — è l'ingresso della pipeline.
    Il formato sorgente (rosbag, CSV, …) è già stato dimenticato.

    Campi
    -----
    timestamp_ns  : timestamp del frame camera [ns]
    image         : (H, W, 3) uint8 RGB
    imu_window    : (N, 7) float64  [ts_ns, ax, ay, az, gx, gy, gz]
                    tutti i campioni IMU tra il frame precedente e questo
    gnss_fix      : ultimo fix GPS valido prima di questo timestamp
                    None se non è mai arrivato nessun fix
    alt_agl_m     : quota sopra il terreno calcolata da GNSS - DEM [m]
    gnss_denied   : True se il GPS è stato negato a questo timestamp
    """

    timestamp_ns: int
    image: np.ndarray  # (H, W, 3) uint8 RGB
    imu_window: np.ndarray  # (N, 7) float64
    gnss_fix: LatLon | None  # None se nessun fix ancora ricevuto
    alt_agl_m: float
    gnss_denied: bool
