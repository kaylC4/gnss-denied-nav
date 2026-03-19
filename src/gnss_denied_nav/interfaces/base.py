"""
Interfacce astratte — ogni modulo concreto implementa una di queste classi.
Nessun modulo importa direttamente un altro modulo concreto.
L'istanziazione avviene esclusivamente tramite ModuleFactory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from gnss_denied_nav.interfaces.contracts import (
    CameraPose,
    EmbeddingBatch,
    LatLon,
    MatchResult,
    NavState,
    PatchSet,
    SensorFrame,
    TileMosaic,
    TransformedQuery,
)

# ── I/O — livello indipendente dal formato sorgente ──────────────────────────


class DataLoader(ABC):
    """
    Iteratore di SensorFrame.

    Non sa nulla del formato sorgente originale (rosbag, CSV, …).
    Legge esclusivamente il formato flat (Parquet + cartella immagini)
    prodotto da un Converter.

    Uso tipico
    ----------
    loader = FlatDataLoader(root="data/quarry1_flat/")
    for frame in loader:          # frame è un SensorFrame
        pose = pose_estimator.estimate(frame.imu_window, frame.alt_agl_m)
        ...
    """

    @abstractmethod
    def __iter__(self) -> Iterator[SensorFrame]: ...

    @abstractmethod
    def __len__(self) -> int:
        """Numero totale di frame nella sequenza."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class Converter(ABC):
    """
    Converte un formato sorgente nel formato flat ottimizzato per la pipeline.

    Il formato flat prodotto è sempre:
      <output_dir>/
        imu.parquet      [timestamp_ns, ax, ay, az, gx, gy, gz]
        gnss.parquet     [timestamp_ns, lat, lon, alt_wgs84_m, alt_agl_m, is_gt]
        frames.parquet   [timestamp_ns, filename]
        images/
          <timestamp_ns>.png
          ...

    Ogni implementazione concreta gestisce un formato sorgente specifico
    (rosbag, EuRoC, HILTI, …) e produce sempre lo stesso formato flat.
    Una volta convertito, il formato sorgente non è più necessario.
    """

    @abstractmethod
    def convert(self, source_path: str, output_dir: str) -> None:
        """
        Legge source_path e scrive il formato flat in output_dir.
        Se output_dir esiste già e force=False, è un no-op.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-02 ────────────────────────────────────────────────────────────────────


class PoseEstimator(ABC):
    """
    Stima la posa della camera rispetto al terreno a partire da INS e RadAlt.
    Input:  finestra IMU (N×7 float64) + quota AGL (float)
    Output: CameraPose
    """

    @abstractmethod
    def estimate(
        self,
        imu_window: np.ndarray,  # (N, 7) [ts_ns, ax, ay, az, gx, gy, gz]
        alt_agl_m: float,
    ) -> CameraPose: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-03 ────────────────────────────────────────────────────────────────────


class TileProvider(ABC):
    """
    Fornisce il mosaico di tile satellitari per una search area.
    Input:  ultimo GPS fix valido + buffer di incertezza
    Output: TileMosaic
    """

    @abstractmethod
    def get_mosaic(
        self,
        center: LatLon,
        radius_m: float,
        gsd_m: float,
    ) -> TileMosaic: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-04 ────────────────────────────────────────────────────────────────────


class PatchSampler(ABC):
    """
    Sliding window sul mosaico con allineamento GSD.
    Input:  TileMosaic + dimensione patch attesa + posa
    Output: PatchSet
    """

    @abstractmethod
    def sample(
        self,
        mosaic: TileMosaic,
        patch_size_px: int,
        target_gsd_m: float,
    ) -> PatchSet: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-05 ────────────────────────────────────────────────────────────────────


class ViewTransformer(ABC):
    """
    Riproietta l'immagine drone nel reference frame ortometrico satellite.
    Input:  frame drone + CameraPose + intrinseci camera
    Output: TransformedQuery
    """

    @abstractmethod
    def transform(
        self,
        drone_frame: np.ndarray,  # (H, W, C) uint8
        pose: CameraPose,
        camera_matrix: np.ndarray,  # (3, 3) float64
        target_gsd_m: float,
        target_size_px: int,
    ) -> TransformedQuery: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-06 ────────────────────────────────────────────────────────────────────


class FeatureEncoder(ABC):
    """
    Estrae embedding L2-normalizzati da immagini (batch o singola).
    Input:  array di immagini (N, H, W, C) uint8
    Output: EmbeddingBatch
    """

    @abstractmethod
    def encode(
        self,
        images: np.ndarray,  # (N, H, W, C) uint8
        timestamp_ns: int = 0,
    ) -> EmbeddingBatch: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-07 ────────────────────────────────────────────────────────────────────


class RetrievalEngine(ABC):
    """
    Nearest-neighbor search tra embedding query e libreria di patch.
    Supporta build dell'indice offline e query online.
    """

    @abstractmethod
    def build_index(
        self,
        embeddings: EmbeddingBatch,
        centers: list[LatLon],
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,  # (x,) float32
        timestamp_ns: int = 0,
    ) -> MatchResult: ...

    @abstractmethod
    def save_index(self, path: str) -> None: ...

    @abstractmethod
    def load_index(self, path: str) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── RF-10 ────────────────────────────────────────────────────────────────────


class NavigationFilter(ABC):
    """
    Filtro di navigazione ibrido INS + vision.
    Prediction step pilotato da IMU ad alta frequenza.
    Update step pilotato da MatchResult a bassa frequenza.
    """

    @abstractmethod
    def predict(
        self,
        imu_measurement: np.ndarray,  # (7,) [ts_ns, ax, ay, az, gx, gy, gz]
    ) -> NavState: ...

    @abstractmethod
    def update(
        self,
        match: MatchResult,
        R_measurement: np.ndarray,  # (2, 2) covarianza rumore misura [m²]
    ) -> NavState: ...

    @abstractmethod
    def get_state(self) -> NavState: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
