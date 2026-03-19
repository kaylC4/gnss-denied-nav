"""
FlatDataLoader — legge il formato flat (Parquet + cartella immagini).

Struttura attesa su disco
-------------------------
<root>/
    imu.parquet      colonne: timestamp_ns, ax, ay, az, gx, gy, gz
    gnss.parquet     colonne: timestamp_ns, lat, lon, alt_wgs84_m, alt_agl_m, is_gt
    frames.parquet   colonne: timestamp_ns, filename
    images/
        <timestamp_ns>.png
        ...

Questa classe non sa nulla di ROS, bag file o altri formati sorgente.
Riceve dati già convertiti e produce SensorFrame pronti per la pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gnss_denied_nav.interfaces.base import DataLoader
from gnss_denied_nav.interfaces.contracts import LatLon, SensorFrame


class FlatDataLoader(DataLoader):
    """
    Carica una sequenza dal formato flat e la itera frame per frame.

    Parametri
    ---------
    root : str | Path
        Cartella radice della sequenza convertita.
    deny_after_ns : int | None
        Se impostato, tutti i frame con timestamp >= deny_after_ns
        avranno gnss_denied=True. Simula la perdita del segnale GPS.
        None = GPS sempre disponibile.
    """

    name = "flat"

    def __init__(self, root: str | Path, deny_after_ns: int | None = None) -> None:
        self._root = Path(root)
        self._deny_after_ns = deny_after_ns

        self._frames = pd.read_parquet(self._root / "frames.parquet")
        self._imu = pd.read_parquet(self._root / "imu.parquet")
        self._gnss = pd.read_parquet(self._root / "gnss.parquet")

        # Ordina per timestamp — necessario per le ricerche binarie
        self._frames = self._frames.sort_values("timestamp_ns").reset_index(drop=True)
        self._imu = self._imu.sort_values("timestamp_ns").reset_index(drop=True)
        self._gnss = self._gnss.sort_values("timestamp_ns").reset_index(drop=True)

        # Array NumPy pre-estratti per lookups veloci senza overhead pandas
        self._imu_ts = self._imu["timestamp_ns"].to_numpy(dtype=np.int64)
        self._gnss_ts = self._gnss["timestamp_ns"].to_numpy(dtype=np.int64)
        self._imu_data = self._imu[["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(
            dtype=np.float64
        )

    # ── interfaccia pubblica ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[SensorFrame]:
        prev_ts = int(self._imu_ts[0]) if len(self._imu_ts) > 0 else 0
        for row in self._frames.itertuples(index=False):
            yield self._build_frame(row, prev_ts)
            prev_ts = int(row.timestamp_ns)

    # ── costruzione di un singolo SensorFrame ────────────────────────────────

    def _build_frame(self, row: object, prev_ts: int) -> SensorFrame:
        ts = int(row.timestamp_ns)  # type: ignore[attr-defined]

        image = self._load_image(row.filename)  # type: ignore[attr-defined]
        imu_window = self._slice_imu(prev_ts, ts)
        gnss_fix, alt_agl_m = self._last_gnss_fix(ts)
        gnss_denied = self._deny_after_ns is not None and ts >= self._deny_after_ns

        return SensorFrame(
            timestamp_ns=ts,
            image=image,
            imu_window=imu_window,
            gnss_fix=gnss_fix,
            alt_agl_m=alt_agl_m,
            gnss_denied=gnss_denied,
        )

    def _load_image(self, filename: str) -> np.ndarray:
        path = self._root / "images" / filename
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Immagine non trovata: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # pipeline usa RGB

    def _slice_imu(self, prev_ts: int, ts: int) -> np.ndarray:
        """
        Restituisce i campioni IMU nell'intervallo (prev_ts, ts].
        Nell'asse temporale: campioni arrivati *dopo* il frame precedente
        e *entro* il frame corrente.
        """
        lo = int(np.searchsorted(self._imu_ts, prev_ts, side="right"))
        hi = int(np.searchsorted(self._imu_ts, ts, side="right"))
        return self._imu_data[lo:hi]  # shape (N, 7) — può essere (0, 7) se N=0

    def _last_gnss_fix(self, ts: int) -> tuple[LatLon | None, float]:
        """
        Restituisce l'ultimo fix GPS con timestamp <= ts.
        Usa solo i fix in ingresso (is_gt=False), non il ground truth.
        """
        # Filtra solo i fix in ingresso (non ground truth)
        mask = ~self._gnss["is_gt"].to_numpy(dtype=bool)
        ts_filtered = self._gnss_ts[mask]
        rows_filtered = self._gnss[mask]

        idx = int(np.searchsorted(ts_filtered, ts, side="right")) - 1
        if idx < 0:
            return None, 0.0

        last = rows_filtered.iloc[idx]
        return (
            LatLon(lat=float(last["lat"]), lon=float(last["lon"])),
            float(last["alt_agl_m"]),
        )
