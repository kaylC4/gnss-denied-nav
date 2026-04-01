"""
config.py — Typed loader per la configurazione YAML della pipeline.

Legge i campi da sensors.camera, sensors.flight e pipeline.preprocessing
e li espone come dataclass tipizzate con validazione all'ingresso.

Uso:
    cfg = PipelineConfig.from_yaml("config/lighthouse_benchmarking.yaml")
    K   = cfg.camera.K                   # np.ndarray (3, 3)
    gsd = cfg.preprocessing.satellite_gsd_m
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

# ── tipi alias ───────────────────────────────────────────────────────────────

CameraType = Literal["pinhole", "fisheye"]
CameraOrientation = Literal["downward", "forward"]
NormMethod = Literal["none", "hist", "clahe"]

_VALID_CAMERA_TYPES: frozenset[str] = frozenset({"pinhole", "fisheye"})
_VALID_ORIENTATIONS: frozenset[str] = frozenset({"downward", "forward"})
_VALID_NORM_METHODS: frozenset[str] = frozenset({"none", "hist", "clahe"})


# ── DomainNormConfig ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DomainNormConfig:
    """
    Configurazione per lo Stage 6 (domain normalization).

    method          : strategia di normalizzazione radiometrica
    reference_tile  : path a PNG satellite di riferimento (solo method='hist')
    clip_limit      : intensità CLAHE — soglia di contrast limiting (solo method='clahe')
    tile_grid_size  : griglia CLAHE in pixel, es. (8, 8) (solo method='clahe')
    """

    method: NormMethod
    reference_tile: Path | None
    clip_limit: float
    tile_grid_size: tuple[int, int]

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> DomainNormConfig:
        method_raw: str = str(d.get("method", "none"))
        if method_raw not in _VALID_NORM_METHODS:
            raise ValueError(
                f"domain_normalization.method non valido: {method_raw!r}. "
                f"Atteso: {sorted(_VALID_NORM_METHODS)}"
            )
        method = cast(NormMethod, method_raw)

        ref_raw = d.get("reference_tile")
        reference_tile = Path(ref_raw) if ref_raw is not None else None

        clip_limit = float(d.get("clip_limit", 2.0))
        if clip_limit <= 0.0:
            raise ValueError(f"clip_limit deve essere > 0, ricevuto: {clip_limit}")

        tgs = d.get("tile_grid_size", [8, 8])
        if len(tgs) != 2 or tgs[0] <= 0 or tgs[1] <= 0:
            raise ValueError(f"tile_grid_size deve essere [w>0, h>0], ricevuto: {tgs}")
        tile_grid_size = (int(tgs[0]), int(tgs[1]))

        return cls(
            method=method,
            reference_tile=reference_tile,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )


# ── CameraConfig ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CameraConfig:
    """
    Parametri fisici e di calibrazione della camera.

    K            : matrice intrinseca 3×3 costruita da fx/fy/cx/cy
    dist_coeffs  : [k1,k2,p1,p2] per pinhole | [k1,k2,k3,k4] per fisheye
    image_size   : (width, height) in pixel
    pixel_pitch_um: dimensione fisica del pixel [µm] — necessaria per il GSD
    """

    camera_type: CameraType
    camera_orientation: CameraOrientation
    pixel_pitch_um: float
    K: np.ndarray  # (3, 3) float64
    dist_coeffs: np.ndarray  # (4,) o (5,) float64
    image_size: tuple[int, int]  # (width, height)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> CameraConfig:
        camera_type_raw: str = d["camera_type"]
        if camera_type_raw not in _VALID_CAMERA_TYPES:
            raise ValueError(
                f"camera_type non valido: {camera_type_raw!r}. "
                f"Atteso: {sorted(_VALID_CAMERA_TYPES)}"
            )
        camera_type = cast(CameraType, camera_type_raw)

        orientation_raw: str = d["camera_orientation"]
        if orientation_raw not in _VALID_ORIENTATIONS:
            raise ValueError(
                f"camera_orientation non valido: {orientation_raw!r}. "
                f"Atteso: {sorted(_VALID_ORIENTATIONS)}"
            )
        camera_orientation = cast(CameraOrientation, orientation_raw)

        pixel_pitch_um = float(d["pixel_pitch"])
        if pixel_pitch_um <= 0.0:
            raise ValueError(f"pixel_pitch deve essere > 0, ricevuto: {pixel_pitch_um}")

        K = np.array(
            [
                [float(d["fx"]), 0.0, float(d["cx"])],
                [0.0, float(d["fy"]), float(d["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.array(d["dist_coeffs"], dtype=np.float64)
        if camera_type == "fisheye" and len(dist_coeffs) != 4:
            raise ValueError(
                f"dist_coeffs per fisheye deve avere esattamente 4 elementi, "
                f"ricevuti {len(dist_coeffs)}"
            )
        if camera_type == "pinhole" and len(dist_coeffs) not in (4, 5):
            raise ValueError(
                f"dist_coeffs per pinhole deve avere 4 o 5 elementi, ricevuti {len(dist_coeffs)}"
            )

        return cls(
            camera_type=camera_type,
            camera_orientation=camera_orientation,
            pixel_pitch_um=pixel_pitch_um,
            K=K,
            dist_coeffs=dist_coeffs,
            image_size=(int(d["width"]), int(d["height"])),
        )


# ── FlightConfig ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FlightConfig:
    """
    Dati di volo statici (fallback).

    In produzione heading e imu_rotation vengono aggiornati per ogni frame
    leggendo i topic IMU/odometry.  Questi valori servono come default quando
    i dati live non sono disponibili (es. modalità offline, test).

    heading_deg  : heading magnetico/GPS (0=nord, CW) [gradi]
    imu_rotation : matrice di rotazione camera-to-world (3×3)
    """

    heading_deg: float
    imu_rotation: np.ndarray  # (3, 3) float64

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> FlightConfig:
        heading_deg = float(d.get("heading", 0.0))

        raw_R = d.get("imu_rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        imu_rotation = np.array(raw_R, dtype=np.float64)
        if imu_rotation.shape != (3, 3):
            raise ValueError(
                f"imu_rotation deve essere una matrice 3×3, ricevuto shape: {imu_rotation.shape}"
            )

        return cls(heading_deg=heading_deg, imu_rotation=imu_rotation)


# ── PreprocessingConfig ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Parametri della pipeline di pre-processing drone-to-satellite.

    satellite_gsd_m  : GSD target del satellite [m/px] (Stage 4)
    undistort_balance: trade-off FOV/bordi neri in [0, 1] (Stage 1)
    tile_size_px     : dimensione input encoder [px] (Stage 5)
    domain_norm      : configurazione normalizzazione radiometrica (Stage 6)
    """

    satellite_gsd_m: float
    undistort_balance: float
    tile_size_px: int
    domain_norm: DomainNormConfig

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PreprocessingConfig:
        satellite_gsd_m = float(d["satellite_gsd"])
        if satellite_gsd_m <= 0.0:
            raise ValueError(f"satellite_gsd deve essere > 0, ricevuto: {satellite_gsd_m}")

        undistort_balance = float(d.get("undistort_balance", 0.0))
        if not 0.0 <= undistort_balance <= 1.0:
            raise ValueError(
                f"undistort_balance deve essere in [0.0, 1.0], ricevuto: {undistort_balance}"
            )

        tile_size_px = int(d.get("tile_size", 512))
        if tile_size_px <= 0:
            raise ValueError(f"tile_size deve essere > 0, ricevuto: {tile_size_px}")

        norm_raw = d.get("domain_normalization", {})
        # Supporta sia il formato dict (nuovo) che stringa piatta (legacy)
        if isinstance(norm_raw, str):
            norm_raw = {"method": norm_raw}
        domain_norm = DomainNormConfig._from_dict(norm_raw)

        return cls(
            satellite_gsd_m=satellite_gsd_m,
            undistort_balance=undistort_balance,
            tile_size_px=tile_size_px,
            domain_norm=domain_norm,
        )


# ── PipelineConfig (root) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configurazione completa della pipeline drone-to-satellite.

    Caricata una volta sola all'avvio — tutti i moduli downstream
    ricevono questa istanza invece di leggere il YAML direttamente.
    """

    camera: CameraConfig
    flight: FlightConfig
    preprocessing: PreprocessingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Carica e valida la config da un file YAML."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        sensors: dict[str, Any] = raw.get("sensors", {})
        pipeline: dict[str, Any] = raw.get("pipeline", {})

        return cls(
            camera=CameraConfig._from_dict(sensors.get("camera", {})),
            flight=FlightConfig._from_dict(sensors.get("flight", {})),
            preprocessing=PreprocessingConfig._from_dict(pipeline.get("preprocessing", {})),
        )
