"""
ModuleFactory — unico punto dove la config tocca le implementazioni concrete.
Nessun altro modulo deve istanziare direttamente una classe concreta.

Uso:
    factory = ModuleFactory.from_config("config/mun_frl_m600.yaml")
    encoder = factory.build("feature_encoder")
"""
from __future__ import annotations

import importlib
from typing import Any

import yaml

from gnss_denied_nav.interfaces.base import (
    FeatureEncoder,
    NavigationFilter,
    PatchSampler,
    PoseEstimator,
    RetrievalEngine,
    TileProvider,
    ViewTransformer,
)

_INTERFACE_MAP: dict[str, type] = {
    "pose_estimator":    PoseEstimator,
    "tile_provider":     TileProvider,
    "patch_sampler":     PatchSampler,
    "view_transformer":  ViewTransformer,
    "feature_encoder":   FeatureEncoder,
    "retrieval_engine":  RetrievalEngine,
    "navigation_filter": NavigationFilter,
}

# Registry: backend_name → (module_path, class_name)
_REGISTRY: dict[str, dict[str, tuple[str, str]]] = {
    "pose_estimator": {
        "ins_radalt": (
            "gnss_denied_nav.modules.pose.ins_radalt", "InsRadAltPoseEstimator"),
    },
    "tile_provider": {
        "offline_mbtiles": (
            "gnss_denied_nav.modules.tiles.mbtiles", "MBTilesTileProvider"),
        "google_maps": (
            "gnss_denied_nav.modules.tiles.google_maps", "GoogleMapsTileProvider"),
    },
    "patch_sampler": {
        "uniform_stride": (
            "gnss_denied_nav.modules.sampling.uniform", "UniformPatchSampler"),
    },
    "view_transformer": {
        "homography_inverse": (
            "gnss_denied_nav.modules.transform.homography", "HomographyViewTransformer"),
    },
    "feature_encoder": {
        "onnx": (
            "gnss_denied_nav.modules.encoder.onnx_encoder", "OnnxFeatureEncoder"),
        "dinov2": (
            "gnss_denied_nav.modules.encoder.dinov2", "DINOv2FeatureEncoder"),
    },
    "retrieval_engine": {
        "faiss_flat": (
            "gnss_denied_nav.modules.retrieval.faiss_flat", "FAISSFlatRetrievalEngine"),
        "faiss_hnsw": (
            "gnss_denied_nav.modules.retrieval.faiss_hnsw", "FAISSHNSWRetrievalEngine"),
        "mp_stochastic": (
            "gnss_denied_nav.modules.retrieval.matrix_profile", "MPStochasticRetrievalEngine"),
    },
    "navigation_filter": {
        "ekf_loosely_coupled": (
            "gnss_denied_nav.filters.ekf", "EKFNavigationFilter"),
        "particle_filter": (
            "gnss_denied_nav.filters.particle", "ParticleNavigationFilter"),
    },
}


class ModuleFactory:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @classmethod
    def from_config(cls, path: str) -> ModuleFactory:
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def build(self, module_key: str) -> object:
        """
        Istanzia il modulo richiesto leggendo backend e params dalla config.
        Lancia ValueError se il backend non è registrato.
        """
        if module_key not in _REGISTRY:
            raise ValueError(f"Modulo sconosciuto: {module_key!r}")

        section = self._config["pipeline"][module_key]
        backend = section if isinstance(section, str) else section["backend"]
        params  = section.get("params", {}) if isinstance(section, dict) else {}

        if backend not in _REGISTRY[module_key]:
            available = list(_REGISTRY[module_key])
            raise ValueError(
                f"Backend {backend!r} non trovato per {module_key!r}. "
                f"Disponibili: {available}"
            )

        module_path, class_name = _REGISTRY[module_key][backend]
        mod = importlib.import_module(module_path)
        cls_ = getattr(mod, class_name)

        # Verifica che implementi l'interfaccia corretta
        expected = _INTERFACE_MAP[module_key]
        if not issubclass(cls_, expected):
            raise TypeError(
                f"{class_name} deve estendere {expected.__name__}"
            )

        return cls_(**params)

    @staticmethod
    def register(module_key: str, backend_name: str,
                 module_path: str, class_name: str) -> None:
        """Registra un nuovo plug-in senza modificare il codice esistente."""
        if module_key not in _REGISTRY:
            _REGISTRY[module_key] = {}
        _REGISTRY[module_key][backend_name] = (module_path, class_name)
