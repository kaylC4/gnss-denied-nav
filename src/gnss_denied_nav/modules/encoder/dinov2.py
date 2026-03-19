"""Stub — FeatureEncoder DINOv2 via PyTorch Hub."""
from __future__ import annotations
import numpy as np
from gnss_denied_nav.interfaces.base import FeatureEncoder
from gnss_denied_nav.interfaces.contracts import EmbeddingBatch

class DINOv2FeatureEncoder(FeatureEncoder):
    """
    TODO:
        - torch.hub.load("facebookresearch/dinov2", model_name)
        - Preprocessing torchvision.transforms con ImageNet stats
        - torch.no_grad() + .cpu().numpy() per output
        - L2-normalizzazione
    """
    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        self._model_name = model_name
        self._model      = None
        self._dim: int   = 384  # vits14 default

    def encode(self, images: np.ndarray, timestamp_ns: int = 0) -> EmbeddingBatch:
        raise NotImplementedError("DINOv2FeatureEncoder.encode() — da implementare")

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "dinov2"
