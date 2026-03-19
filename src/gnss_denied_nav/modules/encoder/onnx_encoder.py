"""Stub — FeatureEncoder via ONNX Runtime (CPU embedded)."""
from __future__ import annotations
import numpy as np
from gnss_denied_nav.interfaces.base import FeatureEncoder
from gnss_denied_nav.interfaces.contracts import EmbeddingBatch

class OnnxFeatureEncoder(FeatureEncoder):
    """
    TODO:
        - Lazy load: onnxruntime.InferenceSession al primo .encode()
        - Preprocessing: resize a input_size, normalize con ImageNet mean/std
        - Batch inference, fallback su singola immagine se OOM
        - L2-normalizzazione degli output (norma unitaria per cosine similarity)
    """
    def __init__(self, checkpoint: str = "", input_size: int = 224) -> None:
        self._checkpoint = checkpoint
        self._input_size = input_size
        self._session    = None
        self._dim: int   = 0

    def encode(self, images: np.ndarray, timestamp_ns: int = 0) -> EmbeddingBatch:
        raise NotImplementedError("OnnxFeatureEncoder.encode() — da implementare")

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "onnx"
