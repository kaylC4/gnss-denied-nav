"""Stub — RetrievalEngine SCRIMP++ stocastico."""
from __future__ import annotations
import numpy as np
from gnss_denied_nav.interfaces.base import RetrievalEngine
from gnss_denied_nav.interfaces.contracts import EmbeddingBatch, LatLon, MatchResult

class MPStochasticRetrievalEngine(RetrievalEngine):
    """
    Matrix Profile stocastico con stride = embedding_dim.
    Serie 1D = concatenazione di tutti gli embedding (n * x).
    TODO:
        - stumpy.scrimp con pearson=False e campionamento random
        - stride custom: scorrere solo a multipli di embedding_dim
        - Confrontare latenza vs FAISS nel benchmark harness (RA-03)
    """
    def __init__(self, sample_fraction: float = 0.1) -> None:
        self._frac    = sample_fraction
        self._series: np.ndarray | None = None
        self._centers: list[LatLon] = []
        self._dim: int = 0

    def build_index(self, embeddings: EmbeddingBatch, centers: list[LatLon]) -> None:
        raise NotImplementedError

    def query(self, embedding: np.ndarray, timestamp_ns: int = 0) -> MatchResult:
        raise NotImplementedError

    def save_index(self, path: str) -> None:
        raise NotImplementedError

    def load_index(self, path: str) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "mp_stochastic"
