"""Stub — RetrievalEngine HNSW (ANN per produzione)."""
from __future__ import annotations

import numpy as np

from gnss_denied_nav.interfaces.base import RetrievalEngine
from gnss_denied_nav.interfaces.contracts import EmbeddingBatch, LatLon, MatchResult


class FAISSHNSWRetrievalEngine(RetrievalEngine):
    """
    faiss.IndexHNSWFlat — ANN con ottimo trade-off latency/recall.
    TODO: M e ef_construction da config, benchmark recall vs faiss_flat (RA-03).
    """
    def __init__(self, M: int = 32, ef_construction: int = 200) -> None:
        self._M   = M
        self._efc = ef_construction
        self._index   = None
        self._centers: list[LatLon] = []

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
        return "faiss_hnsw"
