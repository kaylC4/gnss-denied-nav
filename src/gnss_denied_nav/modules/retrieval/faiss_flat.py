"""Stub — RetrievalEngine brute-force coseno (baseline esatta)."""

from __future__ import annotations

import numpy as np

from gnss_denied_nav.interfaces.base import RetrievalEngine
from gnss_denied_nav.interfaces.contracts import EmbeddingBatch, LatLon, MatchResult


class FAISSFlatRetrievalEngine(RetrievalEngine):
    """
    faiss.IndexFlatIP — prodotto interno su vettori L2-normalizzati ≡ similarità coseno.
    Esatto, senza approssimazione. Usare come baseline per confronto con HNSW e MP.
    TODO: faiss.IndexFlatIP(dim), add, search top-1, faiss.write/read_index.
    """

    def __init__(self) -> None:
        self._index = None
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
        return "faiss_flat"
