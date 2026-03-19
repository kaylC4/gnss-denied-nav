"""Test che la factory carichi correttamente i moduli e validi le interfacce."""
import pytest
from gnss_denied_nav.interfaces.factory import ModuleFactory
from gnss_denied_nav.interfaces.base import RetrievalEngine


def _make_factory(backend: str) -> ModuleFactory:
    config = {"pipeline": {"retrieval_engine": {"backend": backend, "params": {}}}}
    return ModuleFactory(config)


def test_factory_faiss_flat():
    factory = _make_factory("faiss_flat")
    engine = factory.build("retrieval_engine")
    assert isinstance(engine, RetrievalEngine)
    assert engine.name == "faiss_flat"


def test_factory_unknown_backend_raises():
    factory = _make_factory("nonexistent_backend")
    with pytest.raises(ValueError, match="nonexistent_backend"):
        factory.build("retrieval_engine")


def test_factory_unknown_module_raises():
    factory = ModuleFactory({"pipeline": {}})
    with pytest.raises(ValueError, match="unknown_module"):
        factory.build("unknown_module")
