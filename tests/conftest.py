import pytest
from pathlib import Path
from floodproof import ROOT, CACHE

VID = ROOT / "data" / "videos"


def vid(key):
    g = [p for p in VID.glob(f"{key}.*") if p.suffix != ".part"]
    return g[0] if g else None


def need(*keys):
    ps = [vid(k) for k in keys]
    if not all(ps):
        pytest.skip(f"media missing: {keys}")
    return ps


_feat_cache = {}


@pytest.fixture(autouse=True)
def isolated_gvision_state(tmp_path, monkeypatch):
    """The Vision 'billing disabled' marker lives under sources.CACHE; never let a test read or write the real one."""
    from floodproof import sources
    monkeypatch.setattr(sources, "CACHE", tmp_path / "cache")


@pytest.fixture(autouse=True)
def no_live_gate(monkeypatch):
    """pipeline.investigate must not call the network relevance gate in unit tests (gate tests use floodproof.gate directly)."""
    import types
    from floodproof import gate, pipeline
    monkeypatch.setattr(pipeline, "gate_mod", types.SimpleNamespace(
        check=lambda *a, **k: dict(decision="run", cost=0.0, seconds=0.0, metadata_signals={}, refs=[]), headline=gate.headline))


@pytest.fixture(scope="session")
def feats():
    from floodproof.matcher import ClipFeatures

    def get(key):
        if key not in _feat_cache:
            (p,) = need(key)
            _feat_cache[key] = ClipFeatures.from_path(p)
        return _feat_cache[key]
    return get


@pytest.fixture(scope="session")
def variant():
    from floodproof.matcher import ClipFeatures, VARIANTS, make_variant
    d = CACHE / "test_var"; d.mkdir(exist_ok=True)

    def get(key, name):
        (src,) = need(key)
        dst = d / f"{key}_{name}.mp4"
        if not dst.exists():
            make_variant(src, dst, VARIANTS[name])
        return ClipFeatures.from_path(dst)
    return get
