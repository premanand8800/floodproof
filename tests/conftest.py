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
