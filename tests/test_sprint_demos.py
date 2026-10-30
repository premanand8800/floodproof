"""D1: showcase demos validate, are titled, and are ordered recycled, AI, genuine, then the rest."""
import json
from pathlib import Path

from floodproof import web
from floodproof.report import validate_report

DEMO = Path(__file__).resolve().parents[1] / "demo"


def test_every_demo_validates_and_has_title():
    for d in DEMO.glob("*/report.json"):
        validate_report(json.loads(d.read_text()))
        assert d.parent.name in web.DEMO_TITLES, d.parent.name


def test_showcase_order_and_assets_exist():
    assert list(web.DEMO_TITLES)[:3] == ["success_recycled", "ai_glacier_animation", "genuine_japan_tsunami"]
    names = [x["name"] for x in web.list_demos()]
    assert names[:3] == ["success_recycled", "ai_glacier_animation", "genuine_japan_tsunami"]
    assert all(x["label"] != "INVALID" for x in web.list_demos())
    for name in ("genuine_japan_tsunami", "ai_glacier_animation"):
        r = json.loads((DEMO / name / "report.json").read_text())
        assert "runs/" not in json.dumps(r)
        for p in (r["clues"].get("frame_images") or []):
            assert (DEMO / name / "frames" / Path(p).name).is_file()
    th = {x["name"]: x["thumb"] for x in web.list_demos()}
    assert th["ai_glacier_animation"] and (DEMO / "ai_glacier_animation" / th["ai_glacier_animation"]).is_file()
    assert th["genuine_japan_tsunami"] and (DEMO / "genuine_japan_tsunami" / th["genuine_japan_tsunami"]).is_file()
