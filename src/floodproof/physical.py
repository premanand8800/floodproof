"""Physical-world evidence layer: geocoding + Copernicus EMS + GloFAS/rain.

Returns independent evidence (not a score). Free sources, no API key.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import date as _date, datetime, timedelta
from pathlib import Path

import httpx

UA = "FloodProof hackathon demo, contact talkprem384@gmail.com"
CACHE_DIR = Path(".cache/physical")
NOMINATIM = "https://nominatim.openstreetmap.org/search"
EMS_LIST = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
EMS_DETAIL = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
ELEV_API = "https://api.open-meteo.com/v1/elevation"

LIMIT_TEXT = ("GloFAS ~5 km resolution cannot resolve flash floods or glacial lake outbursts; "
              "absence of a model signal is not evidence that no flood happened")
MAX_KM = 150.0
EMS_TTL_S = 6 * 3600
MIN_PEAK_M3S = 20.0  # ignore ratio spikes on tiny dry-season flows (documented deviation)
PREFERRED_CLASSES = {"natural", "waterway", "place", "boundary", "water", "landuse"}

_last_nominatim = 0.0


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers={"User-Agent": UA}, follow_redirects=True)


def _http_json(url: str, params: dict) -> object:
    """GET json; isolated so tests can monkeypatch."""
    with _client() as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()


def _cache_path(url: str, params: dict) -> Path:
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _cached_json(url: str, params: dict, use_cache: bool = True, ttl: float | None = None):
    p = _cache_path(url, params)
    if use_cache and p.exists() and (ttl is None or time.time() - p.stat().st_mtime < ttl):
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    data = _http_json(url, params)
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
        except Exception:
            pass
    return data


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- geocoding
def geocode(place: str, countrycodes: str | None = None) -> dict:
    """Nominatim search with fallbacks for comma-separated hierarchies.

    Order: full query; "name, district"; "name, country"; "name" alone. For
    queries naming a district/state (middle components) every fallback result
    must mention one of them, so we never silently return a same-named feature
    in another district; if nothing qualifies the location is unresolved.
    """
    parts = [x.strip() for x in place.split(",") if x.strip()]
    mids = [m.lower() for m in parts[1:-1]]
    attempts = [(place, mids)]
    if len(parts) > 2:
        attempts += [(f"{parts[0]}, {parts[1]}", mids), (f"{parts[0]}, {parts[-1]}", mids), (parts[0], mids)]
    elif len(parts) == 2:
        attempts.append((parts[0], []))
    loc = None
    for q, must in attempts:
        loc = _geocode_once(q, countrycodes, must)
        if loc["lat"] is not None or loc.get("error"):
            break
    loc["query"] = place
    if loc["lat"] is None:
        loc["note"] = "no acceptable match (no result, or result outside the named district/state)"
    return loc


def _geocode_once(place: str, countrycodes: str | None = None, must: list[str] | None = None) -> dict:
    global _last_nominatim
    params = {"q": place, "format": "jsonv2", "limit": 5, "accept-language": "en"}
    if countrycodes:
        params["countrycodes"] = countrycodes
    cached = _cache_path(NOMINATIM, params).exists()
    if not cached:
        wait = 1.1 - (time.time() - _last_nominatim)
        if wait > 0:
            time.sleep(wait)
    try:
        results = _cached_json(NOMINATIM, params)
    except Exception as e:  # noqa: BLE001
        return {"query": place, "name": None, "lat": None, "lon": None, "source": "nominatim",
                "confidence": "low", "alternatives": [], "error": str(e)}
    finally:
        if not cached:
            _last_nominatim = time.time()
    if must:
        results = [r for r in results if any(m in (r.get("display_name") or "").lower() for m in must)]
    if not results:
        return {"query": place, "name": None, "lat": None, "lon": None, "source": "nominatim",
                "confidence": "low", "alternatives": []}

    def rank(r):
        pref = 1 if r.get("category", r.get("class")) in PREFERRED_CLASSES else 0
        return (pref, float(r.get("importance", 0) or 0))

    ordered = sorted(results, key=rank, reverse=True)
    top = ordered[0]
    cls = top.get("category", top.get("class"))
    imp = float(top.get("importance", 0) or 0)
    if cls in PREFERRED_CLASSES and imp >= 0.4:
        conf = "high"
    elif cls in PREFERRED_CLASSES or imp >= 0.4:
        conf = "medium"
    else:
        conf = "low"
    # a whole-country match (e.g. "Nepal") is too coarse: distances from a country centroid mean nothing
    country_level = top.get("addresstype") == "country" or int(top.get("place_rank", 30) or 30) <= 4
    if country_level:
        conf = "low"
    alts = [{"name": r.get("display_name"), "lat": float(r["lat"]), "lon": float(r["lon"]),
             "class": r.get("category", r.get("class")), "type": r.get("type")} for r in ordered[1:]]
    return {"query": place, "name": top.get("display_name"), "lat": float(top["lat"]),
            "lon": float(top["lon"]), "source": "nominatim", "confidence": conf,
            "class": cls, "type": top.get("type"), "country_level": country_level, "alternatives": alts}


# ---------------------------------------------------------------- Copernicus
def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "")[:19])
    except ValueError:
        return None


def _parse_point(wkt) -> tuple[float, float] | None:
    try:
        inner = wkt.split("(")[1].split(")")[0].split()
        return float(inner[1]), float(inner[0])  # lat, lon
    except Exception:  # noqa: BLE001
        return None


def official_event(lat: float, lon: float, day: _date, window_days: int) -> dict:
    out = {"source": "copernicus_ems", "status": "NO_RECORD", "activations": []}
    lo = datetime.combine(day - timedelta(days=window_days), datetime.min.time())
    hi = datetime.combine(day + timedelta(days=window_days + 1), datetime.min.time())
    try:
        cands = []
        offset = 0
        for _ in range(15):  # at most 1500 activations
            page = _cached_json(EMS_LIST, {"limit": 100, "offset": offset}, ttl=EMS_TTL_S)
            results = page.get("results", [])
            if not results:
                break
            for a in results:
                if (a.get("category") or {}).get("slug") != "flood":
                    continue
                act = _parse_dt(a.get("activationTime"))
                if act is None or act < lo:
                    continue
                pt = _parse_point(a.get("centroid") or "")
                dist = haversine_km(lat, lon, *pt) if pt else None
                if dist is not None and dist <= MAX_KM:
                    cands.append((a, dist))
            last = _parse_dt(results[-1].get("activationTime"))
            if last and last < lo or not page.get("next"):
                break
            offset += 100
        for a, dist in cands:
            detail = {}
            try:
                d = _http_json(EMS_DETAIL, {"code": a["code"]})
                detail = (d.get("results") or [{}])[0]
            except Exception:  # noqa: BLE001
                pass
            ev = _parse_dt(detail.get("eventTime"))
            act = _parse_dt(a.get("activationTime"))
            ref = ev or act
            if not (lo <= ref < hi):
                continue
            out["activations"].append({
                "code": a["code"], "title": a.get("name"),
                "event_time": detail.get("eventTime"), "activation_time": a.get("activationTime"),
                "country": ", ".join(c.get("short_name") or c.get("name", "") for c in a.get("countries", [])),
                "distance_km": round(dist, 1), "reason": detail.get("reason") or a.get("search_snippet"),
                "url": f"https://rapidmapping.emergency.copernicus.eu/{a['code']}",
            })
        if out["activations"]:
            out["status"] = "CONFIRMED_EVENT"
    except Exception as e:  # noqa: BLE001
        out["status"] = "ERROR"
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------- hydrology
def hydrology(lat: float, lon: float, day: _date, window_days: int) -> dict:
    out = {"source": "open-meteo glofas v4", "river_discharge": None, "rain": None, "status": "UNKNOWN"}
    ws, we = day - timedelta(days=window_days), day + timedelta(days=window_days)
    start = ws - timedelta(days=30)
    steps = (-0.1, 0.0, 0.1)
    cells = [(round(lat + a, 4), round(lon + b, 4)) for a in steps for b in steps]
    errors = []
    try:
        data = _cached_json(FLOOD_API, {
            "latitude": ",".join(str(c[0]) for c in cells), "longitude": ",".join(str(c[1]) for c in cells),
            "daily": "river_discharge", "start_date": start.isoformat(), "end_date": we.isoformat()})
        if isinstance(data, dict):
            data = [data]
        best = None
        for cell, d in zip(cells, data):
            times = d["daily"]["time"]
            vals = d["daily"]["river_discharge"]
            win = [(t, v) for t, v in zip(times, vals) if v is not None and ws.isoformat() <= t <= we.isoformat()]
            base = [v for t, v in zip(times, vals) if v is not None and t < ws.isoformat()]
            if not win:
                continue
            peak = max(v for _, v in win)
            if best is None or peak > best[0]:
                best = (peak, cell, win, base, (d.get("latitude"), d.get("longitude")))
        if best is None:
            raise ValueError("no discharge data")
        peak, cell, win, base, snapped = best
        baseline = statistics.median(base) if base else None
        ratio = (peak / baseline) if baseline and baseline > 0 else None
        out["river_discharge"] = {"series": [{"date": t, "m3s": v} for t, v in win],
                                  "baseline": baseline, "peak": peak,
                                  "ratio": round(ratio, 2) if ratio is not None else None,
                                  "cell": list(snapped if snapped[0] is not None else cell)}
    except Exception as e:  # noqa: BLE001
        errors.append(f"discharge: {e}")
    try:
        d = _cached_json(ARCHIVE_API, {"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                                       "start_date": ws.isoformat(), "end_date": we.isoformat(), "timezone": "GMT"})
        series = [{"date": t, "mm": v} for t, v in zip(d["daily"]["time"], d["daily"]["precipitation_sum"])]
        vals = [s["mm"] for s in series if s["mm"] is not None]
        out["rain"] = {"series": series, "max_mm": max(vals) if vals else None}
    except Exception as e:  # noqa: BLE001
        errors.append(f"rain: {e}")
    if errors:
        out["errors"] = errors
    rd, rn = out["river_discharge"], out["rain"]
    ratio = rd and rd.get("ratio")
    mm = rn and rn.get("max_mm")
    if (ratio is not None and ratio >= 3 and rd["peak"] >= MIN_PEAK_M3S) or (mm is not None and mm >= 100):
        out["status"] = "ANOMALOUS"
    elif ratio is not None and mm is not None and ratio < 1.5 and mm < 20:
        out["status"] = "NORMAL"
    elif ratio is not None and mm is None and ratio < 1.5:
        out["status"] = "NORMAL"
    return out


def elevation(lat: float, lon: float) -> float | None:
    try:
        d = _cached_json(ELEV_API, {"latitude": lat, "longitude": lon})
        return float(d["elevation"][0])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- main
EVENT_TYPES = {"flood", "flash_flood", "glof", "landslide", "cyclone", "storm_surge", "unknown"}
NOT_RESOLVED_BY_GLOFAS = {
    "flash_flood": "flash floods are too small and fast for GloFAS river discharge",
    "glof": "glacial lake outburst floods are not represented in GloFAS river discharge",
    "landslide": "landslides are not represented in river discharge or daily rain totals",
    "cyclone": "cyclone wind/surge damage is not represented in GloFAS river discharge",
    "storm_surge": "storm surge/coastal flooding is not represented in GloFAS river discharge",
    "unknown": "event type unknown, so the river-flood model may not apply",
}


def check(place: str | None = None, lat: float | None = None, lon: float | None = None,
          date: str = "YYYY-MM-DD", window_days: int = 3, event_type: str = "flood",
          country: str | None = None) -> dict:
    if event_type not in EVENT_TYPES:
        event_type = "unknown"
    reasons: list[str] = []
    limits = [LIMIT_TEXT]
    links: list[str] = []
    if lat is not None and lon is not None:
        loc = {"query": place, "name": place, "lat": lat, "lon": lon, "source": "given",
               "confidence": "high", "alternatives": []}
    elif place:
        loc = geocode(place, country)
    else:
        loc = {"query": place, "name": None, "lat": None, "lon": None, "source": "given",
               "confidence": "low", "alternatives": []}
    res = {"location": loc, "official_event": {"source": "copernicus_ems", "status": "ERROR", "activations": []},
           "hydrology": {"source": "open-meteo glofas v4", "river_discharge": None,
                         "rain": None, "status": "UNKNOWN"},
           "verdict": "UNKNOWN", "reasons": reasons, "limits": limits, "links": links}
    try:
        day = _date.fromisoformat(date)
    except ValueError:
        reasons.append(f"invalid date {date!r}")
        return res
    if loc["lat"] is None:
        reasons.append("location could not be resolved")
        return res
    la, lo = loc["lat"], loc["lon"]
    ev = official_event(la, lo, day, window_days)
    hy = hydrology(la, lo, day, window_days)
    res["official_event"], res["hydrology"] = ev, hy
    links.append("https://mapping.emergency.copernicus.eu/activations/")
    links.append(f"https://flood-api.open-meteo.com/v1/flood?latitude={la}&longitude={lo}"
                 f"&daily=river_discharge&start_date={day - timedelta(days=window_days)}"
                 f"&end_date={day + timedelta(days=window_days)}")
    links.append(f"https://www.openstreetmap.org/?mlat={la}&mlon={lo}#map=10/{la}/{lo}")

    rd, rn = hy.get("river_discharge"), hy.get("rain")
    ratio = rd and rd.get("ratio")
    mm = rn and rn.get("max_mm")
    if loc.get("country_level"):
        reasons.append("location resolved only to a whole country; too coarse to corroborate a local event")
        return res
    support = False
    if ev["status"] == "CONFIRMED_EVENT":
        support = True
        for a in ev["activations"]:
            links.append(a["url"])
            reasons.append(f"Copernicus EMS {a['code']} ({a['title']}), {a['distance_km']} km away, "
                           f"event {a['event_time'] or a['activation_time']}")
    if ratio is not None and ratio >= 3 and rd["peak"] >= MIN_PEAK_M3S:
        support = True
        reasons.append(f"river discharge peak {rd['peak']:.1f} m3/s is {ratio}x the 30-day baseline "
                       f"{rd['baseline']:.1f} m3/s")
    if mm is not None and mm >= 100:
        support = True
        reasons.append(f"rain {mm:.0f} mm/day in window (>= 100)")
    if support:
        res["verdict"] = "SUPPORTS"
        return res

    if loc["confidence"] == "low":
        reasons.append("location confidence low")
        return res
    if ev["status"] == "ERROR" or ratio is None or mm is None:
        reasons.append("a data source failed; cannot assess")
        return res

    reasons.append(f"no Copernicus flood activation within {MAX_KM:.0f} km / +-{window_days} d")
    reasons.append(f"discharge ratio {ratio} (peak {rd['peak']:.1f} vs baseline "
                   f"{rd['baseline'] if rd['baseline'] is None else round(rd['baseline'], 1)} m3/s), max rain {mm:.1f} mm/day")
    if event_type != "flood":
        limits.append(f"event_type={event_type}: {NOT_RESOLVED_BY_GLOFAS[event_type]}; "
                      "strongest negative verdict is NO_EVIDENCE, never CONTRADICTS")
    elif ratio < 1.5 and mm < 20 and loc["confidence"] == "high":
        elev = elevation(la, lo)
        res["location"]["elevation_m"] = elev
        if elev is not None and elev < 2000:
            res["verdict"] = "CONTRADICTS"
            reasons.append(f"lowland rain-fed setting (elevation {elev:.0f} m < 2000 m) with normal flow and little rain")
            limits.append("CONTRADICTS is weak: it only means no physical trace was found at model resolution, "
                          "not that the video is fake or the event did not occur")
            return res
        reasons.append("elevation unknown or >= 2000 m: model cannot rule out flash flood / GLOF")
    res["verdict"] = "NO_EVIDENCE"
    return res
