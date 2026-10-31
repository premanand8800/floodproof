"""Minimal OpenRouter client with a cost ledger."""
from __future__ import annotations

import os
import re
import json
import threading

import requests

from . import load_env

_lock = threading.Lock()
LEDGER = {"usd": 0.0, "calls": 0}


_tl = threading.local()
META_URL = "https://api.meta.ai/v1"
# OpenRouter id -> Meta Model API id. Only Muse Spark 1.3 is served by the Meta API; Spark Open and Glimmer stay on OpenRouter.
META_MODELS = {"meta/muse-spark-1.3": "muse-spark-1.3"}
META_PRICE = (1.25e-6, 4.25e-6)      # USD per prompt / completion token; estimate only (Meta usage has no cost field)


def last_provider():
    """Provider that served this thread's most recent chat(): "meta_api" | "openrouter"."""
    return getattr(_tl, "provider", "openrouter")


def meta_enabled():
    load_env()
    return os.environ.get("FLOODPROOF_META_API", "1") != "0" and bool(os.environ.get("META_API_KEY"))


def _redact(msg):
    for k in ("META_API_KEY", "OPENROUTER_API_KEY", "BASETEN_API_KEY"):
        v = os.environ.get(k)
        if v:
            msg = msg.replace(v, "***")
    return msg


def _chat_meta(mid, msgs, max_tokens, timeout):
    r = requests.post(META_URL + "/chat/completions", headers={"Authorization": f"Bearer {os.environ['META_API_KEY']}"},
                      json={"model": mid, "max_tokens": max_tokens, "messages": msgs}, timeout=min(timeout, 120))
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"Meta API HTTP {r.status_code}: non-JSON reply")
    if r.status_code != 200 or "choices" not in j:
        raise RuntimeError(_redact(f"Meta API HTTP {r.status_code}: {str(j)[:200]}"))
    u = j.get("usage") or {}
    cost = u.get("prompt_tokens", 0) * META_PRICE[0] + u.get("completion_tokens", 0) * META_PRICE[1]
    return j["choices"][0]["message"].get("content") or "", cost


BASETEN_URL = "https://inference.baseten.co/v1"      # OpenAI-compatible Model APIs, pay per call; NOT yet verified against a live key
BASETEN_MODELS = {"meta/muse-spark-open": "meta-models/muse-spark-open-1026"}


def baseten_enabled():
    load_env()
    return os.environ.get("FLOODPROOF_BASETEN", "1") != "0" and bool(os.environ.get("BASETEN_API_KEY"))


def _chat_baseten(mid, msgs, max_tokens, timeout):
    r = requests.post(BASETEN_URL + "/chat/completions", headers={"Authorization": f"Bearer {os.environ['BASETEN_API_KEY']}"},
                      json={"model": mid, "max_tokens": max_tokens, "messages": msgs}, timeout=min(timeout, 120))
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"Baseten HTTP {r.status_code}: non-JSON reply")
    if r.status_code != 200 or "choices" not in j:
        raise RuntimeError(_redact(f"Baseten HTTP {r.status_code}: {str(j)[:200]}"))
    u = j.get("usage") or {}
    return j["choices"][0]["message"].get("content") or "", u.get("prompt_tokens", 0) * META_PRICE[0] + u.get("completion_tokens", 0) * META_PRICE[1]


def chat(model, content, max_tokens=4500, system=None, timeout=240, baseten=False):
    """content: list of OpenAI-style parts. Returns (text, cost_usd). Keys are read from env, never logged.
    Muse Spark 1.3 goes to the Meta Model API first; with baseten=True (text-only calls) Muse Spark Open goes to Baseten first; both fall back to OpenRouter on any error; last_provider() says which served it."""
    load_env()
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    if baseten and model in BASETEN_MODELS and baseten_enabled():
        try:
            txt, cost = _chat_baseten(BASETEN_MODELS[model], msgs, max_tokens, timeout)
            if not txt.strip():
                raise RuntimeError("Baseten empty reply")
            with _lock:
                LEDGER["usd"] += cost; LEDGER["calls"] += 1
            _tl.provider = "baseten"
            return txt, cost
        except Exception:
            pass
    if model in META_MODELS and meta_enabled():
        try:
            txt, cost = _chat_meta(META_MODELS[model], msgs, max_tokens, timeout)
            if not txt.strip():
                raise RuntimeError("Meta API empty reply")
            with _lock:
                LEDGER["usd"] += cost; LEDGER["calls"] += 1
            _tl.provider = "meta_api"
            return txt, cost
        except Exception:
            pass                                  # any error/timeout/429/billing: fall back to OpenRouter below
    _tl.provider = "openrouter"
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json={"model": model, "max_tokens": max_tokens, "messages": msgs, "usage": {"include": True}},
                      timeout=timeout)
    j = r.json()
    if "choices" not in j:
        raise RuntimeError(f"OpenRouter error: {str(j)[:300]}")
    u = j.get("usage") or {}
    cost = u.get("cost") or (u.get("prompt_tokens", 0) * 1.25e-6 + u.get("completion_tokens", 0) * 4.25e-6)
    with _lock:
        LEDGER["usd"] += cost; LEDGER["calls"] += 1
    return j["choices"][0]["message"].get("content") or "", cost


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON object in model reply")
    return json.loads(m.group(0))


TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"


def transcribe_meta(wav_path, timeout=120):
    """Muse Voice Transcribe via the Meta Model API (WAV mono 16-bit 16 kHz, <= 10 min). Returns {"transcript", "turns": [{start_s,end_s,text}],
    "duration_s"} with turn-level timestamps (the API gives no word-level timing). Raises on any failure; the key is never logged."""
    load_env()
    if not meta_enabled():
        raise RuntimeError("Meta API disabled or no key")
    import pathlib
    req = json.dumps({"mode": "ENDPOINTING", "model": TRANSCRIBE_MODEL, "audioEncoding": "WAV"})
    r = requests.post(META_URL + "/asr/transcribe", headers={"Authorization": f"Bearer {os.environ['META_API_KEY']}", "Accept": "application/json"},
                      files={"request": (None, req, "application/json"), "audio": ("audio.wav", pathlib.Path(wav_path).read_bytes(), "audio/wav")},
                      timeout=timeout)
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"Meta ASR HTTP {r.status_code}: non-JSON reply")
    if r.status_code != 200 or not isinstance(j, dict) or "transcript" not in j:
        raise RuntimeError(_redact(f"Meta ASR HTTP {r.status_code}: {str(j)[:200]}"))
    turns = [dict(start_s=round(t["startMs"] / 1000, 2), end_s=round(t.get("endMs", t["startMs"]) / 1000, 2), text=str(t.get("transcript") or "").strip())
             for t in (j.get("turns") or []) if isinstance(t, dict) and isinstance(t.get("startMs"), (int, float)) and str(t.get("transcript") or "").strip()]
    return dict(transcript=str(j.get("transcript") or ""), turns=turns, duration_s=(j.get("audioDurationMs") or 0) / 1000)
