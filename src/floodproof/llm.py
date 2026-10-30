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


def chat(model, content, max_tokens=4500, system=None, timeout=240):
    """content: list of OpenAI-style parts. Returns (text, cost_usd). Key is read from env, never logged."""
    load_env()
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
