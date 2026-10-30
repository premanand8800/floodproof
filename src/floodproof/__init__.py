"""FloodProof: autonomous disaster-media investigation agent."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FLOODPROOF_ROOT", Path(__file__).resolve().parents[2]))
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
# keep every temp file inside the project cache (the system /tmp is not used: disk is nearly full)
(CACHE / "tmp").mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(CACHE / "tmp"))

__version__ = "0.1.0"


def load_env():
    """Load .env (OPENROUTER_API_KEY, TAVILY_API_KEY) without ever printing it."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
