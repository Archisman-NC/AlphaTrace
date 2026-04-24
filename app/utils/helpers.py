import os
import logging
from dotenv import load_dotenv
from langfuse import Langfuse
from contextlib import contextmanager
import time

# Load environment variables early
load_dotenv()
logger = logging.getLogger(__name__)

def safe_float(x, default=0.0):
    """Zero-crash float conversion."""
    try:
        if x is None: return default
        return float(x)
    except:
        return default

def safe_slice(obj, k=3, reverse=False):
    """Zero-crash sequence slicing."""
    try:
        if not isinstance(obj, (list, tuple)):
            return []
        if reverse:
            return obj[-k:] if len(obj) >= k else obj
        return obj[:k]
    except:
        return []

def build_stock_to_sector_map(data):
    """Maps tickers to sectors from structured list or dict."""
    if isinstance(data, dict):
        return {ticker: details.get("sector", "Unknown") for ticker, details in data.items()}
    if isinstance(data, list):
        return {item.get("ticker", item.get("symbol")): item.get("sector", "Unknown") for item in data}
    return {}

def normalize_sector(sector: str) -> str:
    if not sector: return ""
    return str(sector).strip().upper()

# Langfuse Initialization
public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip('\"\'')
secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip('\"\'')
host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip('\"\'')

try:
    langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
except:
    langfuse = None

@contextmanager
def timed_phase(name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(f"[PHASE] {name} completed in {elapsed}ms")

print("[DEBUG] AlphaTrace-Structural-V41: Initialized.")
