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
    """Zero-crash float conversion with strict type guards."""
    try:
        if x is None: return default
        return float(x)
    except:
        return default

def safe_slice(obj, k=3, reverse=False):
    """Zero-crash sequence slicing. Guaranteed to return a list."""
    try:
        if not isinstance(obj, (list, tuple)):
            return []
        if reverse:
            return list(obj)[-k:] if len(obj) >= k else list(obj)
        return list(obj)[:k]
    except:
        return []

def build_stock_to_sector_map(data):
    """
    FULLY ROBUST: Maps tickers to sectors from varied formats.
    Handles malformed dicts, lists, and non-dict children.
    """
    stock_map = {}
    
    # CASE 1: Dictionary Input { "TICKER": { "sector": "..." } }
    if isinstance(data, dict):
        for ticker, details in data.items():
            if isinstance(details, dict):
                sector = details.get("sector", "Unknown")
                stock_map[str(ticker)] = str(sector)
            else:
                # Skip invalid entries silently
                continue
                
    # CASE 2: List Input [ { "ticker": "...", "sector": "..." } ]
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Check varied identity keys
                ticker = item.get("ticker", item.get("symbol", item.get("ID")))
                sector = item.get("sector", "Unknown")
                if ticker:
                    stock_map[str(ticker)] = str(sector)
            else:
                continue
                
    return stock_map

def normalize_sector(sector: str) -> str:
    if not sector: return "UNKNOWN"
    return str(sector).strip().upper()

# Langfuse Configuration
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

print("[DEBUG] AlphaTrace-Hardened-V42: Hot-Reload Active.")
