import os
import logging
from dotenv import load_dotenv
from langfuse import Langfuse
from contextlib import contextmanager
import time

# Load environment variables early
load_dotenv()
logger = logging.getLogger(__name__)

def safe_float(x):
    """Hardened float conversion: No crashes."""
    try:
        return float(x)
    except:
        return 0.0


def safe_slice(x, n=3, **kwargs):
    """
    Omni-Argument slicing utility.
    Ingests and safely ignores inconsistent kwargs (k, reverse, etc.)
    """
    if "k" in kwargs:
        n = kwargs["k"]
    return x[:n] if isinstance(x, list) else x

def build_stock_to_sector_map(data):
    """
    ULTRA-ROBUST: Maps tickers to sectors from any financial data shape.
    Handles: Dict-of-Dict, Dict-of-List, List-of-Dict.
    """
    result = {}
    if not data: return result

    if isinstance(data, dict):
        for ticker, details in data.items():
            if isinstance(details, dict):
                result[str(ticker)] = str(details.get("sector", "Unknown"))
            elif isinstance(details, list):
                # Take first valid dict in the details list
                for item in details:
                    if isinstance(item, dict):
                        result[str(ticker)] = str(item.get("sector", "Unknown"))
                        break
            else:
                result[str(ticker)] = "Unknown"

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                ticker = item.get("ticker") or item.get("symbol")
                if ticker:
                    result[str(ticker)] = str(item.get("sector", "Unknown"))

    return result

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

print("[DEBUG] AlphaTrace-Hardened-V43: Hot-Reload Active.")
