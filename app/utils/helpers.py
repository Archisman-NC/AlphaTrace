import os
from dotenv import load_dotenv
from langfuse import Langfuse
from contextlib import contextmanager
import time
import logging

# Load environment variables early
load_dotenv()

logger = logging.getLogger(__name__)

def safe_slice(obj, k=3, reverse=False):
    """
    Hardened safety utility for slicing.
    Ensures zero-crash policy for non-sequence types.
    """
    try:
        if not isinstance(obj, (list, tuple)):
            return []
        
        if reverse:
            return obj[-k:] if len(obj) >= k else obj
        return obj[:k]
    except Exception as e:
        logger.error(f"safe_slice trapped an error: {e}")
        return []

print("[DEBUG] Delta-Reliability: safe_slice utility loaded.")

def normalize_sector(sector: str) -> str:
    if not sector: return ""
    return sector.strip().upper()

def build_stock_to_sector_map(sector_mapping: dict) -> dict:
    stock_map = {}
    sectors = sector_mapping.get("sectors", {})
    for sector_name, details in sectors.items():
        stocks = details.get("stocks", [])
        for stock in stocks:
            stock_map[stock] = normalize_sector(sector_name)
    return stock_map

# Langfuse Configuration
public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip('\"\'')
secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip('\"\'')
host = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")).strip('\"\'')

langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

@contextmanager
def timed_phase(name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(f"[PHASE] {name} completed in {elapsed}ms")
