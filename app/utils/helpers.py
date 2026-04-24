import os
from dotenv import load_dotenv
from langfuse import Langfuse
from contextlib import contextmanager
import time
import logging

# Load environment variables early
load_dotenv()

logger = logging.getLogger(__name__)

def safe_float(x, default=0.0):
    """
    Hardened float conversion. 
    Prevents crashes on 'critical', None, or malformed strings.
    """
    try:
        if x is None: return default
        return float(x)
    except (ValueError, TypeError):
        return default

def safe_slice(obj, k=3, reverse=False):
    """
    Zero-crash sequence slicing. 
    Always returns a list.
    """
    try:
        if not isinstance(obj, (list, tuple)):
            if obj is None: return []
            return [] # Fail silently for dicts/others
        
        if reverse:
            return obj[-k:] if len(obj) >= k else obj
        return obj[:k]
    except Exception:
        return []

def normalize_sector(sector: str) -> str:
    if not sector: return ""
    return str(sector).strip().upper()

# Langfuse Configuration
public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip('\"\'')
secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip('\"\'')
host = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")).strip('\"\'')

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

print("[DEBUG] AlphaTrace-Reliability-V40: Loaded.")
