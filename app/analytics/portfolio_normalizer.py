import logging
from typing import Dict, Any
from app.ingestion.data_loader import DataLoader
from app.utils.helpers import safe_float

logger = logging.getLogger(__name__)

def normalize_holdings(loader: DataLoader, portfolio: dict) -> Dict[str, dict]:
    """
    Hardened normalizer for stock holdings. Converts raw data into clean,
    mathematical units [0, 1] weight and float day_change.
    """
    # Fix 9: Extract raw stocks safely from the portfolio copy
    raw_stocks = portfolio.get("stocks", [])
    stock_to_sector = loader.stock_to_sector
    
    normalized = {}
    
    for item in raw_stocks:
        # Fix 4: Symbol validation
        raw_symbol = item.get("symbol")
        if not raw_symbol or not isinstance(raw_symbol, str):
            logger.warning(f"Invalid or missing symbol in holding: {item}")
            continue
            
        symbol = raw_symbol.strip().upper()
        
        # Fix 5: Sector validation (FALLBACK TO "Other" instead of skipping)
        sector = stock_to_sector.get(symbol, "Other")
            
        # Fix 3: Explicit float casting for weight and change
        original_weight_val = item.get("weight_in_portfolio", item.get("weight_percent", item.get("weight", 0)))
        
        weight = safe_float(original_weight_val)
        
        # Fix 1: Weight normalization safety
        if weight > 1.0:
            weight = weight / 100.0
        
        # Fix 2: Skip invalid weights
        if weight <= 0:
            logger.warning(f"Skipping {symbol}: Invalid weight value {weight}.")
            continue
            
        # Clamp to [0, 1] for safety
        weight = max(0.0, min(1.0, weight))
            
        # Day change validation
        raw_change = item.get("day_change_percent", item.get("day_change", 0.0))
        day_change = safe_float(raw_change)
            
        # Fix 8: Ensure clean output / Fix 6: Debug field
        normalized[symbol] = {
            "sector": sector,
            "weight": weight,
            "day_change": day_change,
            "raw_weight": original_weight_val  # Optional debug field
        }
        
    return normalized
