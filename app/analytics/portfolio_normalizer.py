import logging
from typing import Dict, Any
from app.ingestion.data_loader import DataLoader

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
        
        # Fix 5: Sector validation
        sector = stock_to_sector.get(symbol)
        if sector is None:
            # Fix 7: Logging per skipped entry
            logger.warning(f"Skipping {symbol}: No sector mapping found in registry.")
            continue
            
        # Fix 3: Explicit float casting for weight and change
        original_weight_val = item.get("weight_in_portfolio", item.get("weight_percent", item.get("weight", 0)))
        try:
            weight = float(original_weight_val)
            
            # Fix 1: Weight normalization safety
            if weight > 1.0:
                weight = weight / 100.0
            
            # Fix 2: Skip invalid weights
            if weight <= 0:
                logger.warning(f"Skipping {symbol}: Invalid weight value {weight}.")
                continue
                
            # Clamp to [0, 1] for safety
            weight = max(0.0, min(1.0, weight))
            
        except (ValueError, TypeError):
            logger.warning(f"Skipping {symbol}: Non-numeric weight value '{original_weight_val}'.")
            continue
            
        # Day change validation
        raw_change = item.get("day_change_percent", item.get("day_change", 0.0))
        try:
            day_change = float(raw_change)
        except (ValueError, TypeError):
            day_change = 0.0
            
        # Fix 8: Ensure clean output / Fix 6: Debug field
        normalized[symbol] = {
            "sector": sector,
            "weight": weight,
            "day_change": day_change,
            "raw_weight": original_weight_val  # Optional debug field
        }
        
    return normalized
