import logging
from typing import Dict, Any, List
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

# Strict targets defined in PRD
MAJOR_INDICES = {"NIFTY50", "SENSEX", "BANKNIFTY"}

def compute_market_sentiment(loader: DataLoader) -> dict:
    """
    Computes deterministic market sentiment based on native index shifts.
    Relies purely on Phase 0 DataLoader structural payloads.
    """
    # 1. Fetch raw market data strictly through DataLoader bounds
    market_data: dict = loader.get_market_data()
    indices: dict = market_data.get("indices", {})
    
    supporting_indices: List[Dict[str, Any]] = []
    total_change = 0.0
    
    # 2 & 3. Iterate and exclusively select major indices gracefully
    for idx_key, details in indices.items():
        if idx_key in MAJOR_INDICES:
            try:
                name = details.get("name", idx_key)
                
                # Coerce cleanly allowing for native string casting overrides defensively
                change = float(details.get("change_percent", 0.0))
                
                supporting_indices.append({
                    "name": name,
                    "change": change
                })
                total_change += change
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse index change for {idx_key}: {e}")
                
    # 7. Edge Cases: Guard against total loss of connectivity or failed data targets
    if not supporting_indices:
        return {
            "market_sentiment": "neutral",
            "avg_index_change": 0.0,
            "supporting_indices": []
        }
        
    # 4. Mathematical average resolution, rounded natively
    avg_change = round(total_change / len(supporting_indices), 2)
    
    # 5. Deterministic categorical classification pipeline
    if avg_change > 0.5:
        sentiment = "bullish"
    elif avg_change < -0.5:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
        
    # 6. Structured Output Format enforcement
    return {
        "market_sentiment": sentiment,
        "avg_index_change": avg_change,
        "supporting_indices": supporting_indices
    }
