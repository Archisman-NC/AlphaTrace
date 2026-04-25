from app.utils.helpers import safe_float
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def rank_holdings(normalized_holdings: Dict[str, dict], top_n: Optional[int] = None) -> List[dict]:
    """
    Ranks portfolio holdings by their weight in descending order.
    Identifies the most impactful assets for the reasoning engine.
    """
    if not normalized_holdings:
        return []

    ranked_list = []
    
    for symbol, data in normalized_holdings.items():
        weight = data.get("weight")
        if weight is None:
            logger.warning(f"Missing weight for {symbol} in ranking, skipping.")
            continue
            
        ranked_list.append({
            "symbol": symbol,
            "weight": safe_float(weight),
            "sector": data.get("sector", "UNKNOWN"),
            "day_change": safe_float(data.get("day_change", 0.0))
        })

    # Sort by weight descending
    ranked_list.sort(key=lambda x: x["weight"], reverse=True)

    # Apply limit if provided
    if top_n is not None and top_n > 0:
        return ranked_list[:top_n]

    return ranked_list
