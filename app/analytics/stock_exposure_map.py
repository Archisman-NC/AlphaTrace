import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def build_stock_exposure_map(
    normalized_holdings: Dict[str, dict],
    ranked_holdings: List[dict]
) -> Dict[str, dict]:
    """
    Creates a unified mapping of stocks to their metadata, including
    importance rank derived from portfolio weight.
    """
    if not normalized_holdings:
        return {}

    # Step 2: Build Rank Index (1-based)
    # Using ranked_holdings which is already sorted by weight descending
    rank_map = {item["symbol"]: i + 1 for i, item in enumerate(ranked_holdings)}

    exposure_map = {}

    # Step 3: Build Final Mapping
    for symbol, data in normalized_holdings.items():
        sector = data.get("sector")
        weight = data.get("weight")
        
        if not sector:
            logger.warning(f"Skipping {symbol} in exposure map: Missing sector.")
            continue
            
        rank = rank_map.get(symbol)
        
        exposure_map[symbol] = {
            "sector": sector,
            "weight": weight,
            "importance_rank": rank
        }

    return exposure_map
