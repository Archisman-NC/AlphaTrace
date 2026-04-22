import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_stock_level_impact(
    top_sectors: List[dict],
    stock_exposure_map: Dict[str, dict],
    top_n_per_sector: int = 2
) -> Dict[str, List[dict]]:
    """
    Identifies the top contributing stocks within the most impactful sectors.
    Provides the causal "drilled-down" evidence for reasoning.
    """
    if not top_sectors or not stock_exposure_map:
        return {}

    drilldown_map = {}
    
    for sector_data in top_sectors:
        sector_name = sector_data.get("sector")
        if not sector_name:
            continue
            
        # Collect stocks in this sector from the linked exposure map
        sector_stocks = []
        for sym, metadata in stock_exposure_map.items():
            if metadata.get("sector") == sector_name:
                sector_stocks.append({
                    "symbol": sym,
                    "weight": metadata.get("weight", 0),
                    "rank": metadata.get("importance_rank", 999)
                })
        
        # Sort by importance rank (calculated in analytics phase)
        # Handle None ranks by treating them as lowest importance (999)
        sector_stocks.sort(key=lambda x: x["rank"] if x["rank"] is not None else 999)
        
        # Select Top N
        top_n = sector_stocks[:top_n_per_sector]
        
        if top_n:
            drilldown_map[sector_name] = [
                {"symbol": s["symbol"], "impact_driver": True} for s in top_n
            ]

    return drilldown_map
