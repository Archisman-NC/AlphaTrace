from app.utils.helpers import safe_float
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def link_portfolio_to_sector_trends(
    sector_exposure: Dict[str, float],
    sector_trends: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Links portfolio sector exposure with market sector trends.
    Creates a unified view of weight vs performance for the reasoning engine.
    """
    if not sector_exposure:
        return {}

    linked_data = {}
    
    for sector, weight in sector_exposure.items():
        # Step 2 & 3: Iterate and handle missing data
        trend_data = sector_trends.get(sector)
        
        if trend_data:
            change = trend_data.get("change", 0.0)
            sentiment = trend_data.get("sentiment", "neutral")
        else:
            # Fallback for sectors present in portfolio but missing in market trend data
            change = 0.0
            sentiment = "neutral"
            
        # Step 4 & 5: Build output structure
        linked_data[sector] = {
            "portfolio_weight": safe_float(weight),
            "sector_change": safe_float(change),
            "sentiment": str(sentiment)
        }

    return linked_data
