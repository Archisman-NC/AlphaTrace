import logging
from typing import Dict
from app.ingestion.data_loader import DataLoader
from app.utils.helpers import normalize_sector

logger = logging.getLogger(__name__)

def compute_sector_trends(loader: DataLoader) -> Dict[str, dict]:
    """
    Extracts sector performance trends and classifies sentiment deterministically.
    """
    market_data = loader.get_market_data()
    sector_data = market_data.get("sector_performance", {})
    
    if not sector_data:
        logger.warning("No sector performance data found.")
        return {}
        
    trends = {}
    
    for sector_name, details in sector_data.items():
        change = details.get("change_percent")
        
        # Step 6: Skip sector if change_percent is missing
        if change is None:
            logger.warning(f"Missing change_percent for sector: {sector_name}")
            continue
            
        # Step 3 & 5: Normalize and classify sentiment
        normalized_name = normalize_sector(sector_name)
        
        if change > 0.5:
            sentiment = "bullish"
        elif change < -0.5:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
            
        trends[normalized_name] = {
            "change": change,
            "sentiment": sentiment
        }
        
    return trends
