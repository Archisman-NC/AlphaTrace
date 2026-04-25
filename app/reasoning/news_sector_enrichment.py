import logging
from app.utils.helpers import safe_float
from typing import List, Dict
from app.utils.helpers import safe_float

logger = logging.getLogger(__name__)

def attach_sector_trends_to_news(
    news_links: List[dict],
    sector_trends: Dict[str, dict]
) -> List[dict]:
    """
    Enriches news signals with real-time sector performance data.
    Creates a flattened list where each entry links a news event to a specific sector movement.
    """
    if not news_links:
        return []

    enriched_results = []
    
    for item in news_links:
        headline = item.get("headline", "Unknown News")
        affected_sectors = item.get("affected_sectors", [])
        is_exposed = item.get("portfolio_exposed", False)
        
        # Step 5: Multi-sector handling - Create one entry per sector
        # This allows the reasoning engine to attribute sentiment correctly to specific reactions
        for sector in affected_sectors:
            trend_data = sector_trends.get(sector, {})
            
            # Step 3 & 6: Attach sector data or defaults
            sector_change = trend_data.get("change", 0.0)
            sector_sentiment = trend_data.get("sentiment", "neutral")
            
            # Step 4: Build Enriched Output Object
            enriched_item = {
                "news": headline,
                "sector": sector,
                "sector_change": safe_float(sector_change),
                "sector_sentiment": str(sector_sentiment),
                "portfolio_exposed": is_exposed,
                "news_sentiment": item.get("sentiment", "neutral")
            }
            
            enriched_results.append(enriched_item)
            
    return enriched_results
