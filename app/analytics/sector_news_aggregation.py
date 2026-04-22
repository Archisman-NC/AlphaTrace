import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def aggregate_sector_news(mapped_news: List[dict]) -> Dict[str, dict]:
    """
    Groups news by sector and computes aggregated metrics (count, net sentiment).
    Essential for mapping news pulses to structural market entities.
    """
    sector_summary = {}

    for item in mapped_news:
        sectors = item.get("affected_sectors", [])
        direction = item.get("impact_direction", 0)
        
        # Step 6: Skip item if no sectors are resolved
        if not sectors:
            continue

        for sector in sectors:
            # Step 2 & 3: Initialize and group
            if sector not in sector_summary:
                sector_summary[sector] = {
                    "news_count": 0,
                    "net_sentiment": 0,
                    "news": []
                }
            
            # Step 4: Compute aggregated metrics
            sector_summary[sector]["news_count"] += 1
            sector_summary[sector]["net_sentiment"] += direction
            sector_summary[sector]["news"].append(item)
            
    return sector_summary
