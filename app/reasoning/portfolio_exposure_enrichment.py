import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def attach_portfolio_exposure(
    sector_enriched_news: List[dict],
    sector_exposure: Dict[str, float]
) -> List[dict]:
    """
    Enriches sector-linked news signals with user-specific portfolio weights.
    Filters out news for sectors where the user has zero exposure.
    """
    if not sector_enriched_news:
        return []

    personalized_signals = []
    
    for item in sector_enriched_news:
        sector = item.get("sector")
        if not sector:
            continue
            
        # Step 2: Attach Exposure
        portfolio_weight = sector_exposure.get(sector, 0.0)
        
        # Step 4: Optional Filter - Only keep relevant signals
        if portfolio_weight <= 0:
            continue
            
        # Step 3: Build Output
        enriched_item = {
            "news": item.get("news"),
            "sector": sector,
            "sector_change": item.get("sector_change", 0.0),
            "portfolio_weight": float(portfolio_weight),
            "sector_sentiment": item.get("sector_sentiment", "neutral"),
            "news_sentiment": item.get("news_sentiment", "neutral")
        }
        
        personalized_signals.append(enriched_item)
        
    return personalized_signals
