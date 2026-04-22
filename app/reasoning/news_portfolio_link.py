import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def link_news_to_portfolio(
    filtered_news: List[dict],
    sector_exposure: Dict[str, float],
    stock_exposure_map: Dict[str, dict]
) -> List[dict]:
    """
    Filters market news signals based on the user's actual portfolio exposure.
    Determines user-relevance for the reasoning engine.
    """
    if not filtered_news:
        return []

    relevant_news = []
    
    for item in filtered_news:
        is_relevant = False
        reason = ""
        
        # Step 4: Handle Market-Scope News
        if item.get("scope") == "market":
            is_relevant = True
            reason = "Market-wide impact"
            
        else:
            # Step 2: Check Sector exposure
            affected_sectors = item.get("affected_sectors", [])
            for sector in affected_sectors:
                if sector in sector_exposure and sector_exposure[sector] > 0:
                    is_relevant = True
                    reason = f"Portfolio exposed to {sector}"
                    break
            
            # Step 3: Check Stock exposure (only if not already marked relevant)
            if not is_relevant:
                affected_stocks = item.get("affected_stocks", [])
                for stock in affected_stocks:
                    if stock in stock_exposure_map:
                        is_relevant = True
                        reason = f"Holding in {stock}"
                        break
        
        if is_relevant:
            # Step 5: Build Output
            news_record = item.copy()
            news_record["portfolio_exposed"] = True
            news_record["relevance_reason"] = reason
            relevant_news.append(news_record)
            
    return relevant_news
