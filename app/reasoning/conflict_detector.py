import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def detect_conflicts(
    causal_chains: List[dict],
    normalized_holdings: Dict[str, dict],
    sector_trends: Dict[str, dict]
) -> List[dict]:
    """
    Detects actionable contradictions between qualitative news sentiment 
    and quantitative market/stock performance.
    """
    if not causal_chains or not normalized_holdings or not sector_trends:
        return []

    conflicts = []
    seen_conflicts = set()

    for chain in causal_chains:
        sector = chain.get("sector")
        news_sentiment = chain.get("news_sentiment")
        
        if not sector or not news_sentiment:
            continue
            
        trend_data = sector_trends.get(sector, {})
        sector_change = trend_data.get("change")
        
        if sector_change is None:
            continue
            
        stocks = chain.get("stocks", [])
        
        if not stocks: # purely sector-level conflict
            if news_sentiment == "positive" and sector_change < 0:
                conflicts.append({
                    "stock": "N/A",
                    "sector": sector,
                    "conflict": True,
                    "reason": f"Positive sector news but {sector} declined by {sector_change}%"
                })
            elif news_sentiment == "negative" and sector_change > 0:
                 conflicts.append({
                    "stock": "N/A",
                    "sector": sector,
                    "conflict": True,
                    "reason": f"Negative sector news but {sector} grew by +{sector_change}%"
                })
            continue

        for stock_symbol in stocks:
            stock_data = normalized_holdings.get(stock_symbol, {})
            stock_change = stock_data.get("day_change")
            
            if stock_change is None:
                continue
                
            conflict_reason = None
            
            # CASE 1: Positive news + negative sector or stock
            if news_sentiment == "positive" and (sector_change < 0 or stock_change < 0):
                if stock_change < 0 and sector_change >= 0:
                    conflict_reason = f"Mixed signals across holdings: positive news but {stock_symbol} declined ({stock_change}%)"
                else:
                    conflict_reason = f"Mixed signals across holdings: positive fundamental news but negative market reaction"
                    
            # CASE 1 (Symmetric): Negative news + positive sector or stock
            elif news_sentiment == "negative" and (sector_change > 0 or stock_change > 0):
                if stock_change > 0 and sector_change <= 0:
                    conflict_reason = f"Mixed signals across holdings: negative news but {stock_symbol} rallied (+{stock_change}%)"
                else:
                    conflict_reason = f"Mixed signals across holdings: negative fundamental news but positive market reaction"
                    
            # CASE 2: Sector up + stock down (regardless of news)
            elif sector_change > 0 and stock_change < 0:
                 conflict_reason = f"Stock-specific divergence: {sector} performed well but {stock_symbol} underperformed"
                 
            # CASE 2 (Symmetric): Sector down + stock up
            elif sector_change < 0 and stock_change > 0:
                 conflict_reason = f"Stock-specific divergence: {sector} trend did not fully reflect {stock_symbol} performance"
                 
            if conflict_reason:
                # De-duplicate conflicts
                conflict_key = f"{stock_symbol}_{conflict_reason}"
                if conflict_key not in seen_conflicts:
                    seen_conflicts.add(conflict_key)
                    conflicts.append({
                        "stock": stock_symbol,
                        "sector": sector,
                        "conflict": True,
                        "reason": conflict_reason
                    })

    return conflicts
