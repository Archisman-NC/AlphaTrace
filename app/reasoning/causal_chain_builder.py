import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def build_causal_chains(
    enriched_news: List[dict],
    sector_impact: Dict[str, dict],
    stock_drilldown: Dict[str, List[dict]]
) -> List[dict]:
    """
    Constructs full causal chains linking news events, sector trends, 
    portfolio exposure, and specific stock drivers.
    """
    if not enriched_news or not sector_impact:
        return []

    causal_chains = []

    for item in enriched_news:
        if item.get("portfolio_exposed", False) is False and item.get("portfolio_weight", 0.0) <= 0:
            continue

        sector = item.get("sector")
        
        if not sector or sector not in sector_impact:
            continue

        impact_data = sector_impact[sector]
        s_impact = impact_data.get("impact", 0.0)
        
        sector_change = item.get("sector_change", 0.0)
        if sector_change < 0:
            impact_direction = "negative"
        elif sector_change > 0:
            impact_direction = "positive"
        else:
            impact_direction = "neutral"

        stocks_info = stock_drilldown.get(sector, [])
        stock_symbols = [s["symbol"] for s in stocks_info]

        causal_chains.append({
            "news": item.get("news", "Unknown News"),
            "news_sentiment": item.get("news_sentiment", "neutral"),
            "sector": sector,
            "sector_change": sector_change,
            "portfolio_weight": item.get("portfolio_weight", impact_data.get("portfolio_weight", 0.0)),
            "stocks": stock_symbols,
            "impact_direction": impact_direction,
            "sector_impact": s_impact
        })

    causal_chains.sort(key=lambda x: abs(x["sector_impact"]), reverse=True)

    return causal_chains
