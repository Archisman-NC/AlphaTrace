import logging
from typing import List, Dict
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

import re

def map_news_to_sectors(news: dict) -> List[str]:
    headline = news.get("headline", "").lower()
    if re.search(r'\b(rbi|interest rate|repo rate)\b', headline):
        return ["BANKING", "FINANCIAL_SERVICES"]
    if re.search(r'\b(fii|fiis|foreign investors)\b', headline):
        return ["MARKET"]
    if re.search(r'\b(oil|crude|opec)\b', headline):
        return ["ENERGY"]
    if re.search(r'\b(earnings|tech|it)\b', headline):
        return ["INFORMATION_TECHNOLOGY"]
    return []

def map_news_to_entities(loader: DataLoader, prepared_news: Dict[str, List[dict]]) -> List[dict]:
    """
    Maps prepared news signals to specific sectors and stocks based on strictly deterministic rules.
    """
    mapped_results = []
    all_sectors = list(loader.get_sector_mapping().keys())
    news_counter = 1

    categories = [
        ("market_news", "market"),
        ("sector_news", "sector"),
        ("stock_news", "stock")
    ]

    for category_key, scope_label in categories:
        items = prepared_news.get(category_key, [])
        for item in items:
            affected_sectors = map_news_to_sectors(item)
            
            if affected_sectors == ["MARKET"]:
                affected_sectors = all_sectors
                
            if not affected_sectors:
                logger.debug(f"No deterministic sectors resolved for: {item.get('headline')[:50]}")
                continue
                
            affected_stocks = []
            if scope_label == "stock":
                affected_stocks = [s.strip().upper() for s in item.get("entities", [])]

            mapped_results.append({
                "news_id": f"NEWS_{news_counter}",
                "affected_sectors": affected_sectors,
                "affected_stocks": affected_stocks,
                "scope": scope_label,
                "sentiment": item.get("sentiment"),
                "impact": item.get("impact"),
                "headline": item.get("headline")
            })
            news_counter += 1

    return mapped_results
