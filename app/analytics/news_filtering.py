import logging
from typing import List, Dict
from app.ingestion.data_loader import DataLoader
from app.utils.helpers import normalize_sector

logger = logging.getLogger(__name__)

def prepare_news(loader: DataLoader) -> Dict[str, List[dict]]:
    """
    Filters, normalizes, and categorizes news for the reasoning engine.
    Only retains HIGH and MEDIUM impact news.
    """
    raw_news = loader.get_news()
    
    # Categorization structure
    prepared = {
        "market_news": [],
        "sector_news": [],
        "stock_news": []
    }
    
    # Mapping for raw scope values in news_data.json to internal categories
    scope_map = {
        "MARKET_WIDE": "market",
        "SECTOR_SPECIFIC": "sector",
        "STOCK_SPECIFIC": "stock",
        "market": "market",
        "sector": "sector",
        "stock": "stock"
    }

    for item in raw_news:
        # Step 2: Filter HIGH and MEDIUM impact
        impact = item.get("impact_level", "LOW").upper()
        if impact not in ["HIGH", "MEDIUM"]:
            continue
            
        # Step 6: Validate mandatory fields
        # Note: news_data.json uses 'entities', but handling 'related_entities' for schema consistency
        entities_data = item.get("entities") or item.get("related_entities")
        scope_raw = item.get("scope")
        
        if not entities_data or not scope_raw:
            logger.warning(f"News item {item.get('id', 'unknown')} missing entities or scope. Skipping.")
            continue
            
        # Determine category
        category_key = scope_map.get(scope_raw)
        if not category_key:
            logger.warning(f"News item {item.get('id')} has invalid scope: {scope_raw}. Skipping.")
            continue
            
        # Step 3: Normalize Entities
        # Handles both flat list (related_entities) and structured dict (entities)
        clean_entities = []
        if isinstance(entities_data, dict):
            sectors = entities_data.get("sectors", [])
            stocks = entities_data.get("stocks", [])
            
            clean_entities.extend([normalize_sector(s) for s in sectors])
            clean_entities.extend([s.strip().upper() for s in stocks])
        elif isinstance(entities_data, list):
            # Fallback if news was already partially processed into a flat list
            clean_entities = [normalize_sector(e) if "_" in e or len(e) > 5 else e.strip().upper() for e in entities_data]

        # Step 5: Build clean news object
        clean_item = {
            "headline": item.get("headline"),
            "sentiment": item.get("sentiment", "NEUTRAL").upper(),
            "entities": clean_entities,
            "impact": impact
        }
        
        # Step 4: Categorize
        prepared[f"{category_key}_news"].append(clean_item)
        
    return prepared
