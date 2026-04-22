import logging
from typing import List, Dict
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def map_news_to_entities(loader: DataLoader, prepared_news: Dict[str, List[dict]]) -> List[dict]:
    """
    Maps prepared news signals to specific sectors and stocks based on scope.
    Acts as the entry point for causal impact reasoning.
    """
    mapped_results = []
    
    # Pre-fetch all available sectors for market-wide expansion
    all_sectors = list(loader.get_sector_mapping().keys())
    stock_to_sector_map = loader.stock_to_sector
    
    # Counter for unique ID generation
    news_counter = 1

    # Define scope categories to iterate
    categories = [
        ("market_news", "market"),
        ("sector_news", "sector"),
        ("stock_news", "stock")
    ]

    for category_key, scope_label in categories:
        items = prepared_news.get(category_key, [])
        
        for item in items:
            entities = item.get("entities", [])
            if not entities and scope_label != "market": # Market news might not have specific entities
                continue

            affected_sectors = []
            affected_stocks = []

            # Step 3: Handle Market News (Expand to all sectors)
            if scope_label == "market":
                affected_sectors = all_sectors
                affected_stocks = []

            # Step 4: Handle Sector News
            elif scope_label == "sector":
                affected_sectors = entities
                affected_stocks = []

            # Step 5: Handle Stock News (Map back to sectors)
            elif scope_label == "stock":
                affected_stocks = [s.strip().upper() for s in entities]
                # Map stocks to sectors, ignoring duplicates and missing mappings
                for stock in affected_stocks:
                    sector = stock_to_sector_map.get(stock)
                    if sector and sector not in affected_sectors:
                        affected_sectors.append(sector)

            # Step 7: Skip if no sectors resolved
            if not affected_sectors:
                logger.warning(f"No sectors resolved for news item: {item.get('headline')[:50]}")
                continue

            # Step 2 & 8: Build and store clean mapping
            mapped_results.append({
                "news_id": f"NEWS_{news_counter}",
                "affected_sectors": affected_sectors,
                "affected_stocks": affected_stocks,
                "scope": scope_label,
                "sentiment": item.get("sentiment"),
                "impact": item.get("impact"),
                "headline": item.get("headline") # Keeping for test visibility
            })
            
            news_counter += 1

    return mapped_results
