import logging
from typing import Dict, Any

# Primary signal modules
from app.analytics.market_sentiment import compute_market_sentiment
from app.analytics.sector_intelligence import compute_sector_trends
from app.analytics.news_filtering import prepare_news
from app.analytics.news_mapping import map_news_to_entities
from app.analytics.news_impact import assign_directional_impact
from app.analytics.sector_news_aggregation import aggregate_sector_news

from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def build_market_intelligence(loader: DataLoader) -> dict:
    """
    Orchestrates the Phase 1 intelligence pipeline to produce a 
    unified Market Intelligence object.
    """
    try:
        market_sentiment = compute_market_sentiment(loader)
        sector_trends = compute_sector_trends(loader)
        filtered_news_raw = prepare_news(loader)
        mapped_news = map_news_to_entities(loader, filtered_news_raw)
        news_with_impact = assign_directional_impact(mapped_news)
        sector_news_map = aggregate_sector_news(news_with_impact)

        return {
            "market_sentiment": market_sentiment,
            "sector_trends": sector_trends,
            "sector_news_map": sector_news_map,
            "filtered_news": news_with_impact
        }
        
    except Exception as e:
        logger.error(f"Error building market intelligence: {e}")
        return {
            "market_sentiment": {},
            "sector_trends": {},
            "sector_news_map": {},
            "filtered_news": []
        }
