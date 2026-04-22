import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def assign_directional_impact(mapped_news: List[dict]) -> List[dict]:
    """
    Converts qualitative sentiment labels into quantitative directional signals.
    'POSITIVE' -> +1, 'NEGATIVE' -> -1, 'NEUTRAL' -> 0.
    """
    enriched_news = []
    
    # Mapping for deterministic transformation
    sentiment_to_direction = {
        "POSITIVE": 1,
        "NEGATIVE": -1,
        "NEUTRAL": 0,
        "positive": 1,
        "negative": -1,
        "neutral": 0
    }

    for item in mapped_news:
        sentiment = item.get("sentiment")
        
        # Step 5: Edge case handling
        if sentiment is None:
            logger.warning(f"Sentiment missing for news item {item.get('news_id')}. Skipping.")
            continue
            
        # Step 3: Add impact_direction
        # Default to 0 if sentiment label is unrecognized
        direction = sentiment_to_direction.get(sentiment, 0)
        
        # Build enriched object
        enriched_item = item.copy()
        enriched_item["impact_direction"] = direction
        
        enriched_news.append(enriched_item)
        
    return enriched_news
