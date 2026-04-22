import logging
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def compute_news_intelligence(loader: DataLoader) -> dict:
    """
    Analyzes news feed to extract deterministic sentiment signals.
    Aggregates global and sector-specific sentiment metrics.
    """
    news_items: List[dict] = loader.get_news()
    
    if not news_items:
        return {
            "global_news_sentiment": "neutral",
            "global_sentiment_score": 0.0,
            "sector_sentiments": {},
            "top_impact_news": []
        }

    total_score = 0.0
    sector_scores: DefaultDict[str, List[float]] = defaultdict(list)
    high_impact_stories = []

    for item in news_items:
        score = float(item.get("sentiment_score", 0.0))
        total_score += score
        
        # Track sector-specific sentiment
        entities = item.get("entities", {})
        sectors = entities.get("sectors", [])
        for sector in sectors:
            sector_scores[sector].append(score)
            
        # Identify High Impact News
        if item.get("impact_level") == "HIGH":
            high_impact_stories.append({
                "headline": item.get("headline"),
                "sentiment": item.get("sentiment"),
                "score": score
            })

    # Calculate Global Sentiment
    avg_global_score = round(total_score / len(news_items), 2)
    global_sentiment = "neutral"
    if avg_global_score > 0.2:
        global_sentiment = "bullish"
    elif avg_global_score < -0.2:
        global_sentiment = "bearish"

    # Calculate Sector Sentiments
    final_sector_sentiments = {}
    for sector, scores in sector_scores.items():
        avg_sector_score = round(sum(scores) / len(scores), 2)
        sentiment_label = "neutral"
        if avg_sector_score > 0.2:
            sentiment_label = "bullish"
        elif avg_sector_score < -0.2:
            sentiment_label = "bearish"
            
        final_sector_sentiments[sector] = {
            "score": avg_sector_score,
            "sentiment": sentiment_label,
            "article_count": len(scores)
        }

    return {
        "global_news_sentiment": global_sentiment,
        "global_sentiment_score": avg_global_score,
        "sector_sentiments": final_sector_sentiments,
        "top_impact_news": high_impact_stories[:5] # Limit to top 5
    }
