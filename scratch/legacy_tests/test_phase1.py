import os
import sys
import logging

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.data_loader import DataLoader
from app.analytics.market_sentiment import compute_market_sentiment
from app.analytics.sector_intelligence import compute_sector_trends
from app.analytics.news_filtering import prepare_news
from app.analytics.news_mapping import map_news_to_entities
from app.analytics.news_impact import assign_directional_impact
from app.analytics.sector_news_aggregation import aggregate_sector_news
from app.analytics.market_intelligence import build_market_intelligence

# Configure logging for test output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_validation():
    print("====================================================")
    print("PHASE 1: END-TO-END SYSTEM VALIDATION")
    print("====================================================")

    try:
        # STEP 3: EXECUTE FULL PIPELINE
        loader = DataLoader(os.path.join("data", "mock"))
        
        # 1. Market Sentiment
        sentiment = compute_market_sentiment(loader)
        
        # 2. Sector Trends
        trends = compute_sector_trends(loader)
        
        # 3. News Filtering
        filtered_news_raw = prepare_news(loader)
        
        # 4. News Mapping
        mapped_news = map_news_to_entities(loader, filtered_news_raw)
        
        # 5. News Impact
        news_with_impact = assign_directional_impact(mapped_news)
        
        # 6. Sector Aggregation
        sector_agg = aggregate_sector_news(news_with_impact)
        
        # 7. Final Intelligence Object
        final_intel = build_market_intelligence(loader)

        # STEP 4: VALIDATION CHECKS (CRITICAL)
        errors = []

        # A. Market Sentiment
        if not isinstance(sentiment.get("avg_index_change"), float):
            errors.append("Market Sentiment: 'avg_index_change' is not a float.")
        if sentiment.get("market_sentiment") not in ["bullish", "bearish", "neutral"]:
            errors.append(f"Market Sentiment: invalid sentiment value '{sentiment.get('market_sentiment')}'.")

        # B. Sector Trends
        valid_sentiments = ["bullish", "bearish", "neutral"]
        for sector, data in trends.items():
            if not sector:
                errors.append("Sector Trends: detected empty sector name.")
            if not isinstance(data.get("change"), float):
                errors.append(f"Sector Trends: '{sector}' change is not a float.")
            if data.get("sentiment") not in valid_sentiments:
                errors.append(f"Sector Trends: '{sector}' has invalid sentiment '{data.get('sentiment')}'.")

        # C. News Filtering
        # Note: Filtering stage should have excluded LOW impact.
        # But prepare_news returns a dict of lists, we consolidated it in news_with_impact.
        for category, items in filtered_news_raw.items():
            for item in items:
                if item.get("impact") == "LOW":
                    errors.append(f"News Filtering: Detected LOW impact news in '{category}'.")
                if not item.get("headline") or not item.get("sentiment") or "entities" not in item:
                    errors.append(f"News Filtering: Missing required fields in news item: {item.get('headline')[:20]}")

        # D. News Mapping
        for item in mapped_news:
            if not item.get("affected_sectors") and not item.get("affected_stocks"):
                errors.append(f"News Mapping: item {item.get('news_id')} has no affected entities.")
            
            # Market expansion check
            if item.get("scope") == "market":
                if len(item.get("affected_sectors", [])) <= 1:
                    errors.append(f"News Mapping: Market news {item.get('news_id')} failed to expand to multiple sectors.")

        # E. Directional Impact
        for item in news_with_impact:
            if item.get("impact_direction") not in [-1, 0, 1]:
                errors.append(f"News Impact: item {item.get('news_id')} has invalid direction {item.get('impact_direction')}.")

        # F. Sector Aggregation
        for sector, data in sector_agg.items():
            calc_net = sum(n.get("impact_direction", 0) for n in data.get("news", []))
            if data.get("net_sentiment") != calc_net:
                errors.append(f"Sector Aggregation: '{sector}' net_sentiment mismatch. Expected {calc_net}, got {data.get('net_sentiment')}.")
            if data.get("news_count") != len(data.get("news", [])):
                errors.append(f"Sector Aggregation: '{sector}' news_count mismatch.")

        # G. Final Intelligence Object
        required_keys = ["market_sentiment", "sector_trends", "sector_news_map", "filtered_news"]
        for key in required_keys:
            if key not in final_intel:
                errors.append(f"Final Intelligence: Missing key '{key}'.")

        # STEP 5: PRINT DEBUG SUMMARY
        print("\n--- Validation Summary ---")
        sentiment_label = sentiment['market_sentiment'].upper()
        avg_change = sentiment['avg_index_change']
        print(f"Market Sentiment : {sentiment_label} ({avg_change}%)")
        
        top_sectors = sorted(trends.items(), key=lambda x: abs(x[1]['change']), reverse=True)[:3]
        top_movers_list = [f"{s[0]} ({s[1]['change']}%)" for s in top_sectors]
        print(f"Top 3 Movers    : {', '.join(top_movers_list)}")
        
        worst_news_sectors = sorted(sector_agg.items(), key=lambda x: x[1]['net_sentiment'])[:3]
        worst_news_list = [f"{s[0]} ({s[1]['net_sentiment']})" for s in worst_news_sectors]
        print(f"Worst news flow : {', '.join(worst_news_list)}")
        
        sample_headline = news_with_impact[0].get('headline', '')[:50]
        sample_id = news_with_impact[0].get('news_id', '')
        print(f"News Sample     : {sample_id} -> {sample_headline}...")
        
        # STEP 6: EDGE CASE TESTING
        print("\n--- Edge Case Testing ---")
        
        # 1. Empty news
        loader.news_data["news"] = []
        empty_news_prep = prepare_news(loader)
        if any(empty_news_prep.values()):
            errors.append("Edge Case: prepare_news failed to handle empty input.")
        else:
            print("[PASS] Empty News Handled")
            
        # 2. Unknown Stock Mapping
        # Temporarily inject unknown stock into news
        fake_news = [{"scope": "STOCK_SPECIFIC", "entities": {"stocks": ["UNKNOWN_TKR"], "sectors": []}, "impact_level": "HIGH", "sentiment": "NEGATIVE", "headline": "Bad things"}]
        loader.news_data["news"] = fake_news
        prep_fake = prepare_news(loader)
        mapped_fake = map_news_to_entities(loader, prep_fake)
        if any(m.get("affected_sectors") for m in mapped_fake):
             # Since it should fail to map to a sector but keep the item if it has affected_stocks?
             # Actually Step 7 of mapping says: "If no sectors resolved -> skip item"
             if len(mapped_fake) > 0:
                 errors.append("Edge Case: map_news_to_entities should skip items with unresolvable sectors.")
        else:
            print("[PASS] Unknown Stock Mapping Handled")

        # Final Result
        if not errors:
            print("\n" + "="*25)
            print("PHASE 1 VALIDATION PASSED")
            print("="*25)
        else:
            print("\n" + "!"*25)
            print("PHASE 1 VALIDATION FAILED")
            print("!"*25)
            for err in errors:
                print(f" - {err}")

    except Exception as e:
        print(f"\nCRITICAL FAILURE during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_validation()
