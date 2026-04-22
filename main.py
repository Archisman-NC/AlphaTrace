import os
from app.ingestion.data_loader import DataLoader
from app.analytics.market_intelligence import build_market_intelligence

def main():
    print("Starting Autonomous Financial Advisor Agent - Intelligence Pipeline...\n")
    
    # Initialize DataLoader
    loader = DataLoader(os.path.join("data", "mock"))
    
    # Capture Phase 0 / Early Phase 1 Data
    portfolios_payload = loader.portfolios.get("portfolios", {})
    fallback_id = list(portfolios_payload.keys())[0] if portfolios_payload else "TEST_PORTFOLIO"
    one_portfolio = loader.get_portfolio(fallback_id)
    
    print("--- Phase 0: Data Registry Status ---")
    print(f"Indices Cached     : {len(loader.market_data.get('indices', {}))}")
    print(f"Sectors Mapped    : {len(set(loader.stock_to_sector.values()))}")
    print(f"Holdings Resolved : {len(one_portfolio.get('holdings', {}).get('stocks', []))}")
    print("--------------------------------------\n")
    
    # Phase 1: Consolidated Market Intelligence
    market_intelligence = build_market_intelligence(loader)
    
    sentiment = market_intelligence["market_sentiment"]
    trends = market_intelligence["sector_trends"]
    news = market_intelligence["filtered_news"]
    
    print("--- Unified Market Intelligence S.S.O.T ---")
    print(f"Market Sentiment : {sentiment.get('market_sentiment', 'UNKNOWN').upper()} ({sentiment.get('avg_index_change', 0)}%)")
    
    # Identify Top 3 Absolute Moover Sectors
    top_movers = sorted(
        trends.items(), 
        key=lambda x: abs(x[1]['change']), 
        reverse=True
    )[:3]
    
    print("\n[Top 3 Sector Trends]")
    for name, trend in top_movers:
        print(f" - {name}: {trend['change']}% ({trend['sentiment'].upper()})")
    
    print(f"\nTotal Enriched News Signals: {len(news)}")
    print("--------------------------------------------\n")
    
    print("Intelligence Pipeline Execution Successful!")

if __name__ == "__main__":
    main()
