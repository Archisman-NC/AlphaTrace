import os
from app.ingestion.data_loader import DataLoader
from app.analytics.market_intelligence import build_market_intelligence
from app.analytics.portfolio_loader import load_portfolio
from app.analytics.portfolio_normalizer import normalize_holdings

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
    
    # Phase 2: Portfolio Selection (First Entry)
    print("--- Phase 2: Portfolio Intelligence (Loading) ---")
    for p_id in ["PORTFOLIO_001", "PORTFOLIO_002"]:
        p_loaded = load_portfolio(loader, p_id)
        if p_loaded:
            print(f"[{p_id}] Type: {p_loaded['type'].upper()}")
            print(f"  > Stocks: {len(p_loaded['stocks'])} | MFs: {len(p_loaded['mutual_funds'])}")
            
            # Step 2: Normalize
            normalized = normalize_holdings(loader, p_loaded)
            print(f"  > Normalized Holdings: {len(normalized)}")
            
            # Step 3: Verify Weight Sum
            total_weight = sum(h['weight'] for h in normalized.values())
            print(f"  > Total Stock Weight: {total_weight:.4f}")
            
            # Sample 2
            if normalized:
                samples = list(normalized.items())[:2]
                for sym, data in samples:
                    print(f"    - {sym}: {data['sector']} | Weight: {data['weight']:.4f} | Change: {data['day_change']}%")
    print("-------------------------------------------------\n")
    
    print("Intelligence Pipeline Execution Successful!")

if __name__ == "__main__":
    main()
