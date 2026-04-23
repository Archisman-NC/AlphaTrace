import os
from dotenv import load_dotenv
from app.ingestion.data_loader import DataLoader
from app.analytics.market_intelligence import build_market_intelligence
from app.analytics.portfolio_loader import load_portfolio
from app.analytics.portfolio_normalizer import normalize_holdings
from app.analytics.portfolio_metrics import compute_portfolio_metrics
from app.analytics.sector_exposure import compute_sector_exposure
from app.analytics.holding_ranker import rank_holdings
from app.analytics.risk_detection import detect_concentration_risk
from app.analytics.stock_exposure_map import build_stock_exposure_map
from app.analytics.sector_portfolio_link import link_portfolio_to_sector_trends
from app.analytics.sector_impact import compute_sector_impact
from app.analytics.top_impact_sectors import get_top_impact_sectors
from app.reasoning.stock_impact_drilldown import get_stock_level_impact
from app.reasoning.mutual_fund_handler import process_mutual_funds
from app.evaluation.output_validator import validate_outputs
from app.reasoning.news_portfolio_link import link_news_to_portfolio
from app.reasoning.news_sector_enrichment import attach_sector_trends_to_news
from app.reasoning.portfolio_exposure_enrichment import attach_portfolio_exposure
from app.reasoning.causal_chain_builder import build_causal_chains
from app.reasoning.impact_scorer import compute_impact_scores
from app.reasoning.top_drivers import select_top_drivers
from app.reasoning.conflict_detector import detect_conflicts
from app.reasoning.llm_explainer import generate_llm_explanation
from app.evaluation.llm_evaluator import evaluate_explanation, compute_confidence, build_final_output

# Load environment variables from .env if present
load_dotenv()

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
    
    # --- PHASE 2: PORTFOLIO INTELLIGENCE ---
    print("--- Phase 2: Portfolio Intelligence Dashboard ---")
    portfolios_to_test = ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"]
    
    for pid in portfolios_to_test:
        print(f"\n{'='*60}")
        print(f" PROCESSING: {pid}")
        print(f"{'='*60}")
        
        # 1. Load Portfolio
        raw_portfolio = load_portfolio(loader, pid)
        if not raw_portfolio:
            print(f"Skipping {pid}: Not found.")
            continue
            
        # 2. Normalize Holdings
        normalized_holdings = normalize_holdings(loader, raw_portfolio)
        
        # 3. Compute Metrics
        metrics = compute_portfolio_metrics(raw_portfolio)
        
        # 4. Compute True Exposure
        exposure = compute_sector_exposure(loader, normalized_holdings, raw_portfolio)
        
        # 5. Rank Holdings by Weight (NEW)
        ranked = rank_holdings(normalized_holdings, top_n=3)
        
        # 6. Detect Concentration Risks
        risks = detect_concentration_risk(exposure)
        
        # 7. Build Stock Exposure Map
        stock_map = build_stock_exposure_map(normalized_holdings, ranked)
        
        # 8. Link Portfolio to Sector Trends
        linked_trends = link_portfolio_to_sector_trends(exposure, trends)
        
        # 9. Compute Sector Impact
        impacts = compute_sector_impact(linked_trends)
        
        # 10. Identify Top Impact Sectors
        top_impacts = get_top_impact_sectors(impacts, top_n=3)
        
        # 11. Reasoning: Stock Impact Drilldown (PHASE 3)
        stock_drivers = get_stock_level_impact(top_impacts, stock_map)
        
        # 12. Reasoning: Mutual Fund Interpretation (PHASE 3)
        mf_reasoning = process_mutual_funds(loader, raw_portfolio, mode="simple")
        
        # 13. Reasoning: News-Portfolio Linkage (PHASE 3)
        relevant_news = link_news_to_portfolio(news, exposure, stock_map)
        
        # 14. Reasoning: News-Sector Enrichment (PHASE 3)
        enriched_news = attach_sector_trends_to_news(relevant_news, trends)
        
        # 15. Reasoning: Portfolio Exposure Enrichment (PHASE 3)
        personalized_news = attach_portfolio_exposure(enriched_news, exposure)
        
        # 16. Reasoning: Causal Chain Builder (PHASE 3)
        causal_chains = build_causal_chains(personalized_news, impacts, stock_drivers)
        
        # 17. Reasoning: Impact Scorer (PHASE 3)
        scored_chains = compute_impact_scores(causal_chains)
        
        # 18. Reasoning: Top Drivers (PHASE 3)
        top_causal_drivers = select_top_drivers(scored_chains, top_n=2)
        
        # 19. Reasoning: Conflict Detector (PHASE 3)
        conflicts = detect_conflicts(causal_chains, normalized_holdings, trends)
        
        # 20. Output Validation (Final Step)
        validation = validate_outputs(exposure, top_impacts, stock_map, risks)
        
        # Compute Signal Strength Class (Advisory Extension)
        daily_chg = abs(metrics.get("daily_change_percent", 0.0))
        if daily_chg < 0.1:
            sig_class = "weak"
        elif daily_chg < 1:
            sig_class = "moderate"
        else:
            sig_class = "strong"

        # 21. Narrative Generation (Advisory Extension)
        explanation = generate_llm_explanation(metrics, top_causal_drivers, conflicts, risks)
        
        # 22. Quality Evaluation & Confidence (Final Layer)
        original_input = {
            "portfolio_change": metrics.get("daily_change_percent", 0.0),
            "top_drivers": top_causal_drivers,
            "conflicts": conflicts,
            "risks": risks
        }
        
        eval_score = evaluate_explanation(explanation, original_input)
        
        # compute heuristics for confidence
        align_str = sum(abs(v['impact']) for v in top_causal_drivers) if top_causal_drivers else 0
        
        confidence = compute_confidence(conflicts, align_str, float(metrics.get('daily_change_percent', 0)))
        final_output = build_final_output(explanation, eval_score, confidence, signal_strength=sig_class)
        
        # Display Results
        p_type = raw_portfolio.get('portfolio_type', raw_portfolio.get('type', 'N/A'))
        print(f"\n[PROFILE: {p_type.upper()}]")
        print(f"Owner: {raw_portfolio.get('user_name', 'Unknown')}")
        print(f"Total Value: \u20b9{metrics['total_value']:,.2f}")
        print(f"Daily PnL: \u20b9{metrics['daily_pnl']:,.2f}")
        print(f"Daily Change: {metrics['daily_change_percent']}%")
        
        print("\n[VALIDATION STATUS]")
        v_icon = "✔" if validation["is_valid"] else "❌"
        print(f" {v_icon} {validation['summary']}")
        for w in validation["warnings"]:
            print(f"   ! Warning: {w}")

        print("\n[FINAL ADVISORY EXPLANATION]")
        print(f" {final_output.get('summary', 'No summary generated.')}")
        
        if final_output.get("drivers"):
            print("\n  Top Drivers:")
            for d in final_output.get("drivers", []):
                print(f"  - {d}")
                
        if final_output.get("risks"):
            print("\n  Risks & Anomalies:")
            for r in final_output.get("risks", []):
                 print(f"  \u26A0\uFE0F {r}")
                 
        print(f"\n  [SYSTEM CONFIDENCE] {final_output.get('confidence', 0) * 100}%")
        print(f"  [AI JUDGE SCORE]    {final_output.get('evaluation_score', 0)}")
        print(f"  [SIGNAL STRENGTH]   {final_output.get('signal_strength', 'unknown').upper()}")

        print("\n[TOP QUANTITATIVE DRIVERS]")
        if top_causal_drivers:
            for i, driver in enumerate(top_causal_drivers, 1):
                sign = "+" if driver['impact'] > 0 else ""
                print(f" {i}. {driver['sector']} \u2192 {sign}{driver['impact']:.2f}% \u2192 {driver['reason']}")
        else:
            print(" - No top drivers identified.")

        print("\n[CONFLICTS DETECTED]")
        if conflicts:
            for conflict in conflicts:
                print(f" \u26A0\uFE0F {conflict['stock']} \u2192 {conflict['reason']}")
        else:
            print(" - None. Signals perfectly align.")
            
        print("\n[CAUSAL IMPACT SCORES]")
        if scored_chains:
            for chain in scored_chains[:3]:
                sign = "+" if chain['impact'] > 0 else ""
                print(f" - {chain['sector']:<20} \u2192 {sign}{chain['impact']}% \u2190 {chain['news'][:30]}...")
        else:
            print(" - No causal impact scores computed.")
        
        print("\n[NEWS IMPACT BY EXPOSURE]")
        if personalized_news:
            for item in sorted(personalized_news, key=lambda x: x['portfolio_weight'], reverse=True)[:3]:
                print(f" - {item['news'][:50]}... \u2192 {item['sector']}: {item['portfolio_weight']*100:.1f}% exposure")
        else:
            print(" - No portfolio-linked news events identified.")

        print("\n[EVENT -> REACTION MAPPING]")
        if enriched_news:
            # Show first 3 for brevity
            for item in enriched_news[:3]:
                sign = "+" if item['sector_change'] > 0 else ""
                print(f" - {item['news'][:50]}... \u2192 {item['sector']}: {sign}{item['sector_change']}%")
        else:
            print(" - No causal news events identified.")

        print("\n[RELEVANT NEWS SIGNALS]")
        if relevant_news:
            for item in relevant_news[:3]: # Show top 3 relevant
                sentiment_icon = "🔴" if item['sentiment'] == "negative" else "🟢"
                print(f" {sentiment_icon} {item['headline']}")
                print(f"    Reason: {item['relevance_reason']}")
        else:
            print(" - No direct news impact detected for this portfolio.")

        print("\n[REASONING DRILLDOWN: KEY DRIVERS]")
        for sector, stocks in stock_drivers.items():
            sym_list = [s['symbol'] for s in stocks]
            print(f" - {sector:<20}: Driven by {', '.join(sym_list)}")

        print("\n[MUTUAL FUND INTERPRETATION]")
        if mf_reasoning:
            print(f" Mode: {mf_reasoning['mode'].upper()} | Contribution: {mf_reasoning['mf_contribution'].upper()}")
            for detail in mf_reasoning['mf_details'][:2]: # Show first 2
                print(f" - {detail['fund']}: {detail['note']}")
        else:
            print(" - No mutual funds detected.")

        print("\n[TOP IMPACT SECTORS]")
        for item in top_impacts:
            sign = "+" if item['impact'] > 0 else ""
            print(f" - {item['sector']:<20}: {sign}{item['impact']:.2f}%")

        print("\n[SECTOR IMPACT (CONTRIBUTION TO RETURN)]")
        # Keep the fuller impact list display below as well
        full_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]['impact']), reverse=True)[:3]
        for sector, data in full_impacts:
            sign = "+" if data['impact'] > 0 else ""
            print(f" - {sector:<20}: {sign}{data['impact']:.2f}% impact ({data['portfolio_weight']*100:.1f}% weight)")

        if risks:
            print("\n[RISK WARNINGS]")
            for r in risks:
                print(f" ! {r['description']}")

        print("\n[TOP 3 HOLDINGS (LINKED METADATA)]")
        for item in ranked[:3]:
            sym = item["symbol"]
            data = stock_map[sym]
            print(f" - {sym:<12}: Rank {data['importance_rank']} | {data['sector']} | {data['weight']*100:.2f}%")

    print(f"\n{'='*60}")
    print(" PHASE 2: DATA NORMALIZATION COMPLETE")
    print(f"{'='*60}\n")
    
    print("Intelligence Pipeline Execution Successful!")

if __name__ == "__main__":
    main()
