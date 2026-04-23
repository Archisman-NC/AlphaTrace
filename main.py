import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables at the absolute entry point
load_dotenv()

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
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from app.evaluation.llm_evaluator import evaluate_explanation, compute_confidence, build_final_output, rule_check, compute_rule_score
from app.utils.helpers import langfuse, timed_phase

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.jsonl")


def write_log(entry: dict):
    """Append a structured JSON log entry to the pipeline log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_pipeline(portfolio_ids: list):
    print("Starting Autonomous Financial Advisor Agent - Intelligence Pipeline...\n")

    # Initialize DataLoader
    loader = DataLoader(os.path.join("data", "mock"))

    # Global Market Intelligence
    with timed_phase("market_intelligence"):
        market_intelligence = build_market_intelligence(loader)
        sentiment = market_intelligence["market_sentiment"]
        trends = market_intelligence["sector_trends"]
        news = market_intelligence["filtered_news"]

    if not news:
        print("[WARN] No news data available — reasoning will rely on quantitative signals only.")

    for pid in portfolio_ids:
        # Portfolio Intelligence Phase
        with timed_phase("portfolio_analytics"):
            raw_portfolio = load_portfolio(loader, pid)
            if not raw_portfolio:
                print(f"[ERROR] Portfolio '{pid}' not found. Skipping.")
                continue

            if "holdings" not in raw_portfolio and "stocks" not in raw_portfolio:
                print(f"[ERROR] Portfolio '{pid}' has no holdings data. Skipping.")
                continue

            normalized_holdings = normalize_holdings(loader, raw_portfolio)
            metrics = compute_portfolio_metrics(raw_portfolio)
            exposure = compute_sector_exposure(loader, normalized_holdings, raw_portfolio)
            ranked = rank_holdings(normalized_holdings, top_n=3)
            risks = detect_concentration_risk(exposure)
            stock_map = build_stock_exposure_map(normalized_holdings, ranked)
            linked_trends = link_portfolio_to_sector_trends(exposure, trends)
            impacts = compute_sector_impact(linked_trends)
            top_impacts = get_top_impact_sectors(impacts, top_n=3)

        # Reasoning & Evaluation Phase
        with timed_phase("reasoning_and_evaluation"):
            stock_drivers = get_stock_level_impact(top_impacts, stock_map)
            mf_reasoning = process_mutual_funds(loader, raw_portfolio, mode="simple")
            relevant_news = link_news_to_portfolio(news, exposure, stock_map)
            enriched_news = attach_sector_trends_to_news(relevant_news, trends)
            personalized_news = attach_portfolio_exposure(enriched_news, exposure)
            causal_chains = build_causal_chains(personalized_news, impacts, stock_drivers)
            scored_chains = compute_impact_scores(causal_chains)
            top_causal_drivers = select_top_drivers(scored_chains, top_n=2)
            conflicts = detect_conflicts(causal_chains, normalized_holdings, trends)
            validation = validate_outputs(exposure, top_impacts, stock_map, risks)

            daily_chg = abs(metrics.get("daily_change_percent", 0.0))
            if daily_chg < 0.1:
                sig_class = "weak"
            elif daily_chg < 1:
                sig_class = "moderate"
            else:
                sig_class = "strong"

            explanation = generate_llm_explanation(metrics, top_causal_drivers, conflicts, risks, portfolio_id=pid)

            original_input = {
                "portfolio_change": metrics.get("daily_change_percent", 0.0),
                "top_drivers": top_causal_drivers,
                "conflicts": conflicts,
                "risks": risks
            }

            eval_score = evaluate_explanation(explanation, original_input, portfolio_id=pid)
            align_str = sum(abs(v['impact']) for v in top_causal_drivers) if top_causal_drivers else 0
            has_mixed = len([d for d in top_causal_drivers if d['impact'] > 0]) > 0 and len([d for d in top_causal_drivers if d['impact'] < 0]) > 0

            confidence = compute_confidence(conflicts, align_str, float(metrics.get('daily_change_percent', 0)), signal_strength=sig_class, has_mixed_signals=has_mixed)
            final_output = build_final_output(explanation, eval_score, confidence, signal_strength=sig_class)

        # Deterministic Guard Rails
        summary_text = final_output.get("summary", "")
        checks = rule_check(summary_text, top_causal_drivers)
        r_score = compute_rule_score(checks)
        llm_score = float(final_output.get("evaluation_score", 0))
        hybrid_score = min(10.0, llm_score + (r_score * 2))
        final_output["evaluation_score"] = round(hybrid_score, 1)

        # Terminal Visualization
        p_type = raw_portfolio.get('portfolio_type', raw_portfolio.get('type', 'N/A'))
        owner = raw_portfolio.get('user_name', 'Unknown')
        print(f"\n{'─'*60}")
        print(f" 📊 {pid} | {owner} | {p_type.title()}")
        print(f"{'─'*60}")
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

        print(f"\n  [SYSTEM CONFIDENCE] {final_output.get('confidence', 0) * 100:.1f}%")
        print(f"  [AI JUDGE SCORE]    {final_output.get('evaluation_score', 0):.1f} / 10")
        print(f"  [SIGNAL STRENGTH]   {final_output.get('signal_strength', 'unknown').upper()}")
        print(f"  [RULE CHECK]        Sector: {'✔' if checks['mentions_sector'] else '✘'} | Stock: {'✔' if checks['mentions_stock'] else '✘'} | Cause: {'✔' if checks['mentions_cause'] else '✘'}")

        # Telemetry Logging
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_id": pid,
            "owner": owner,
            "portfolio_change": metrics.get("daily_change_percent", 0.0),
            "signal_strength": sig_class,
            "top_drivers": [d.get("sector", "") for d in top_causal_drivers],
            "confidence": round(confidence, 2),
            "evaluation_score": final_output["evaluation_score"],
            "rule_check": checks
        }
        write_log(log_entry)

    print(f"\n{'='*60}")
    print(" Analysis Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaTrace — Autonomous Financial Intelligence Pipeline"
    )
    parser.add_argument(
        "--portfolio",
        default="all",
        help="Portfolio ID to analyze (e.g. PORTFOLIO_001). Use 'all' to run all portfolios."
    )
    args = parser.parse_args()

    if args.portfolio.lower() == "all":
        run_pipeline(["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
    else:
        run_pipeline([args.portfolio])

    try:
        langfuse.flush()
    except:
        pass
