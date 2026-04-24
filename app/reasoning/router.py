import logging
import traceback
import os
import json
from typing import Dict, List, Any, Optional

# Core Ingestion & Analytics
from app.ingestion.data_loader import DataLoader
from app.analytics.market_intelligence import build_market_intelligence
from app.analytics.portfolio_loader import load_portfolio
from app.analytics.portfolio_normalizer import normalize_holdings
from app.analytics.sector_exposure import compute_sector_exposure
from app.analytics.stock_exposure_map import build_stock_exposure_map
from app.analytics.sector_portfolio_link import link_portfolio_to_sector_trends
from app.analytics.sector_performance import compute_sector_performance
from app.analytics.sector_impact import compute_sector_impact
from app.analytics.top_impact_sectors import get_top_impact_sectors
from app.analytics.risk_detection import detect_concentration_risk

# Reasoning Logic
from app.reasoning.stock_impact_drilldown import get_stock_level_impact
from app.reasoning.news_portfolio_link import link_news_to_portfolio
from app.reasoning.news_sector_enrichment import attach_sector_trends_to_news
from app.reasoning.portfolio_exposure_enrichment import attach_portfolio_exposure
from app.reasoning.causal_chain_builder import build_causal_chains
from app.reasoning.conflict_detector import detect_conflicts

logger = logging.getLogger(__name__)

# Global Singleton Loader for Mock Environment
_loader = DataLoader(os.path.join("data", "mock"))

def build_error_payload(tool_type: str, error: Exception) -> dict:
    return {
        "type": tool_type,
        "status": "error",
        "summary": f"Analytical failure in {tool_type}.",
        "drivers": [], "risks": [], "metrics": {"error": str(error)}
    }

# --- GLOBAL CONTEXT HELPER ---
def get_portfolio_context_data(portfolio_id: str) -> dict:
    """Computes the base identity of the portfolio for visual consistency."""
    try:
        raw = load_portfolio(_loader, portfolio_id)
        norm_map = normalize_holdings(_loader, raw)
        exp = compute_sector_exposure(_loader, norm_map, raw)
        
        # CONVERT DICT TO RANKED LIST FOR SLICING
        ranked_holdings = []
        for ticker, h_data in norm_map.items():
            ranked_holdings.append({
                "ticker": ticker,
                "sector": h_data.get("sector"),
                "weight": h_data.get("weight"),
                "daily_change": h_data.get("day_change", 0.0)
            })
        ranked_holdings.sort(key=lambda x: x["weight"], reverse=True)
        
        return {
            "exposure": exp, 
            "holdings_map": norm_map,
            "ranked_holdings": ranked_holdings
        }
    except Exception as e:
        logger.error(f"Context data failure: {e}")
        return {"exposure": {}, "holdings_map": {}, "ranked_holdings": []}

# --- ENRICHED PRODUCTION WRAPPERS ---

def run_reason_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    try:
        m_intel = build_market_intelligence(_loader)
        ctx = get_portfolio_context_data(portfolio_id)
        
        stock_map = build_stock_exposure_map(ctx["holdings_map"], {}) 
        linked_trends = link_portfolio_to_sector_trends(ctx["exposure"], m_intel["sector_trends"])
        impacts = compute_sector_impact(linked_trends)
        top_impacts = get_top_impact_sectors(impacts, top_n=3)
        stock_drivers = get_stock_level_impact(top_impacts, stock_map)

        relevant_news = link_news_to_portfolio(m_intel["filtered_news"], ctx["exposure"], stock_map)
        enriched_news = attach_sector_trends_to_news(relevant_news, m_intel["sector_trends"])
        personalized_news = attach_portfolio_exposure(enriched_news, ctx["exposure"])
        
        chains = build_causal_chains(personalized_news, impacts, stock_drivers)
        conflicts = detect_conflicts(chains)

        return {
            "type": "reason",
            "status": "success",
            "summary": f"Causal analysis complete for {portfolio_id}.",
            "drivers": chains, "risks": conflicts,
            "metrics": {
                "chain_count": len(chains),
                "sector_performance": compute_sector_performance(_loader, m_intel["sector_trends"]),
                "sector_exposure": ctx["exposure"],
                "ranked_holdings": ctx["ranked_holdings"] # LIST
            }
        }
    except Exception as e:
        return build_error_payload("reason", e)

def run_risk_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    try:
        ctx = get_portfolio_context_data(portfolio_id)
        risks = detect_concentration_risk(ctx["exposure"])

        return {
            "type": "risk",
            "status": "success",
            "summary": "Risk audit complete.",
            "drivers": [], "risks": risks,
            "metrics": {
                "risk_count": len(risks),
                "sector_exposure": ctx["exposure"], 
                "ranked_holdings": ctx["ranked_holdings"] # LIST
            }
        }
    except Exception as e:
        return build_error_payload("risk", e)

def run_full_analysis_wrapper(portfolio_id: str) -> Dict[str, Any]:
    try:
        reason_data = run_reason_engine_wrapper(portfolio_id)
        risk_data = run_risk_engine_wrapper(portfolio_id)
        metrics = {**reason_data.get("metrics", {}), **risk_data.get("metrics", {})}
        return {
            "type": "full_analysis", "status": "success",
            "summary": "Full analysis complete.",
            "drivers": reason_data.get("drivers", []),
            "risks": risk_data.get("risks", []),
            "metrics": metrics
        }
    except Exception as e:
        return build_error_payload("full_analysis", e)

def switch_portfolio_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    return {
        "type": "switch_portfolio", "status": "success",
        "summary": f"Switched to {portfolio_id}.",
        "drivers": [], "risks": [], "metrics": {
            "portfolio_id": portfolio_id,
            "sector_exposure": ctx["exposure"],
            "ranked_holdings": ctx["ranked_holdings"]
        }
    }

EXECUTION_PRIORITY = ["switch_portfolio", "reason", "risk", "full_analysis"]
ROUTER = {
    "full_analysis": run_full_analysis_wrapper,
    "reason": run_reason_engine_wrapper,
    "risk": run_risk_engine_wrapper,
    "switch_portfolio": switch_portfolio_wrapper
}

def execute_intents(classification: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    target_portfolio_id = classification.get("portfolio_id", session.get("current_portfolio"))
    intents_to_run = classification.get("intent", [])
    ordered_intents = [i for i in EXECUTION_PRIORITY if i in intents_to_run]
    
    execution_results = []
    active_portfolio = session.get("current_portfolio")

    for intent in ordered_intents:
        tool_func = ROUTER.get(intent)
        if not tool_func: continue
        if intent == "switch_portfolio":
            active_portfolio = target_portfolio_id
            session["current_portfolio"] = active_portfolio

        try:
            data = tool_func(active_portfolio)
            execution_results.append(data)
        except Exception as e:
            execution_results.append(build_error_payload(intent, e))

    return {
        "portfolio_id": active_portfolio,
        "results": execution_results
    }
