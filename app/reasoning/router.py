import logging
import os
from typing import Dict, List, Any
# Standardized Imports
# Standardized Imports
from app.utils.helpers import safe_slice, safe_float

# Analytical Components
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

# Reasoning Components
from app.reasoning.stock_impact_drilldown import get_stock_level_impact
from app.reasoning.news_portfolio_link import link_news_to_portfolio
from app.reasoning.news_sector_enrichment import attach_sector_trends_to_news
from app.reasoning.portfolio_exposure_enrichment import attach_portfolio_exposure
from app.reasoning.causal_chain_builder import build_causal_chains
from app.reasoning.conflict_detector import detect_conflicts

logger = logging.getLogger(__name__)

_loader = DataLoader(os.path.join("data", "mock"))
VALID_PORTFOLIOS = ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"]

def build_safe_error_payload(tool_type: str) -> dict:
    return {
        "type": tool_type, "status": "error",
        "summary": "I'm currently missing some required data to provide an accurate breakdown.",
        "drivers": [], "risks": [], "metrics": {}
    }

def get_portfolio_context_data(portfolio_id: str) -> dict:
    # Portfolio Validation
    if portfolio_id not in VALID_PORTFOLIOS:
        portfolio_id = "PORTFOLIO_001"

    try:
        raw = load_portfolio(_loader, portfolio_id)
        if not raw: return {"error": "no_data"}
        
        norm_map = normalize_holdings(_loader, raw)
        exp = compute_sector_exposure(_loader, norm_map, raw)
        
        ranked_holdings = []
        for ticker, h_data in norm_map.items():
            ranked_holdings.append({
                "ticker": str(ticker), 
                "sector": str(h_data.get("sector", "Unknown")),
                "weight": safe_float(h_data.get("weight", 0.0)),
                "daily_change": safe_float(h_data.get("day_change", 0.0))
            })
        ranked_holdings.sort(key=lambda x: x["weight"], reverse=True)
        
        return {"exposure": exp, "holdings_map": norm_map, "ranked_holdings": ranked_holdings, "portfolio_id": portfolio_id}
    except:
        return {"error": "exception"}

def run_reason_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    if "error" in ctx: return build_safe_error_payload("reason")
    p_id = ctx["portfolio_id"]

    try:
        m_intel = build_market_intelligence(_loader)
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
            "type": "reason", "status": "success",
            "summary": f"Analytical check complete for {p_id}.",
            "drivers": safe_slice(chains, k=5), "risks": safe_slice(conflicts, k=3),
            "metrics": {
                "sector_performance": compute_sector_performance(_loader, m_intel["sector_trends"]),
                "sector_exposure": ctx["exposure"],
                "ranked_holdings": ctx["ranked_holdings"]
            }
        }
    except:
        return build_safe_error_payload("reason")

def run_risk_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    if "error" in ctx: return build_safe_error_payload("risk")

    try:
        risks = detect_concentration_risk(ctx["exposure"])
        return {
            "type": "risk", "status": "success",
            "summary": "Risk scan complete.",
            "drivers": [], "risks": safe_slice(risks, k=3),
            "metrics": {
                "sector_exposure": ctx["exposure"], 
                "ranked_holdings": ctx["ranked_holdings"]
            }
        }
    except:
        return build_safe_error_payload("risk")

def run_full_analysis_wrapper(portfolio_id: str) -> Dict[str, Any]:
    reason = run_reason_engine_wrapper(portfolio_id)
    risk = run_risk_engine_wrapper(portfolio_id)
    if reason["status"] == "error" and risk["status"] == "error":
        return build_safe_error_payload("full_analysis")
    
    metrics = {**reason.get("metrics", {}), **risk.get("metrics", {})}
    return {
        "type": "full_analysis", "status": "success",
        "summary": "Full analysis complete.",
        "drivers": reason.get("drivers", []), "risks": risk.get("risks", []),
        "metrics": metrics
    }

def switch_portfolio_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    if "error" in ctx:
        return {
            "type": "switch_portfolio", "status": "error",
            "summary": "I couldn't find that portfolio. Please provide a valid ID.",
            "drivers": [], "risks": [], "metrics": {}
        }
    return {
        "type": "switch_portfolio", "status": "success",
        "summary": f"Context as {ctx['portfolio_id']}.",
        "drivers": [], "risks": [], 
        "metrics": {"portfolio_id": ctx["portfolio_id"], "sector_exposure": ctx["exposure"]}
    }

EXECUTION_PRIORITY = ["switch_portfolio", "reason", "risk", "full_analysis"]
ROUTER = {
    "full_analysis": run_full_analysis_wrapper, "reason": run_reason_engine_wrapper, "risk": run_risk_engine_wrapper, "switch_portfolio": switch_portfolio_wrapper
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

        data = tool_func(active_portfolio)
        execution_results.append(data)

    return {
        "portfolio_id": active_portfolio,
        "results": execution_results
    }
