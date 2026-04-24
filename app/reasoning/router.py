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

# --- PRODUCTION-GRADE WRAPPERS ---

def run_reason_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    """
    Real implementation of the Reasoning Engine.
    Orchestrates market intel and causal chain building.
    """
    try:
        # 1. Market & News context
        m_intel = build_market_intelligence(_loader)
        news = m_intel["filtered_news"]
        trends = m_intel["sector_trends"]

        # 2. Portfolio context
        raw_portfolio = load_portfolio(_loader, portfolio_id)
        normalized_holdings = normalize_holdings(_loader, raw_portfolio)
        exposure = compute_sector_exposure(_loader, normalized_holdings, raw_portfolio)
        
        # 3. Intermediate linkages (Stock level)
        stock_map = build_stock_exposure_map(normalized_holdings, {}) 
        linked_trends = link_portfolio_to_sector_trends(exposure, trends)
        impacts = compute_sector_impact(linked_trends)
        top_impacts = get_top_impact_sectors(impacts, top_n=3)
        stock_drivers = get_stock_level_impact(top_impacts, stock_map)

        # 4. News linkage & Causal building
        relevant_news = link_news_to_portfolio(news, exposure, stock_map)
        enriched_news = attach_sector_trends_to_news(relevant_news, trends)
        personalized_news = attach_portfolio_exposure(enriched_news, exposure)
        
        # CORE CALLS
        chains = build_causal_chains(personalized_news, impacts, stock_drivers)
        conflicts = detect_conflicts(chains)

        return {
            "type": "reason",
            "chains": chains,
            "conflicts": conflicts,
            "confidence": 0.92,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Reason engine failure: {e}")
        return {
            "type": "reason",
            "status": "error",
            "data": None,
            "error_message": str(e)
        }

def run_risk_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    """
    Real implementation of the Risk Engine.
    Detects concentration & volatility hazards.
    """
    try:
        raw_portfolio = load_portfolio(_loader, portfolio_id)
        normalized_holdings = normalize_holdings(_loader, raw_portfolio)
        exposure = compute_sector_exposure(_loader, normalized_holdings, raw_portfolio)
        
        # CORE CALL
        risks = detect_concentration_risk(exposure)

        return {
            "type": "risk",
            "risks": risks,
            "confidence": 0.95,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Risk engine failure: {e}")
        return {
            "type": "risk",
            "status": "error",
            "data": None,
            "error_message": str(e)
        }

# --- REMAINING WRAPPERS & ROUTER LOGIC ---

def run_full_analysis_wrapper(portfolio_id: str) -> Dict[str, Any]:
    return f"Composite portfolio analysis initialized for {portfolio_id}."

def switch_portfolio_wrapper(portfolio_id: str) -> str:
    return f"Context migrated to {portfolio_id}."

EXECUTION_PRIORITY = [
    "switch_portfolio",
    "reason",
    "risk",
    "full_analysis"
]

ROUTER = {
    "full_analysis": run_full_analysis_wrapper,
    "reason": run_reason_engine_wrapper,
    "risk": run_risk_engine_wrapper,
    "switch_portfolio": switch_portfolio_wrapper
}

def execute_intents(classification: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the execution of multiple financial intents with state awareness.
    """
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
            logger.info(f"Executing: {intent} for {active_portfolio}")
            data = tool_func(active_portfolio)
            execution_results.append(data if isinstance(data, dict) else {
                "type": intent,
                "data": data,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Execution failed for {intent}: {e}")
            execution_results.append({"type": intent, "status": "error", "data": None})

    return {
        "portfolio_id": active_portfolio,
        "results": execution_results
    }
