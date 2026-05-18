import logging
import os
from typing import Dict, List, Any
# Standardized Imports
# Standardized Imports
from app.utils.helpers import safe_float

from app.utils.helpers import safe_float

logger = logging.getLogger(__name__)

from app.data.portfolio_builder import (
    get_portfolio_returns,
    get_portfolio_metadata,
    PORTFOLIOS,
)
from app.data.market_data import fetch_ohlcv
from app.quant.regime_detector import detect_regimes_hmm, get_current_regime
import numpy as np
import pandas as pd
from app.quant.portfolio_signals import (
    scan_portfolio_signals,
    generate_portfolio_signal_summary,
    aggregate_sector_signals,
    generate_signal_diagnostics,
    portfolio_signal_reasoning_context
)
from dataclasses import asdict

def load_portfolio_context(
    portfolio_id: str,
    period: str = "6mo"
) -> dict:
    """
    Load real portfolio context for reasoning.
    
    Returns a deterministic, serializable dictionary of analytics.
    """
    try:
        # Fetch data
        returns_df = get_portfolio_returns(portfolio_id, period=period)
        meta = get_portfolio_metadata(portfolio_id)
        
        if returns_df.empty:
            logger.warning(f"Empty returns for {portfolio_id}, using fallback context.")
            return get_fallback_context(portfolio_id)

        # 1. Latest Portfolio Return
        latest_return = float(returns_df['portfolio_return'].iloc[-1])
        
        # 2. Weekly Compounded Return (5 trading days)
        recent_returns = returns_df['portfolio_return'].tail(5)
        weekly_return = float((1 + recent_returns).prod() - 1)
        
        # 3. Annualized Volatility
        daily_vol = returns_df['portfolio_return'].std()
        annual_vol = float(daily_vol * np.sqrt(252))
        
        # 4. Context Windowing (last 30 days of returns)
        # We only keep the aggregate return to save space
        recent_history_df = returns_df[['portfolio_return']].tail(30)
        # Convert index to strings for JSON serializability
        recent_history_df.index = recent_history_df.index.strftime('%Y-%m-%d')
        recent_history = recent_history_df.to_dict()
        
        # 5. Holdings with daily change for reasoning tools compatibility
        holdings_detail = []
        for ticker, weight in meta["holdings"].items():
            daily_change = 0.0
            if ticker in returns_df.columns:
                daily_change = float(returns_df[ticker].iloc[-1])
            
            holdings_detail.append({
                "ticker": ticker,
                "weight": float(weight),
                "daily_change": daily_change,
                "sector": meta["sector_exposure"].get(ticker, "Unknown")
            })
        
        # 6. Market Regime Intelligence
        # We use Nifty 50 Index (^NSEI) as the regime benchmark
        regime_df = pd.DataFrame()
        market_regime = {
            "current_regime": "Unknown",
            "days_in_regime": 0,
            "regime_distribution": {"Bull": 0.0, "Bear": 0.0, "Sideways": 0.0}
        }
        
        try:
            # Fetch 2y history for regime detection stability
            nifty_raw = fetch_ohlcv("^NSEI", period="2y")
            if not nifty_raw.empty:
                regime_df = detect_regimes_hmm(nifty_raw)
                if not regime_df.empty:
                    current_reg_info = get_current_regime(regime_df)
                    
                    # Calculate Distribution
                    counts = regime_df["regime"].value_counts(normalize=True).to_dict()
                    regime_dist = {
                        "Bull": float(counts.get("Bull", 0.0)),
                        "Bear": float(counts.get("Bear", 0.0)),
                        "Sideways": float(counts.get("Sideways", 0.0))
                    }
                    
                    market_regime = {
                        "current_regime": current_reg_info["current_regime"],
                        "days_in_regime": current_reg_info["days_in_regime"],
                        "regime_distribution": regime_dist
                    }
        except Exception as reg_err:
            logger.error(f"Regime detection failed during context load: {reg_err}")

        # 7. Portfolio Signal Intelligence
        signal_intelligence = {
            "market_bias": "NEUTRAL",
            "top_opportunity": None,
            "sector_clusters": [],
            "signal_diagnostics": [],
            "summary": "Signal intelligence currently unavailable."
        }
        
        try:
            # Prepare ticker data dict for signal scanner
            # We use the returns_df which already contains columns for each ticker
            ticker_data = {}
            for ticker in meta["holdings"].keys():
                if ticker in returns_df.columns:
                    # Signal generator needs OHLCV-like structure, but it mostly needs 'Close'
                    # and indicators. We can create a lightweight proxy or fetch if needed.
                    # To ensure accuracy, we'll fetch indicators if they aren't here.
                    from app.quant.signals import compute_signals
                    raw_ticker = fetch_ohlcv(ticker, period=period)
                    if not raw_ticker.empty:
                        ticker_data[ticker] = compute_signals(raw_ticker)
            
            if ticker_data:
                # Use HMM regimes if we had them per ticker, but for now we use global
                # or empty dict if per-ticker regimes aren't calculated yet.
                signals = scan_portfolio_signals(ticker_data)
                sig_summary = generate_portfolio_signal_summary(signals)
                sector_agg = aggregate_sector_signals(signals)
                sig_diagnostics = generate_signal_diagnostics(signals, sig_summary, sector_agg)
                
                # Identify clusters
                clusters = [d for d in sig_diagnostics if "cluster" in d.lower()]
                
                top_opp = None
                if signals and signals[0].direction != "NEUTRAL":
                    s = signals[0]
                    top_opp = {
                        "ticker": s.ticker,
                        "direction": s.direction,
                        "confidence": float(s.confidence),
                        "signal_strength": s.signal_strength,
                        "sector": meta["sector_exposure"].get(s.ticker, "Unknown"),
                        "causal_reason": s.causal_reason
                    }
                
                signal_intelligence = {
                    "market_bias": sig_summary.market_bias,
                    "top_opportunity": top_opp,
                    "sector_clusters": clusters,
                    "signal_diagnostics": sig_diagnostics,
                    "summary": generate_signal_intelligence_summary(sig_summary, clusters),
                    "raw_summary": asdict(sig_summary)
                }
        except Exception as sig_err:
            logger.error(f"Signal intelligence injection failed: {sig_err}")

        context = {
            "portfolio_id": portfolio_id,
            "name": meta["name"],
            "holdings": meta["holdings"],
            "holdings_detail": holdings_detail,
            "sector_exposure": meta["sector_exposure"],
            "returns_summary": recent_history,
            "latest_return": latest_return,
            "weekly_return": weekly_return,
            "volatility": annual_vol,
            "num_assets": meta["holdings_count"],
            "market_regime": market_regime,
            "signal_intelligence": signal_intelligence,
            "status": "success"
        }
        
        # 7. Add Analytical Summary
        context["market_summary"] = generate_market_summary(context)
        
        return context
        
    except Exception as e:
        logger.error(f"Error loading portfolio context for {portfolio_id}: {e}")
        return get_fallback_context(portfolio_id)

def get_fallback_context(portfolio_id: str) -> dict:
    """Safe fallback context to prevent reasoning engine crashes."""
    return {
        "portfolio_id": portfolio_id,
        "name": "Unknown Portfolio",
        "holdings": {},
        "holdings_detail": [],
        "sector_exposure": {},
        "returns_summary": {},
        "latest_return": 0.0,
        "weekly_return": 0.0,
        "volatility": 0.0,
        "num_assets": 0,
        "market_regime": {
            "current_regime": "Unknown",
            "days_in_regime": 0,
            "regime_distribution": {"Bull": 0.0, "Bear": 0.0, "Sideways": 0.0}
        },
        "signal_intelligence": {
            "market_bias": "NEUTRAL",
            "top_opportunity": None,
            "sector_clusters": [],
            "signal_diagnostics": [],
            "summary": "Signal intelligence currently unavailable."
        },
        "market_summary": "Portfolio analytics are currently unavailable.",
        "status": "fallback"
    }

def generate_market_summary(context: dict) -> str:
    """Produce a concise, strictly analytical summary for the AI layer."""
    regime = context.get("market_regime", {})
    current = regime.get("current_regime", "Unknown")
    days = regime.get("days_in_regime", 0)
    
    p_ret = context.get("latest_return", 0.0)
    vol = context.get("volatility", 0.0)
    
    # 1. Base Market Overview
    summary = f"Market regime is {current} ({days} days). "
    summary += f"Portfolio daily return: {p_ret:+.2%}, Volatility: {vol:.2%}. "
    
    # 2. Signal Context Fusion
    sig_intel = context.get("signal_intelligence", {})
    if sig_intel.get("market_bias") != "NEUTRAL":
        summary += f"Signal structure reflects a {sig_intel['market_bias']} bias. "
        if sig_intel.get("top_opportunity"):
            top = sig_intel["top_opportunity"]
            summary += f"Top opportunity: {top['ticker']} ({top['direction']}). "
    
    return summary.strip()

def generate_signal_intelligence_summary(summary_obj: Any, clusters: List[str]) -> str:
    """Produce concise operational intelligence summary for the signal layer."""
    bias = summary_obj.market_bias
    longs = summary_obj.long_signals
    shorts = summary_obj.short_signals
    
    text = f"Portfolio signals reflect a {bias} bias ({longs} LONG, {shorts} SHORT). "
    if clusters:
        # Use only the first cluster for brevity
        text += f"Key cluster: {clusters[0].split('.')[0]}."
        
    return text.strip()

def build_safe_error_payload(tool_type: str) -> dict:
    return {
        "type": tool_type, "status": "error",
        "summary": "I'm currently missing some required data to provide an accurate breakdown.",
        "drivers": [], "risks": [], "metrics": {}
    }

def get_portfolio_context_data(portfolio_id: str) -> dict:
    # Portfolio Validation (Legacy shim)
    if portfolio_id not in PORTFOLIOS:
        portfolio_id = "PORTFOLIO_001"

    # Use the new high-fidelity context loader
    ctx = load_portfolio_context(portfolio_id)
    
    if ctx.get("status") == "fallback":
        return {"error": "no_data"}
        
    # Map to legacy structure for downstream tool compatibility
    return {
        "exposure": ctx["sector_exposure"],
        "holdings_map": ctx["holdings"], 
        "ranked_holdings": ctx["holdings_detail"],
        "portfolio_id": portfolio_id,
        "analytics": ctx # Pass full context for new tools
    }

def run_reason_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    if "error" in ctx: return build_safe_error_payload("reason")
    p_id = ctx["portfolio_id"]
    analytics = ctx.get("analytics", {})

    try:
        return {
            "type": "reason", "status": "success",
            "summary": analytics.get("market_summary", f"Analytical check complete for {p_id}."),
            "drivers": [], "risks": [],
            "metrics": {
                "sector_exposure": ctx["exposure"],
                "ranked_holdings": ctx["ranked_holdings"]
            }
        }
    except Exception:
        return build_safe_error_payload("reason")

def run_risk_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    ctx = get_portfolio_context_data(portfolio_id)
    if "error" in ctx: return build_safe_error_payload("risk")

    try:
        return {
            "type": "risk", "status": "success",
            "summary": "Risk scan complete.",
            "drivers": [], "risks": [],
            "metrics": {
                "sector_exposure": ctx["exposure"], 
                "ranked_holdings": ctx["ranked_holdings"]
            }
        }
    except Exception:
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
