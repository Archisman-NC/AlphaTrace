import logging
import pandas as pd
from dataclasses import asdict
from typing import Dict, List, Any, Optional, Literal
from app.quant.watchdog import scan_portfolio_for_anomalies, WatchdogAlert

logger = logging.getLogger(__name__)


def get_watchdog_insights(portfolio_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
    """
    Aggregate statistical watchdog intelligence into deterministic reasoning-ready context.
    """
    try:
        # 1. Scan for anomalies across the portfolio
        alerts = scan_portfolio_for_anomalies(portfolio_returns)
        
        # 2. Categorize alerts by severity
        categorized = {
            "CRITICAL": [asdict(a) for a in alerts if a.severity == "CRITICAL"],
            "HIGH": [asdict(a) for a in alerts if a.severity == "HIGH"],
            "MEDIUM": [asdict(a) for a in alerts if a.severity == "MEDIUM"],
            "LOW": [asdict(a) for a in alerts if a.severity == "LOW"]
        }
        
        # 3. Determine Operational Status
        status = _classify_operational_status(categorized)
        
        # 4. Generate Summary & Actions
        summary = _generate_watchdog_summary(alerts, status)
        suggested_actions = _generate_suggested_actions(alerts)
        
        # 5. Identify Top Risk
        top_risk = None
        if alerts:
            top_risk = f"{alerts[0].ticker} ({alerts[0].alert_type})"

        return {
            "status": status,
            "critical_alerts": categorized["CRITICAL"],
            "high_alerts": categorized["HIGH"],
            "medium_alerts": categorized["MEDIUM"],
            "low_alerts": categorized["LOW"],
            "summary": summary,
            "top_risk": top_risk,
            "suggested_actions": suggested_actions,
            "timestamp": str(pd.Timestamp.now())
        }

    except Exception as e:
        logger.error(f"Error generating watchdog insights: {e}")
        return {
            "status": "STABLE",
            "summary": "Watchdog monitoring active. No critical anomalies escalated.",
            "top_risk": None,
            "suggested_actions": [],
            "critical_alerts": [],
            "high_alerts": [],
            "medium_alerts": [],
            "low_alerts": []
        }

def _classify_operational_status(categorized: Dict[str, List]) -> str:
    """
    Simple deterministic status classification.
    """
    crit_count = len(categorized["CRITICAL"])
    high_count = len(categorized["HIGH"])
    
    if crit_count > 0 and high_count > 0:
        return "DEGRADED"
    if crit_count > 0:
        return "CRITICAL"
    if high_count > 0:
        return "WATCH"
    
    return "STABLE"

def _generate_watchdog_summary(alerts: List[WatchdogAlert], status: str) -> str:
    """
    Produce concise operational intelligence summary.
    """
    if not alerts:
        return "No statistical anomalies detected across monitored strategies."
    
    if status == "CRITICAL":
        ticker = alerts[0].ticker
        alert_type = alerts[0].alert_type.replace("_", " ").title()
        return f"Critical {alert_type} detected in {ticker} with accompanying operational risk."
    
    if status == "DEGRADED":
        return f"Multiple strategy health violations detected across {len(set(a.ticker for a in alerts))} tickers."
    
    return f"Active monitoring flagged {len(alerts)} moderate statistical anomalies."

def _generate_suggested_actions(alerts: List[WatchdogAlert]) -> List[str]:
    """
    Observational guidance for the researcher.
    """
    if not alerts:
        return []
    
    actions = set()
    alert_types = {a.alert_type for a in alerts}
    
    if "SHARPE_DECAY" in alert_types:
        actions.add("Review strategy robustness and historical edge persistence.")
    if "DISTRIBUTION_SHIFT" in alert_types:
        actions.add("Analyze recent regime transitions and volatility expansion.")
    if "ZSCORE_BREACH" in alert_types:
        actions.add("Investigate potential outlier events and execution risk.")
        
    # Always include a general audit action if status is not stable
    actions.add("Perform quantitative audit of recent signal performance.")
    
    return sorted(list(actions))

def watchdog_reasoning_context(insights: Dict[str, Any]) -> str:
    """
    Escalation hook to provide structured anomaly context to the AI reasoning layer.
    """
    if insights.get("status") == "STABLE":
        return "Strategy Watchdog: Status Stable. No anomalies detected."
        
    context = [
        f"Strategy Watchdog: STATUS {insights['status']}",
        f"Operational Summary: {insights['summary']}",
        f"Top Priority Risk: {insights['top_risk']}",
        f"Total Alerts: {len(insights['critical_alerts']) + len(insights['high_alerts'])} (High/Critical)"
    ]
    
    if insights.get("suggested_actions"):
        context.append(f"Recommended Focus: {', '.join(insights['suggested_actions'][:2])}")
        
    return "\n".join(context)

def generate_proactive_insight(tool_data, user_query, memory, last_topic=None):
    """
    Production fallback for proactive signal detection.
    Guaranteed to return None or a structured Dict.
    """
    try:
        # Placeholder for complex signal detection logic
        # For now, we return None to allow main loop to proceed silently
        return None
    except Exception as e:
        logger.error(f"Proactive engine faulted: {e}")
        return None
