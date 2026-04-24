import logging
import random
from typing import List, Dict, Optional
from app.utils.helpers import safe_slice

logger = logging.getLogger(__name__)

PREFIXES = [
    "Also worth noting —",
    "One thing to watch —",
    "Quick heads-up —",
    "Another pulse signal —",
    "From your recent data —"
]

IMPACT_PHRASES = [
    "which could signal a structural shift",
    "which may increase downside risk",
    "which may require closer monitoring",
    "which could affect portfolio stability",
    "which may impact your sector alignment"
]

def detect_proactive_signals(tool_data: dict, user_query: str) -> List[dict]:
    """
    Scans data for triggers using Safe-Slicing.
    """
    triggers = []
    
    full_tool = tool_data.get("full_analysis", {})
    reason_tool = tool_data.get("reason", {})
    risk_tool = tool_data.get("risk", {})
    
    metrics = {
        **full_tool.get("metrics", {}), 
        **reason_tool.get("metrics", {}), 
        **risk_tool.get("metrics", {})
    }
    
    all_risks = full_tool.get("risks", []) or reason_tool.get("risks", []) or risk_tool.get("risks", [])

    # 1. Concentration Risk (>40%)
    exposure = metrics.get("sector_exposure", {})
    if isinstance(exposure, dict):
        for sector, alloc in exposure.items():
            if isinstance(alloc, (int, float)) and alloc > 40:
                severity = "high" if alloc > 50 else "medium"
                triggers.append({
                    "type": "concentration", "weight": 3, "severity": severity,
                    "topic": f"concentration_{sector}",
                    "data": {"sector": sector, "alloc": alloc},
                    "msg": f"Your portfolio is heavily concentrated in {sector} ({alloc:.1f}%)."
                })

    # 2. Holding Divergence (SAFE SLICE HOLDINGS)
    holdings = metrics.get("ranked_holdings", [])
    performance = metrics.get("sector_performance", {})
    
    print(f"[DEBUG] Slicing holdings type: {type(holdings)}")
    for h in safe_slice(holdings, k=3):
        stock_change = h.get("daily_change", 0.0)
        sector_change = performance.get(h.get("sector"), 0.0)
        delta = abs(stock_change - sector_change)
        if delta > 2.0:
            severity = "high" if delta > 3.0 else "medium"
            triggers.append({
                "type": "divergence", "weight": 2, "severity": severity,
                "topic": f"divergence_{h.get('ticker')}",
                "data": {"ticker": h.get("ticker"), "sector": h.get("sector")},
                "msg": f"{h.get('ticker')} is decoupling from the {h.get('sector')} sector trends."
            })

    # 3. Conflicts
    if all_risks and "conflict" not in user_query.lower():
        if any("conflict" in str(r).lower() or "mismatch" in str(r).lower() for r in safe_slice(all_risks, k=10)):
            triggers.append({
                "type": "conflict", "weight": 2, "severity": "medium",
                "topic": "price_news_mismatch",
                "msg": "I've detected a mismatch between positive sentiment and price action in your holdings."
            })

    return triggers

def generate_proactive_insight(tool_data: dict, user_query: str, session_memory: list = None, last_topic: str = None) -> Optional[dict]:
    """
    Refined Proactive Engine using Safe-Slicing.
    """
    triggers = detect_proactive_signals(tool_data, user_query)
    if not triggers: return None

    # Filter & Prioritize
    active_triggers = [t for t in triggers if t["topic"] != last_topic]
    if not active_triggers: return None

    for t in active_triggers:
        t["score"] = t["weight"] + (1 if t["severity"] == "high" else 0)
    
    active_triggers.sort(key=lambda x: x["score"], reverse=True)
    selected = active_triggers[0]

    # Narrative Synthesis
    icon = "⚠️" if selected["severity"] == "high" else "ℹ️"
    prefix = random.choice(PREFIXES)
    impact = random.choice(IMPACT_PHRASES)
    
    bridge = ""
    if session_memory:
        # Use safe extraction of last turn
        last_turn = safe_slice(session_memory, k=1, reverse=True)
        if last_turn:
            turn_str = str(last_turn[0])
            if selected["type"] == "concentration" and selected["data"]["sector"] in turn_str:
                bridge = f" This reinforces the {selected['data']['sector']} focus we discussed."
            elif selected["type"] == "divergence" and selected["data"]["ticker"] in turn_str:
                bridge = " This adds context to the ticker move we were just looking at."

    final_text = f"{icon} {prefix} {selected['msg']} {impact}.{bridge} Want a deep dive?"

    follow_up_map = {
        "concentration": f"Analyze rebalancing for {selected['data'].get('sector')} heavy exposure",
        "divergence": f"Why is {selected['data'].get('ticker')} decoupling from {selected['data'].get('sector')}?",
        "conflict": "Give me a full breakdown of the news vs price conflict"
    }

    return {
        "text": final_text,
        "followup_query": follow_up_map.get(selected["type"]),
        "type": selected["type"],
        "topic": selected["topic"]
    }
