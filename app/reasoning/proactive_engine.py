import logging
import random
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

PREFIXES = [
    "Also worth noting —",
    "One thing to watch —",
    "Quick heads-up —",
    "Another pulse signal —",
    "From your recent data —"
]

def generate_proactive_insight(tool_data: dict, user_query: str, session_memory: list = None, last_topic: str = None) -> Optional[dict]:
    """
    Advanced Proactive Engine:
    - Weighted Prioritization
    - Impact Injection ("Why this matters")
    - Anti-Template Generation
    """
    full_analysis = tool_data.get("full_analysis", {})
    reason_results = tool_data.get("reason", {})
    risk_results = tool_data.get("risk", {})
    
    potential_triggers = []

    # 1. Concentration Check (Weight 3)
    exposure = full_analysis.get("sector_exposure", {})
    for sector, alloc in exposure.items():
        if alloc > 40:
            severity = "high" if alloc > 50 else "medium"
            potential_triggers.append({
                "type": "concentration",
                "weight": 3,
                "severity": severity,
                "topic": f"concentration_{sector}",
                "data": {"sector": sector, "alloc": alloc},
                "msg": f"Your portfolio is heavily concentrated in {sector} ({alloc:.1f}%).",
                "impact": "This concentration increases your sensitivity to sector-specific volatility."
            })

    # 2. Divergence Check (Weight 2)
    holdings = full_analysis.get("ranked_holdings", [])
    for h in holdings[:3]:
        stock_change = h.get("daily_change", 0.0)
        sector_change = full_analysis.get("sector_performance", {}).get(h.get("sector"), 0.0)
        delta = abs(stock_change - sector_change)
        if delta > 2.0:
            severity = "high" if delta > 3.0 else "medium"
            potential_triggers.append({
                "type": "divergence",
                "weight": 2,
                "severity": severity,
                "topic": f"divergence_{h.get('ticker')}",
                "data": {"ticker": h.get("ticker"), "sector": h.get("sector")},
                "msg": f"{h.get('ticker')} is decoupling from the {h.get('sector')} sector trends.",
                "impact": "This divergence may indicate a fundamental shift or unique risk in this holding."
            })

    # 3. Conflict Check (Weight 2)
    conflicts = reason_results.get("conflicts", [])
    if conflicts and "conflict" not in user_query.lower():
        potential_triggers.append({
            "type": "conflict",
            "weight": 2,
            "severity": "medium",
            "topic": "price_news_mismatch",
            "msg": "I've detected a mismatch between positive sentiment and price action in some holdings.",
            "impact": "This conflict often signals hidden selling pressure or market skepticism."
        })

    if not potential_triggers:
        return None

    # FILTER & PRIORITIZE
    active_triggers = [t for t in potential_triggers if t["topic"] != last_topic]
    if not active_triggers: return None

    for t in active_triggers:
        t["score"] = t["weight"] + (1 if t["severity"] == "high" else 0)
    
    active_triggers.sort(key=lambda x: x["score"], reverse=True)
    selected = active_triggers[0]

    # NARRATIVE CONSTRUCTION
    icon = "⚠️" if selected["severity"] == "high" else "ℹ️"
    prefix = random.choice(PREFIXES)
    body = selected["msg"]
    impact = selected["impact"]
    
    # MEMORY BRIDGING
    bridge = ""
    if session_memory and len(session_memory) > 0:
        last_turn = session_memory[-1]
        if selected["type"] == "concentration" and selected["data"]["sector"] in [d.get("sector") for d in last_turn.get("drivers", [])]:
            bridge = f" This reinforces the {selected['data']['sector']} focus we analyzed earlier."
        elif selected["type"] == "divergence" and selected["data"]["ticker"] in last_turn.get("user_query", ""):
            bridge = " This adds context to the ticker move we were just looking at."

    final_text = f"{icon} {prefix} {body} {impact}{bridge} Want a deep dive investigation?"

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
