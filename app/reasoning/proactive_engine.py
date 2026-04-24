import logging

logger = logging.getLogger(__name__)

def detect_proactive_signals(tool_data: dict, user_query: str) -> dict:
    """
    Scans analytical data for high-value proactive triggers.
    """
    triggers = []
    full_analysis = tool_data.get("full_analysis", {})
    reason_results = tool_data.get("reason", {})
    risk_results = tool_data.get("risk", {})

    # 1. Concentration Risk (>40%)
    exposure = full_analysis.get("sector_exposure", {})
    for sector, alloc in exposure.items():
        if alloc > 40:
            triggers.append({
                "type": "concentration",
                "sector": sector,
                "value": alloc,
                "msg": f"Your portfolio is heavily concentrated in {sector} ({alloc:.1f}%). This increases sector-specific downside risk."
            })

    # 2. Holding Divergence (>2% from sector)
    holdings = full_analysis.get("ranked_holdings", [])
    for h in holdings[:3]: # Top 3 only
        stock_change = h.get("daily_change", 0.0)
        # Mock sector average lookup logic
        sector_change = full_analysis.get("sector_performance", {}).get(h.get("sector"), 0.0)
        if abs(stock_change - sector_change) > 2.0:
            triggers.append({
                "type": "divergence",
                "ticker": h.get("ticker"),
                "sector": h.get("sector"),
                "msg": f"Interesting — {h.get('ticker')} is diverging from the {h.get('sector')} sector by over 2% today."
            })

    # 3. Unaddressed Conflicts
    conflicts = reason_results.get("conflicts", [])
    query_lower = user_query.lower()
    if conflicts and not any(kw in query_lower for kw in ["conflict", "mismatch", "why falling", "why rising"]):
        triggers.append({
            "type": "conflict",
            "msg": "I detected a conflict between positive news and falling prices in some holdings."
        })

    return triggers

def generate_proactive_insight(tool_data: dict, user_query: str, last_insight: str = None) -> dict:
    """
    Selects ONE unique proactive insight and follow-up query.
    """
    triggers = detect_proactive_signals(tool_data, user_query)
    
    if not triggers:
        return None

    # Pick the most relevant trigger (priority: conflict > divergence > concentration)
    # Filter out if identical to last insight
    selected = None
    for t_type in ["conflict", "divergence", "concentration"]:
        candidate = next((t for t in triggers if t["type"] == t_type), None)
        if candidate and candidate["msg"] != last_insight:
            selected = candidate
            break
    
    if not selected:
        return None

    # Map to follow-up prompt
    follow_up_map = {
        "concentration": f"Analyze concentration risk in {selected.get('sector')}",
        "divergence": f"Why is {selected.get('ticker')} diverging from {selected.get('sector')}?",
        "conflict": "Explain the price-news conflict in detail"
    }

    return {
        "insight": selected["msg"],
        "follow_up": follow_up_map.get(selected["type"]),
        "type": selected["type"]
    }
