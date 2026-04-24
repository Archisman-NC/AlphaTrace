import time
from typing import List, Dict, Any, Optional

def normalize_memory_turn(
    portfolio_id: str,
    user_query: str,
    intents: List[str],
    summary: str,
    tool_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standardizes a memory episode to ensure structural consistency for active reasoning.
    Uses 'drivers' and 'risks' as primary keys.
    """
    # 1. Normalize Drivers (Reasoning)
    # Check both potential keys for backward/transitional consistency
    reason_results = tool_data.get("reason", {})
    raw_drivers = reason_results.get("drivers", reason_results.get("chains", []))
    
    # Force list type before slicing to prevent slice errors
    normalized_drivers = []
    for d in list(raw_drivers)[:3]: 
        normalized_drivers.append({
            "sector": d.get("sector", "Unknown"),
            "cause": d.get("trigger", d.get("cause", "broad market movement")),
            "impact": float(d.get("impact", 0.0))
        })

    # 2. Normalize Risks
    risk_results = tool_data.get("risk", {})
    raw_risks = risk_results.get("risks", [])
    
    normalized_risks = []
    for r in list(raw_risks)[:3]:
        normalized_risks.append({
            "type": r.get("type", "Concentration"),
            "severity": float(r.get("severity", 0.5)),
            "description": r.get("description", "Potential volatility")
        })

    # 3. Normalize Metrics
    full_analysis = tool_data.get("full_analysis", {})
    metrics = {**full_analysis.get("metrics", {}), **reason_results.get("metrics", {}), **risk_results.get("metrics", {})}
    
    normalized_metrics = {
        "portfolio_change": float(metrics.get("daily_change_percent", 0.0)),
        "top_sector": metrics.get("top_sector", "Diversified"),
        "concentration": float(metrics.get("global_concentration_score", 0.0))
    }

    return {
        "portfolio_id": portfolio_id,
        "user_query": user_query,
        "intents": intents,
        "summary": summary,
        "drivers": normalized_drivers,
        "risks": normalized_risks,
        "metrics": normalized_metrics,
        "timestamp": time.time()
    }

def extract_relevant_memory(query: str, memory: List[dict], k=3) -> Dict[str, Any]:
    """
    Prioritization Engine with Recency weighting and feature boosting.
    """
    if not isinstance(memory, list) or not memory:
        return {"drivers": [], "risks": [], "metrics": {}}

    recent = list(memory)[-k:]
    query_lower = query.lower()
    
    boost_risks = any(kw in query_lower for kw in ["risk", "downside", "danger", "safe"])
    boost_drivers = any(kw in query_lower for kw in ["why", "because", "reason", "cause", "driven"])
    
    aggregated_context = {
        "recent_drivers": [],
        "recent_risks": [],
        "last_metrics": recent[-1].get("metrics", {})
    }

    for i, turn in enumerate(reversed(recent)):
        weight = 1.0 / (1.0 + i)
        
        if boost_drivers or not boost_risks:
            for d in turn.get("drivers", []):
                d_weighted = d.copy()
                d_weighted["relevance_score"] = weight * (1.5 if boost_drivers else 1.0)
                aggregated_context["recent_drivers"].append(d_weighted)
        
        if boost_risks or not boost_drivers:
            for r in turn.get("risks", []):
                r_weighted = r.copy()
                r_weighted["relevance_score"] = weight * (1.5 if boost_risks else 1.0)
                aggregated_context["recent_risks"].append(r_weighted)

    aggregated_context["recent_drivers"].sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    aggregated_context["recent_risks"].sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    
    return {
        "drivers": aggregated_context["recent_drivers"][:3],
        "risks": aggregated_context["recent_risks"][:3],
        "metrics": aggregated_context["last_metrics"]
    }
