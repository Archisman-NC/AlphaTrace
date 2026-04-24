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
    """
    # 1. Normalize Drivers
    reason_results = tool_data.get("reason", {})
    raw_chains = reason_results.get("chains", [])
    normalized_drivers = []
    
    for chain in raw_chains[:3]: # Limit to top 3 for context density
        normalized_drivers.append({
            "sector": chain.get("sector", "Unknown"),
            "cause": chain.get("trigger", "broad market movement"),
            "impact": float(chain.get("impact", 0.0))
        })

    # 2. Normalize Risks
    risk_results = tool_data.get("risk", {})
    raw_risks = risk_results.get("risks", [])
    normalized_risks = []
    
    for risk in raw_risks[:3]:
        normalized_risks.append({
            "type": risk.get("type", "Concentration"),
            "severity": float(risk.get("severity", 0.5)),
            "description": risk.get("description", "Potential volatility")
        })

    # 3. Normalize Metrics
    full_analysis = tool_data.get("full_analysis", {})
    normalized_metrics = {
        "portfolio_change": float(full_analysis.get("daily_change_percent", 0.0)),
        "top_sector": full_analysis.get("top_sector", "Diversified"),
        "concentration": float(risk_results.get("global_concentration_score", 0.0))
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
    Prioritization Engine:
    - Keywords (risk, why, sector) boost specific memory features.
    - Applies recency weighting (decay).
    """
    if not memory:
        return {"drivers": [], "risks": [], "metrics": {}}

    recent = memory[-k:]
    query_lower = query.lower()
    
    # Feature Boosting Logic
    boost_risks = any(kw in query_lower for kw in ["risk", "downside", "danger", "safe"])
    boost_drivers = any(kw in query_lower for kw in ["why", "because", "reason", "cause", "driven"])
    
    aggregated_context = {
        "recent_drivers": [],
        "recent_risks": [],
        "last_metrics": recent[-1]["metrics"]
    }

    for i, turn in enumerate(reversed(recent)):
        # Recency Weight: 1/(1+age)
        weight = 1.0 / (1.0 + i)
        
        if boost_drivers or not boost_risks:
            for d in turn["drivers"]:
                d_weighted = d.copy()
                d_weighted["relevance_score"] = weight * (1.5 if boost_drivers else 1.0)
                aggregated_context["recent_drivers"].append(d_weighted)
        
        if boost_risks or not boost_drivers:
            for r in turn["risks"]:
                r_weighted = r.copy()
                r_weighted["relevance_score"] = weight * (1.5 if boost_risks else 1.0)
                aggregated_context["recent_risks"].append(r_weighted)

    # Sort by relevance and trim
    aggregated_context["recent_drivers"].sort(key=lambda x: x["relevance_score"], reverse=True)
    aggregated_context["recent_risks"].sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "drivers": aggregated_context["recent_drivers"][:3],
        "risks": aggregated_context["recent_risks"][:3],
        "metrics": aggregated_context["last_metrics"]
    }
