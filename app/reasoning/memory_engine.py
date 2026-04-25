from app.utils.helpers import safe_float
import time
from typing import List, Dict, Any, Optional
from typing import List, Dict, Any, Optional

def normalize_memory_turn(
    portfolio_id: str,
    user_query: str,
    intents: List[str],
    summary: str,
    tool_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standardizes a memory episode using Direct-Slicing.
    """
    reason_results = tool_data.get("reason", {})
    raw_drivers = reason_results.get("drivers", reason_results.get("chains", []))
    
    # DIRECT SLICE DRIVERS
    print(f"[DEBUG] Slicing drivers type: {type(raw_drivers)}")
    normalized_drivers = []
    # DIRECT SLICE MIGRATION
    for d in raw_drivers[:3]: 
        normalized_drivers.append({
            "sector": d.get("sector", "Unknown"),
            "cause": d.get("trigger", d.get("cause", "broad market movement")),
            "impact": safe_float(d.get("impact", 0.0))
        })

    # DIRECT SLICE RISKS
    risk_results = tool_data.get("risk", {})
    raw_risks = risk_results.get("risks", [])
    print(f"[DEBUG] Slicing risks type: {type(raw_risks)}")
    
    normalized_risks = []
    # DIRECT SLICE MIGRATION
    for r in raw_risks[:3]:
        normalized_risks.append({
            "type": r.get("type", "Concentration"),
            "severity": safe_float(r.get("severity", 0.5)),
            "description": r.get("description", "Potential volatility")
        })

    # AGGREGATE METRICS
    full_analysis = tool_data.get("full_analysis", {})
    metrics = {**full_analysis.get("metrics", {}), **reason_results.get("metrics", {}), **risk_results.get("metrics", {})}
    
    normalized_metrics = {
        "portfolio_change": safe_float(metrics.get("daily_change_percent", 0.0)),
        "top_sector": metrics.get("top_sector", "Diversified"),
        "concentration": safe_float(metrics.get("global_concentration_score", 0.0))
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
    Prioritization Engine using Direct-Slicing.
    """
    # DIRECT SLICE RECENT TURNS
    print(f"[DEBUG] Slicing memory type: {type(memory)}")
    # DIRECT SLICE MIGRATION (REVERSE + LIMIT)
    recent = memory[::-1][:k]
    
    if not recent:
        return {"drivers": [], "risks": [], "metrics": {}}

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
