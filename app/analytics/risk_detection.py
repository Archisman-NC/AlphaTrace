import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def detect_concentration_risk(sector_exposure: Dict[str, float]) -> List[dict]:
    """
    Identifies overexposure to specific sectors or sector combinations.
    Applies deterministic financial concentration rules.
    """
    if not sector_exposure:
        return []

    risks = []
    
    # Sort sectors by exposure (descending) for Rule 2
    sorted_sectors = sorted(
        sector_exposure.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    phrase_pool = ["broadly diversified holdings", "diversified asset allocation", "cross-sector exposure"]
    phrase_idx = 0
    
    # Rule 1: High Risk (Any single sector > 40%)
    for sector, weight in sorted_sectors:
        if weight > 0.40:
            percentage = weight * 100
            if sector == "DIVERSIFIED HOLDINGS":
                term = phrase_pool[phrase_idx % len(phrase_pool)]
                phrase_idx += 1
                desc = f"CRITICAL: {percentage:.2f}% high concentration in {term}"
            else:
                desc = f"CRITICAL: {percentage:.2f}% exposure to {sector}"
                
            risks.append({
                "type": "sector_concentration",
                "severity": "critical",
                "sector": sector,
                "weight": weight,
                "description": desc
            })

    # Rule 2: Medium Risk (Top 2 combined > 70%)
    if len(sorted_sectors) >= 2:
        s1, w1 = sorted_sectors[0]
        s2, w2 = sorted_sectors[1]
        combined_weight = w1 + w2
        
        if combined_weight > 0.70:
            percentage = combined_weight * 100
            if "DIVERSIFIED HOLDINGS" in [s1, s2]:
                term = phrase_pool[phrase_idx % len(phrase_pool)]
                phrase_idx += 1
                desc = f"MEDIUM: {percentage:.2f}% high concentration involving {term}"
            else:
                desc = f"MEDIUM: {percentage:.2f}% combined exposure to {s1} and {s2}"
                
            risks.append({
                "type": "sector_concentration",
                "severity": "medium",
                "sectors": [s1, s2],
                "weight": combined_weight,
                "description": desc
            })

    return risks
