import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_outputs(
    sector_exposure: Dict[str, float],
    top_sectors: List[dict],
    stock_exposure_map: Dict[str, dict],
    risks: List[dict]
) -> Dict[str, Any]:
    """
    Final validation layer ensuring logical and numerical consistency 
    of the portfolio intelligence pipeline.
    """
    warnings = []
    is_valid = True
    
    # 1. Validate Sector Weight Sum (0.95 <= total_weight <= 1.05)
    total_weight = sum(sector_exposure.values())
    if not (0.95 <= total_weight <= 1.05):
        msg = f"Inconsistent weight sum: {total_weight:.4f} (expected ~1.0)"
        warnings.append(msg)
        if total_weight < 0.5 or total_weight > 1.5: # Critical failure
            is_valid = False
            
    # 2. Validate Top Sectors
    if sector_exposure and not top_sectors:
        warnings.append("Analytical Gap: Sector exposure exists but no top impact sectors identified.")
        
    for ts in top_sectors:
        sector = ts.get("sector")
        impact = ts.get("impact")
        if sector not in sector_exposure:
            warnings.append(f"Referential Error: Top sector '{sector}' not found in portfolio exposure map.")
        if not isinstance(impact, (int, float)):
             warnings.append(f"Type Error: Impact for sector '{sector}' is not numeric.")

    # 3. Validate Risk Detection (Causal Audit)
    # Rule 1: High Risk (>40%)
    for sector, weight in sector_exposure.items():
        if weight > 0.40:
            # Look for a corresponding critical risk in the list
            found = any(r.get("severity") == "critical" and (r.get("sector") == sector or sector in r.get("sectors", [])) for r in risks)
            if not found:
                warnings.append(f"Missing Critical Risk: Exposure to '{sector}' is {weight*100:.2f}%, but no risk flag exists.")
                is_valid = False

    # Rule 2: Medium Risk (Top 2 combined > 70%)
    sorted_exp = sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_exp) >= 2:
        top2_weight = sorted_exp[0][1] + sorted_exp[1][1]
        if top2_weight > 0.70:
            found = any(r.get("severity") == "medium" and r.get("weight", 0) >= 0.70 for r in risks)
            if not found:
                warnings.append(f"Missing Medium Risk: Top 2 sectors combined weight is {top2_weight*100:.2f}%, but no risk flag exists.")

    # 4. Validate Stock Coverage
    if stock_exposure_map:
        # Find highest weight stock in map
        max_stock = max(stock_exposure_map.items(), key=lambda x: x[1].get("weight", 0))
        symbol, data = max_stock
        if data.get("importance_rank") != 1:
            warnings.append(f"Ranking Inconsistency: '{symbol}' has highest weight but rank is {data.get('importance_rank')}.")

    # 5. Data Type Integrity (No None checks)
    for sector, weight in sector_exposure.items():
        if weight is None or not isinstance(weight, (int, float)):
            warnings.append(f"Null/Non-numeric Weight: Sector '{sector}' has invalid weight data.")
            is_valid = False

    summary = f"Validation checked {len(sector_exposure)} sectors and {len(risks)} risks."
    if not warnings:
        summary += " No issues found."
    else:
        summary += f" Detected {len(warnings)} issues."

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "summary": summary
    }
