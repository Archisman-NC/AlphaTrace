import logging
from app.utils.helpers import safe_float
from typing import Dict

logger = logging.getLogger(__name__)

def compute_sector_impact(sector_link: Dict[str, dict]) -> Dict[str, dict]:
    """
    Computes the contribution of each sector to the portfolio's total movement.
    Formula: impact = portfolio_weight * sector_change
    """
    if not sector_link:
        return {}

    sector_impacts = {}
    
    for sector, data in sector_link.items():
        weight = data.get("portfolio_weight")
        change = data.get("sector_change")
        
        # Step 5: Edge case handling
        if weight is None or change is None:
            logger.warning(f"Skipping {sector} in impact calculation: Missing weight or change.")
            continue
            
        try:
            # Step 2: Compute Impact
            impact_val = safe_float(weight) * safe_float(change)
            
            # Step 3 & 4: Build output and round
            sector_impacts[sector] = {
                "impact": round(impact_val, 2),
                "portfolio_weight": weight,
                "sector_change": change,
                "sentiment": data.get("sentiment", "neutral")
            }
        except (ValueError, TypeError):
            logger.warning(f"Non-numeric values encountered for {sector}, skipping.")
            continue

    return sector_impacts
