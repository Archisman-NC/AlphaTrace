from app.utils.helpers import safe_float
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_top_impact_sectors(
    sector_impacts: Dict[str, dict],
    top_n: int = 3
) -> List[dict]:
    """
    Identifies the sectors with the highest absolute impact on the portfolio.
    Used to prioritize signals for generated explanations.
    """
    if not sector_impacts:
        return []

    impact_list = []
    
    for sector, data in sector_impacts.items():
        impact_val = data.get("impact")
        
        if impact_val is None:
            logger.warning(f"Skipping {sector} in top impact selection: Missing impact value.")
            continue
            
        try:
            impact_list.append({
                "sector": sector,
                "impact": safe_float(impact_val)
            })
        except (ValueError, TypeError):
            continue

    # Step 3: Sort by absolute impact descending
    impact_list.sort(key=lambda x: abs(x["impact"]), reverse=True)

    # Step 4: Select top N
    return impact_list[:top_n]
