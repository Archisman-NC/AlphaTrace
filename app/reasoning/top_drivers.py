import logging
from typing import List

logger = logging.getLogger(__name__)

def select_top_drivers(
    causal_chains: List[dict],
    top_n: int = 3
) -> List[dict]:
    """
    Sorts and filters the most impactful causal chains explaining portfolio movement.
    Provides clean, minimal output for final explanation generation.
    """
    if not causal_chains:
        return []

    valid_chains = []

    for chain in causal_chains:
        impact = chain.get("impact")
        
        # Step 2: Filter valid chains
        if impact is None or not isinstance(impact, (int, float)):
            continue

        try:
            raw_sector = chain.get("sector", "Unknown Sector")
            clean_sector = "Diversified Holdings" if raw_sector == "UNCLASSIFIED" else raw_sector
            
            raw_stocks = chain.get("stocks", [])
            clean_stocks = raw_stocks[:2] if isinstance(raw_stocks, list) else []

            valid_chains.append({
                "sector": clean_sector,
                "impact": float(impact),
                "reason": chain.get("news", "Unknown Reason"),
                "stocks": clean_stocks
            })
        except (ValueError, TypeError):
            continue

    # Step 3: Sort by absolute impact descending
    valid_chains.sort(key=lambda x: abs(x["impact"]), reverse=True)

    # Step 4: Select Top N
    return valid_chains[:top_n]
