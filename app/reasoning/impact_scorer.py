import logging
from typing import List

logger = logging.getLogger(__name__)

def compute_impact_scores(
    causal_chains: List[dict]
) -> List[dict]:
    """
    Computes and attaches the quantitative impact score for each causal chain.
    Impact = sector_change * portfolio_weight.
    """
    if not causal_chains:
        return []

    enriched_chains = []

    for chain in causal_chains:
        sector_change = chain.get("sector_change")
        portfolio_weight = chain.get("portfolio_weight")

        if sector_change is None or portfolio_weight is None:
            continue

        try:
            impact_val = float(sector_change) * float(portfolio_weight)
            
            # Enrich existing chain object
            chain_copy = chain.copy()
            chain_copy["impact"] = round(impact_val, 2)
            enriched_chains.append(chain_copy)
            
        except (ValueError, TypeError):
            continue

    # Sort chains by absolute impact descending
    enriched_chains.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return enriched_chains
