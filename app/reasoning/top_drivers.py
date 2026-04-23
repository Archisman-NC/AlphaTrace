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

    sector_groups = {}

    for chain in causal_chains:
        impact = chain.get("impact")
        
        # Step 2: Filter valid chains
        if impact is None or not isinstance(impact, (int, float)):
            continue

        try:
            raw_sector = chain.get("sector", "Unknown Sector")
            clean_sector = "Diversified Holdings" if raw_sector == "DIVERSIFIED HOLDINGS" else raw_sector
            impact_val = float(impact)
            reason = str(chain.get("news", "Unknown Reason")).strip()
            
            if clean_sector not in sector_groups:
                sector_groups[clean_sector] = {
                    "sector": clean_sector,
                    "impact": impact_val,
                    "reasons": {reason} if reason else set(),
                    "stocks": chain.get("stocks", [])
                }
            else:
                existing = sector_groups[clean_sector]
                # Keep max absolute impact
                if abs(impact_val) > abs(existing["impact"]):
                    existing["impact"] = impact_val
                
                # Add unique reasons
                if reason:
                    existing["reasons"].add(reason)
                    
                # Store unique stocks preserving append order
                for s in chain.get("stocks", []):
                    if s not in existing["stocks"]:
                        existing["stocks"].append(s)
                        
        except (ValueError, TypeError):
            continue

    # Flatten and build final schemas
    valid_chains = []
    
    for data in sector_groups.values():
        reasons_list = list(data["reasons"])
        if len(reasons_list) > 1:
            reason_str = " and ".join(reasons_list[:2])
        else:
            reason_str = reasons_list[0] if reasons_list else "Unknown Reason"
            
        valid_chains.append({
            "sector": data["sector"],
            "impact": data["impact"],
            "reason": reason_str,
            "stocks": data["stocks"][:2]  # Limit to max 2 stocks
        })

    # Sort by absolute impact descending
    valid_chains.sort(key=lambda x: abs(x["impact"]), reverse=True)

    # Select Top N
    return valid_chains[:top_n]
