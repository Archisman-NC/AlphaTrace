import logging
from typing import Dict, Any, List
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def process_mutual_funds(
    loader: DataLoader,
    portfolio: Dict[str, Any],
    mode: str = "simple"
) -> Dict[str, Any]:
    """
    Interprets mutual fund impact for the reasoning engine.
    Supports 'simple' (baseline Diversified) and 'detailed' (recursive look-through) modes.
    """
    mf_list = portfolio.get("mutual_funds", [])
    if not mf_list:
        return {}

    if mode == "simple":
        # Option A: Baseline Diversified logic
        details = []
        for fund in mf_list:
            name = fund.get("scheme_name", fund.get("mutual_fund_code", "Unknown Fund"))
            details.append({
                "fund": name,
                "type": "mutual_fund",
                "impact": "neutral",
                "note": "Mutual funds assumed diversified across sectors"
            })
            
        return {
            "mode": "simple",
            "mf_contribution": "neutral",
            "mf_details": details
        }
    
    elif mode == "detailed":
        # Option B: Detailed look-through logic
        sector_exposure = {}
        processed_details = []
        
        for fund in mf_list:
            code = fund.get("scheme_code", fund.get("mutual_fund_code", "UNKNOWN"))
            weight = fund.get("weight_in_portfolio", fund.get("weight", 0))
            
            # Normalize weight to decimal
            if weight > 1:
                weight = weight / 100.0
            
            mf_data = loader.get_mutual_fund(code)
            if not mf_data:
                processed_details.append({"fund": code, "status": "data_missing"})
                continue
                
            allocation = mf_data.get("sector_allocation", {})
            for sector, pct in allocation.items():
                influence = (pct / 100.0) * weight
                sector_exposure[sector] = sector_exposure.get(sector, 0) + influence
                
            processed_details.append({
                "fund": code,
                "sectors_mapped": list(allocation.keys()),
                "portfolio_impact": "calculated"
            })
            
        return {
            "mode": "detailed",
            "sector_exposure": {k: round(v, 4) for k, v in sector_exposure.items()},
            "mf_details": processed_details
        }
    
    else:
        logger.error(f"Unknown mode '{mode}' in process_mutual_funds")
        return {}
