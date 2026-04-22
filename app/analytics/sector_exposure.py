import logging
from typing import Dict
from collections import defaultdict
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def compute_sector_exposure(
    loader: DataLoader,
    normalized_holdings: Dict[str, dict],
    portfolio: dict
) -> Dict[str, float]:
    """
    Computes the TRUE sector exposure of a portfolio by combining:
    1. Direct stock holdings (from normalized_holdings)
    2. Indirect mutual fund holdings (by looking up fund allocation in DataLoader)
    
    Returns a dictionary of normalized sector weights (decimal, e.g. 0.42).
    """
    sector_exposure = defaultdict(float)
    
    # --- STEP 1: Direct Stock Exposure ---
    for symbol, data in normalized_holdings.items():
        sector = data.get("sector", "UNCLASSIFIED").upper()
        weight = data.get("weight", 0.0)
        sector_exposure[sector] += weight

    # --- STEP 2: Indirect Mutual Fund Exposure ---
    mf_list = portfolio.get("mutual_funds", [])
    
    for fund in mf_list:
        # Extract fund identification and weight
        scheme_code = fund.get("scheme_code", fund.get("mutual_fund_code", "UNKNOWN")).strip().upper()
        
        # Consistent weight normalization logic [0, 1]
        raw_weight = fund.get("weight_in_portfolio", fund.get("weight_percent", fund.get("weight", 0.0)))
        try:
            fund_weight = float(raw_weight)
            if fund_weight > 1.0:
                fund_weight = fund_weight / 100.0
            fund_weight = max(0.0, min(1.0, fund_weight))
        except (ValueError, TypeError):
            logger.warning(f"Invalid weight {raw_weight} for fund {scheme_code}, skipping.")
            continue

        if fund_weight == 0:
            continue

        # Fetch fund details for sector breakdown
        mf_data = loader.get_mutual_fund(scheme_code)
        if not mf_data:
            logger.warning(f"Metadata missing for fund {scheme_code}. Attributing to UNCLASSIFIED.")
            sector_exposure["UNCLASSIFIED"] += fund_weight
            continue

        # Extract sector allocation (usually in 0-100 range in JSON)
        allocation = mf_data.get("sector_allocation", {})
        if not allocation:
            logger.warning(f"No sector allocation found for fund {scheme_code}. Attributing to UNCLASSIFIED.")
            sector_exposure["UNCLASSIFIED"] += fund_weight
            continue

        # Distribute fund weight across sectors
        total_allocated = 0.0
        for sector_name, percent in allocation.items():
            norm_sector_name = sector_name.upper()
            try:
                sector_percent = float(percent)
                # Convert to decimal (e.g. 26.2 -> 0.262)
                sector_decimal = sector_percent / 100.0
                
                contribution = fund_weight * sector_decimal
                sector_exposure[norm_sector_name] += contribution
                total_allocated += sector_percent
            except (ValueError, TypeError):
                continue
        
        # Handle unallocated portion of the fund (e.g. cash or missing data)
        if total_allocated < 99.0: # Close to 100
            remainder_decimal = (100.0 - total_allocated) / 100.0
            sector_exposure["UNCLASSIFIED"] += fund_weight * remainder_decimal

    # --- STEP 3: Final Normalization ---
    # Convert defaultdict to regular dict and ensure float types
    final_exposure = {s: float(w) for s, w in sector_exposure.items() if w > 0}
    
    # Sort by exposure descending
    return dict(sorted(final_exposure.items(), key=lambda x: x[1], reverse=True))
