import logging
from typing import Dict
from app.ingestion.data_loader import DataLoader
from app.utils.helpers import normalize_sector

logger = logging.getLogger(__name__)

def compute_sector_performance(loader: DataLoader, trends: Dict[str, dict] = None) -> Dict[str, float]:
    """
    Extracts high-fidelity sector performance mapping (Name: % Change).
    Used as the 'Analytical Truth' for asset-sector divergence detection.
    """
    if trends:
        # If we already have processed trends, extract the change mapping
        return {s: d.get("change", 0.0) for s, d in trends.items()}

    # Fallback to direct market data extraction
    try:
        market_data = loader.get_market_data()
        sector_data = market_data.get("sector_performance", {})
        
        perf_map = {}
        for sector, details in sector_data.items():
            normalized = normalize_sector(sector)
            perf_map[normalized] = details.get("change_percent", 0.0)
            
        return perf_map
    except Exception as e:
        logger.error(f"Failed to compute sector performance: {e}")
        return {}
