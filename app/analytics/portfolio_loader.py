import logging
from typing import Dict, Any
from app.ingestion.data_loader import DataLoader

logger = logging.getLogger(__name__)

def load_portfolio(loader: DataLoader, portfolio_id: str) -> Dict[str, Any]:
    """
    Fetches and normalizes a specific user portfolio for analysis.
    Ensures data consistency in symbols and structure.
    """
    # Step 1: Fetch raw data
    portfolio_data = loader.get_portfolio(portfolio_id)
    
    # Step 2: Validate input
    if not portfolio_data:
        logger.warning(f"Portfolio ID '{portfolio_id}' not found.")
        return {}
        
    # Step 3 & 5: Extract and Normalize fields
    # Metadata is at the top level of the portfolio dictionary in portfolios.json
    portfolio_type = portfolio_data.get("portfolio_type", "UNKNOWN").lower()
    p_id = portfolio_data.get("user_id", portfolio_id)
    
    holdings = portfolio_data.get("holdings", {})
    
    # Normalizing stock holdings
    stocks_raw = holdings.get("stocks", [])
    clean_stocks = []
    for s in stocks_raw:
        clean_item = s.copy()
        if "symbol" in clean_item:
            clean_item["symbol"] = clean_item["symbol"].strip().upper()
        clean_stocks.append(clean_item)
        
    # Normalizing mutual fund holdings
    mf_raw = holdings.get("mutual_funds", [])
    clean_mf = []
    for mf in mf_raw:
        clean_item = mf.copy()
        # Supporting multiple MF identification keys for robustness
        for key in ["mutual_fund_code", "scheme_code"]:
            if key in clean_item:
                clean_item[key] = clean_item[key].strip().upper()
        clean_mf.append(clean_item)
        
    # Step 4: Normalize output
    return {
        "portfolio_id": p_id,
        "type": portfolio_type,
        "stocks": clean_stocks,
        "mutual_funds": clean_mf,
        "summary": portfolio_data.get("summary", {}),
        "analytics": portfolio_data.get("analytics", {}),
        "current_value": portfolio_data.get("current_value", 0),
        "user_name": portfolio_data.get("user_name", "Unknown")
    }
