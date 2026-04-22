import logging
from typing import Dict

logger = logging.getLogger(__name__)

def compute_portfolio_metrics(portfolio: dict) -> Dict[str, float]:
    """
    Extracts high-level portfolio performance metrics (Value, PnL, % Change).
    Prioritizes the 'summary' block found in current mock data.
    """
    # Step 1 & 2: Extract values from 'summary' block or top-level keys
    summary = portfolio.get("summary", {})
    
    # Mapping actual mock data keys to the requested internal structure
    total_value = summary.get("current_value", portfolio.get("total_value", 0))
    daily_pnl = summary.get("day_pnl", portfolio.get("daily_pnl", 0))
    daily_change = summary.get("day_change_percent", portfolio.get("daily_change_percent", 0))

    # Step 3 & 4: Type Safety and Final Structure
    try:
        return {
            "total_value": float(total_value),
            "daily_pnl": float(daily_pnl),
            "daily_change_percent": float(daily_change)
        }
    except (ValueError, TypeError) as e:
        logger.warning(f"Error casting portfolio metrics to float: {e}")
        return {
            "total_value": 0.0,
            "daily_pnl": 0.0,
            "daily_change_percent": 0.0
        }
