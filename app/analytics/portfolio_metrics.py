import logging
from app.utils.helpers import safe_float
from typing import Dict

logger = logging.getLogger(__name__)

def compute_portfolio_metrics(portfolio: dict) -> Dict[str, float]:
    """
    Extracts high-level portfolio performance metrics (Value, PnL, % Change).
    Handles multiple schemas: 'summary' block, 'analytics.day_summary', or top-level.
    """
    # Try 'summary' block (Portfolios 001, 002)
    summary = portfolio.get("summary", {})
    
    # Try 'analytics.day_summary' (Portfolio 003)
    analytics_summary = portfolio.get("analytics", {}).get("day_summary", {})

    total_value = summary.get("current_value", analytics_summary.get("current_value", portfolio.get("current_value", 0)))
    daily_pnl = summary.get("day_pnl", analytics_summary.get("day_change_absolute", portfolio.get("daily_pnl", 0)))
    daily_change = summary.get("day_change_percent", analytics_summary.get("day_change_percent", portfolio.get("daily_change_percent", 0)))

    try:
        return {
            "total_value": safe_float(total_value),
            "daily_pnl": safe_float(daily_pnl),
            "daily_change_percent": safe_float(daily_change)
        }
    except (ValueError, TypeError) as e:
        logger.warning(f"Error casting portfolio metrics to float: {e}")
        return {
            "total_value": 0.0,
            "daily_pnl": 0.0,
            "daily_change_percent": 0.0
        }
