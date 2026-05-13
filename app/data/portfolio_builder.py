from app.data.market_data import (
    fetch_portfolio,
    SECTOR_MAP,
    NIFTY50_TICKERS,
    REQUIRED_COLUMNS,
)
import pandas as pd
import logging
from typing import Dict, Any

# Initialize logger
logger = logging.getLogger(__name__)

PORTFOLIOS = {
    "PORTFOLIO_001": {
        "name": "Rahul Sharma",
        "holdings": {
            "HDFCBANK.NS": 0.15,
            "TCS.NS": 0.15,
            "RELIANCE.NS": 0.15,
            "INFY.NS": 0.10,
            "LT.NS": 0.10,
            "ITC.NS": 0.10,
            "AXISBANK.NS": 0.10,
            "SBIN.NS": 0.08,
            "ICICIBANK.NS": 0.07
        }
    },

    "PORTFOLIO_002": {
        "name": "Priya Patel",
        "holdings": {
            "HDFCBANK.NS": 0.30,
            "ICICIBANK.NS": 0.25,
            "KOTAKBANK.NS": 0.20,
            "AXISBANK.NS": 0.15,
            "SBIN.NS": 0.10
        }
    },

    "PORTFOLIO_003": {
        "name": "Arun Krishnamurthy",
        "holdings": {
            "TCS.NS": 0.25,
            "INFY.NS": 0.25,
            "ITC.NS": 0.20,
            "LT.NS": 0.15,
            "RELIANCE.NS": 0.15
        }
    }
}

def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    """
    Validate that portfolio weights sum to approximately 1.0.
    
    Args:
        weights: Dictionary of ticker: weight.
        
    Returns:
        bool: True if valid.
        
    Raises:
        ValueError: If weights are severely malformed.
    """
    if not weights:
        raise ValueError("Portfolio holdings cannot be empty.")
    
    total_weight = sum(weights.values())
    
    # Check for severely malformed weights (e.g., negative or vastly off)
    if any(w < 0 for w in weights.values()):
        raise ValueError("Portfolio weights cannot be negative.")
    
    if not (0.95 <= total_weight <= 1.05):
        logger.warning(f"Portfolio weights sum to {total_weight}, which is outside the expected 1.0 range.")
        if not (0.5 <= total_weight <= 1.5): # Severe mismatch
            raise ValueError(f"Portfolio weights are severely malformed (sum={total_weight}).")
            
    return True

def get_portfolio_returns(
    portfolio_id: str,
    period: str = "6mo"
) -> pd.DataFrame:
    """
    Compute weighted portfolio returns using historical data.
    
    Args:
        portfolio_id: ID of the portfolio to compute.
        period: Historical data period.
        
    Returns:
        pd.DataFrame: Deterministic DataFrame with ticker weighted returns and portfolio_return.
    """
    # Step A: Validate Portfolio Exists
    if portfolio_id not in PORTFOLIOS:
        logger.error(f"Portfolio ID {portfolio_id} not found.")
        raise ValueError(f"Invalid portfolio_id: {portfolio_id}")
    
    portfolio_cfg = PORTFOLIOS[portfolio_id]
    weights = portfolio_cfg["holdings"]
    
    # Step B: Validate Weights
    validate_portfolio_weights(weights)
    
    # Step C: Fetch Historical Data
    tickers = list(weights.keys())
    data_dict = fetch_portfolio(tickers=tickers, period=period)
    
    if not data_dict:
        logger.warning(f"No data fetched for portfolio {portfolio_id}.")
        return pd.DataFrame()

    # Step D: Filter for successful fetches and normalize weights
    successful_tickers = [t for t in tickers if t in data_dict and not data_dict[t].empty]
    
    if not successful_tickers:
        logger.warning(f"No tickers successfully fetched for portfolio {portfolio_id}.")
        return pd.DataFrame()

    # Compute effective total weight for normalization
    effective_total = sum(weights[t] for t in successful_tickers)
    
    if effective_total < 1.0:
        logger.info(f"Normalizing weights for {portfolio_id}: effective total {effective_total:.4f} from {len(successful_tickers)} tickers.")

    # Step E: Compute Returns and Align
    returns_map = {}
    for ticker in successful_tickers:
        df = data_dict[ticker]
        if 'Close' in df.columns:
            # pct_change().dropna() avoids bias from undefined first-row returns
            # normalize the weight to ensure full portfolio exposure even if some tickers fail
            normalized_weight = weights[ticker] / effective_total
            ticker_return = df['Close'].pct_change().dropna() * normalized_weight
            returns_map[ticker] = ticker_return
        else:
            logger.warning(f"Missing 'Close' column for {ticker} in portfolio {portfolio_id}. Skipping.")

    if not returns_map:
        logger.warning(f"Could not compute returns for any tickers in portfolio {portfolio_id}.")
        return pd.DataFrame()

    # Create initial DataFrame from returns
    returns_df = pd.DataFrame(returns_map)
    
    # Step F: Alignment Handling (Index Intersection)
    # Using dropna() on the joined DataFrame effectively performs an inner join
    # which aligns all tickers to the dates where they all have data.
    initial_count = len(returns_df)
    returns_df = returns_df.dropna()
    final_count = len(returns_df)
    
    if final_count < initial_count:
        logger.info(f"Aligned {portfolio_id} returns: reduced from {initial_count} to {final_count} dates via intersection.")

    if returns_df.empty:
        logger.warning(f"Portfolio {portfolio_id} resulted in an empty DataFrame after date alignment.")
        return pd.DataFrame()

    # Step G: Aggregate Portfolio Return
    returns_df['portfolio_return'] = returns_df.sum(axis=1)
    
    # Step H: Schema Contract Enforcement
    # Ensure DatetimeIndex, float-only, single-level columns, no objects
    returns_df.index = pd.to_datetime(returns_df.index)
    
    # Force all columns to float to avoid silent object/int dtypes
    for col in returns_df.columns:
        returns_df[col] = returns_df[col].astype(float)
        
    return returns_df

def get_portfolio_metadata(portfolio_id: str) -> Dict[str, Any]:
    """
    Get metadata for a specific portfolio.
    
    Args:
        portfolio_id: ID of the portfolio.
        
    Returns:
        dict: Portfolio metadata including name, holdings, and sector exposure.
    """
    if portfolio_id not in PORTFOLIOS:
        logger.error(f"Portfolio ID {portfolio_id} not found.")
        raise ValueError(f"Invalid portfolio_id: {portfolio_id}")
        
    cfg = PORTFOLIOS[portfolio_id]
    holdings = cfg["holdings"]
    
    # Map tickers to sectors
    sectors = {}
    for ticker in holdings:
        sector = SECTOR_MAP.get(ticker, "Unknown")
        sectors[sector] = sectors.get(sector, 0.0) + holdings[ticker]
        
    return {
        "id": portfolio_id,
        "name": cfg["name"],
        "holdings": holdings,
        "sector_exposure": sectors,
        "holdings_count": len(holdings)
    }
