import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Initialize logger
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252

# Schema Contract
REQUIRED_BACKTEST_COLUMNS = [
    "Close", "signal", "position", "daily_return", 
    "strategy_return", "equity_curve", "buy_hold_curve"
]

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Run a vectorized backtest on a signal-enriched DataFrame.
    
    Args:
        df: Input DataFrame containing at least 'Close' and 'signal'.
        initial_capital: Starting capital for the backtest.
        
    Returns:
        Dict: Dictionary containing performance metrics and the equity curve.
    """
    # 1. Input Validation
    if df.empty:
        logger.warning("Empty DataFrame provided to run_backtest.")
        return get_fallback_results()
    
    required = ["Close", "signal"]
    if not all(col in df.columns for col in required):
        logger.error(f"Missing required columns for backtest: {[c for c in required if c not in df.columns]}")
        return get_fallback_results()
        
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Backtest input must have a DatetimeIndex.")
        return get_fallback_results()

    try:
        # 2. Position Logic (Lookahead Bias Prevention)
        # We enter the position on the NEXT bar after the signal is generated
        df["position"] = df["signal"].shift(1).fillna(0)
        
        # 3. Return Calculations
        # Asset daily return
        df["daily_return"] = df["Close"].pct_change()
        
        # Drop the first row as returns are undefined (NaN)
        df = df.dropna(subset=["daily_return"])
        
        # Strategy daily return: position * asset_return
        df["strategy_return"] = df["position"] * df["daily_return"]
        
        # 4. Equity Curve Logic (Compounded)
        # Strategy Equity
        df["equity_curve"] = initial_capital * (1 + df["strategy_return"]).cumprod()
        
        # Buy & Hold Benchmark
        df["buy_hold_curve"] = initial_capital * (1 + df["daily_return"]).cumprod()
        
        # 5. Performance Metrics
        final_strategy_equity = float(df["equity_curve"].iloc[-1])
        final_bh_equity = float(df["buy_hold_curve"].iloc[-1])
        
        total_return = (final_strategy_equity / initial_capital) - 1
        buy_hold_return = (final_bh_equity / initial_capital) - 1
        
        # Sharpe Ratio (Annualized)
        std_dev = df["strategy_return"].std()
        avg_return = df["strategy_return"].mean()
        sharpe_ratio = 0.0
        if std_dev > 0:
            sharpe_ratio = float((avg_return / std_dev) * np.sqrt(TRADING_DAYS_PER_YEAR))
            
        # Max Drawdown
        rolling_max = df["equity_curve"].cummax()
        drawdown = (df["equity_curve"] - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        
        # Win Rate (Only on active days)
        active_days = df[df["position"] != 0]
        win_rate = 0.0
        if not active_days.empty:
            wins = (active_days["strategy_return"] > 0).sum()
            win_rate = float(wins / len(active_days))
            
        # Exposure Ratio
        exposure_ratio = float((df["position"] != 0).sum() / len(df))
        
        # Trade Count (Changes in position state)
        # We count any non-zero transition as an execution event
        num_trades = int((df["position"].diff().fillna(0) != 0).sum())

        metrics = {
            "total_return": float(total_return),
            "buy_hold_return": float(buy_hold_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "exposure_ratio": float(exposure_ratio),
            "num_trades": num_trades,
            "status": "success"
        }
        
        # Log Summary
        logger.info(f"Backtest Summary - Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, MaxDD: {max_drawdown:.2%}")
        
        return {
            "metrics": metrics,
            "equity_curve": df[REQUIRED_BACKTEST_COLUMNS].copy()
        }

    except Exception as e:
        logger.error(f"Error during backtest execution: {e}")
        return get_fallback_results()

def get_fallback_results() -> Dict[str, Any]:
    """Safe fallback for failed backtests."""
    return {
        "metrics": {
            "total_return": 0.0,
            "buy_hold_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "exposure_ratio": 0.0,
            "num_trades": 0,
            "status": "error"
        },
        "equity_curve": pd.DataFrame(columns=REQUIRED_BACKTEST_COLUMNS)
    }
