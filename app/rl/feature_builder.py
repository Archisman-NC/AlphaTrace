import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw OHLCV and technical indicators into RL-ready features.
    
    Expected columns in input df:
    - Close, Volume
    - rsi, macd, macd_signal
    - bb_upper, bb_lower
    """
    if df.empty:
        return pd.DataFrame()

    try:
        features = pd.DataFrame(index=df.index)
        
        # 1. Price Momentum & Returns
        # Use log returns for better statistical properties
        features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        features["volatility_20d"] = features["log_return"].rolling(20).std()
        
        # 2. RSI (already bounded 0-100, normalize to 0-1)
        features["rsi_norm"] = df["rsi"] / 100.0
        
        # 3. MACD Normalized by Price
        features["macd_norm"] = (df["macd"] - df["macd_signal"]) / df["Close"]
        
        # 4. Bollinger Position (%B)
        bb_range = df["bb_upper"] - df["bb_lower"]
        features["bb_pct"] = (df["Close"] - df["bb_lower"]) / bb_range
        # Handle division by zero or NaN BB
        features["bb_pct"] = features["bb_pct"].fillna(0.5).clip(-1, 2)
        
        # 5. Volume Scaling (log volume change)
        features["log_vol_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
        
        # Clean up NaNs and Infs from shifting/rolling/log
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features

    except Exception as e:
        logger.error(f"Failed to build RL features: {e}")
        return pd.DataFrame()
