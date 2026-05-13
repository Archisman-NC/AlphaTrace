import pandas as pd
import pandas_ta as ta
import logging
from typing import List

# Initialize logger
logger = logging.getLogger(__name__)

# Signal Schema Constants
SIGNAL_BUY = 1
SIGNAL_HOLD = 0
SIGNAL_SELL = -1

# Schema Contract
REQUIRED_SIGNAL_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
    "signal"
]

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and trading signals for a ticker.
    
    Args:
        df: Input DataFrame with OHLCV data and DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with indicators and signals, ensuring deterministic schema.
    """
    # 1. Input Validation
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_signals.")
        return pd.DataFrame(columns=REQUIRED_SIGNAL_COLUMNS)
    
    if "Close" not in df.columns:
        logger.error("Missing 'Close' column in input DataFrame.")
        return pd.DataFrame(columns=REQUIRED_SIGNAL_COLUMNS)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have a DatetimeIndex.")
        return pd.DataFrame(columns=REQUIRED_SIGNAL_COLUMNS)

    # Ensure numeric Close (Using copy to avoid SettingWithCopyWarning)
    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df = df.dropna(subset=["Close"])

    # 2. Indicator Calculations
    try:
        # RSI(14)
        df["rsi"] = ta.rsi(df["Close"], length=14)
        
        # MACD
        macd_df = ta.macd(df["Close"])
        # pandas_ta returns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        if macd_df is not None:
            df["macd"] = macd_df.iloc[:, 0] # MACD line
            df["macd_signal"] = macd_df.iloc[:, 2] # Signal line
            
        # Bollinger Bands(20, 2)
        bb_df = ta.bbands(df["Close"], length=20, std=2)
        # returns BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        if bb_df is not None:
            df["bb_lower"] = bb_df.iloc[:, 0]
            df["bb_upper"] = bb_df.iloc[:, 2]

    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        return pd.DataFrame(columns=REQUIRED_SIGNAL_COLUMNS)

    # 3. Signal Generation Logic (RSI Mean-Reversion)
    df["signal"] = SIGNAL_HOLD
    
    # RSI < 30 -> Buy
    df.loc[df["rsi"] < 30, "signal"] = SIGNAL_BUY
    
    # RSI > 70 -> Sell
    df.loc[df["rsi"] > 70, "signal"] = SIGNAL_SELL

    # 4. Signal Distribution Logging
    buy_count = (df["signal"] == SIGNAL_BUY).sum()
    sell_count = (df["signal"] == SIGNAL_SELL).sum()
    hold_count = (df["signal"] == SIGNAL_HOLD).sum()
    
    logger.info(f"Signal Distribution - Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")

    # 5. Deterministic Output Requirements
    # Remove rows with NaN indicators (ensures valid backtesting data)
    df = df.dropna()
    
    # Ensure all required columns exist and are in the correct order
    available_cols = [col for col in REQUIRED_SIGNAL_COLUMNS if col in df.columns]
    
    # Cast dtypes
    df["signal"] = df["signal"].astype(int)
    for col in available_cols:
        if col != "signal":
            df[col] = df[col].astype(float)
            
    return df[available_cols]
