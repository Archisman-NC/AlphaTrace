import numpy as np
import pandas as pd
import logging
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Any, List

# Initialize logger
logger = logging.getLogger(__name__)

# Regime Constants
REGIME_BEAR = "Bear"
REGIME_SIDEWAYS = "Sideways"
REGIME_BULL = "Bull"

# Schema Contract
REQUIRED_REGIME_COLUMNS = [
    "Close", "log_return", "rolling_volatility", "regime", "regime_id"
]

@st.cache_data
def detect_regimes_hmm(
    df: pd.DataFrame,
    n_regimes: int = 3
) -> pd.DataFrame:
    """
    Detect market regimes using a Gaussian Hidden Markov Model.
    
    Args:
        df: Input DataFrame with 'Close' column and DatetimeIndex.
        n_regimes: Number of regimes to detect (default 3).
        
    Returns:
        pd.DataFrame: DataFrame with regime labels and features.
    """
    # 1. Input Validation
    if df.empty:
        logger.warning("Empty DataFrame provided to detect_regimes_hmm.")
        return pd.DataFrame(columns=REQUIRED_REGIME_COLUMNS)
    
    if "Close" not in df.columns:
        logger.error("Missing 'Close' column in input DataFrame.")
        return pd.DataFrame(columns=REQUIRED_REGIME_COLUMNS)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have a DatetimeIndex.")
        return pd.DataFrame(columns=REQUIRED_REGIME_COLUMNS)

    try:
        # 2. Feature Engineering
        # Log Returns (Ensuring we work on a copy to avoid SettingWithCopyWarning)
        df = df.copy()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        
        # Rolling Volatility (5-day standard deviation of log returns)
        df["rolling_volatility"] = df["log_return"].rolling(window=5).std()
        
        # 3. Alignment Logic
        # Drop NaNs created by shift and rolling operations
        df = df.dropna(subset=["log_return", "rolling_volatility"])
        
        if len(df) < 20: # Minimum data threshold for HMM
            logger.warning("Insufficient data points for HMM fitting.")
            return pd.DataFrame(columns=REQUIRED_REGIME_COLUMNS)

        # 4. HMM Model Configuration
        # Features for fitting: log_return and rolling_volatility
        X = df[["log_return", "rolling_volatility"]].values
        
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # 5. Model Fitting & Prediction
        model.fit(X)
        hidden_states = model.predict(X)
        df["regime_id"] = hidden_states

        # 6. Deterministic Label Mapping
        # Raw hidden state IDs are arbitrary. We map them based on mean returns.
        state_stats = []
        for i in range(n_regimes):
            state_data = df[df["regime_id"] == i]
            mean_ret = float(state_data["log_return"].mean()) if not state_data.empty else 0.0
            state_stats.append((i, mean_ret))
            
        # Sort states by mean return: Lowest (Bear) -> Middle (Sideways) -> Highest (Bull)
        # Note: This logic assumes n_regimes=3 as per requirement
        sorted_states = sorted(state_stats, key=lambda x: x[1])
        
        mapping = {
            sorted_states[0][0]: REGIME_BEAR,
            sorted_states[1][0]: REGIME_SIDEWAYS,
            sorted_states[2][0]: REGIME_BULL
        }
        
        df["regime"] = df["regime_id"].map(mapping)
        
        # Log Distribution
        dist = df["regime"].value_counts().to_dict()
        logger.info(f"Regime Distribution: {dist}")
        
        return df[REQUIRED_REGIME_COLUMNS].copy()

    except Exception as e:
        logger.error(f"Error in regime detection: {e}")
        return pd.DataFrame(columns=REQUIRED_REGIME_COLUMNS)

def get_current_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract current regime status and persistence from detected regimes.
    """
    if df.empty or "regime" not in df.columns:
        return {
            "current_regime": "Unknown",
            "days_in_regime": 0,
            "regime_history": []
        }
    
    # Current Regime
    current_regime = df["regime"].iloc[-1]
    
    # Calculate Consecutive Days (Persistence)
    # Compare each row with the one before it to find transitions
    regimes = df["regime"].values
    last_regime = regimes[-1]
    duration = 0
    for r in reversed(regimes):
        if r == last_regime:
            duration += 1
        else:
            break
            
    return {
        "current_regime": current_regime,
        "days_in_regime": int(duration),
        "regime_history": df["regime"].tail(10).tolist()
    }
