import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv
    from app.quant.regime_detector import detect_regimes_hmm, get_current_regime, REGIME_BULL, REGIME_BEAR, REQUIRED_REGIME_COLUMNS
    
    ticker = "^NSEI" # Nifty 50 Index
    print(f"--- Testing Regime Detector: {ticker} ---")
    
    # 1. Fetch real data
    print(f"Fetching 2y data for {ticker}...")
    df_raw = fetch_ohlcv(ticker, period="2y")
    
    if df_raw.empty:
        print("Failed to fetch raw data.")
        sys.exit(1)
        
    print(f"Raw data shape: {df_raw.shape}")
    
    # 2. Detect regimes
    print("\nDetecting regimes via HMM...")
    df_regimes = detect_regimes_hmm(df_raw.copy())
    
    if not df_regimes.empty:
        print(f"Success! Result shape: {df_regimes.shape}")
        
        # 3. Verify Deterministic Mapping
        print("\nVerifying Deterministic Mapping...")
        bull_mean = df_regimes[df_regimes["regime"] == REGIME_BULL]["log_return"].mean()
        bear_mean = df_regimes[df_regimes["regime"] == REGIME_BEAR]["log_return"].mean()
        
        print(f"Bull State Mean Return: {bull_mean:.6f}")
        print(f"Bear State Mean Return: {bear_mean:.6f}")
        
        if bull_mean > bear_mean:
            print("SUCCESS: Deterministic mapping confirmed (Bull > Bear).")
        else:
            print("FAILURE: Deterministic mapping failed! Bull return lower than Bear.")
            
        # 4. Verify Schema Contract
        missing = [col for col in REQUIRED_REGIME_COLUMNS if col not in df_regimes.columns]
        if not missing:
            print("SUCCESS: All REQUIRED_REGIME_COLUMNS present.")
        else:
            print(f"FAILURE: Missing columns: {missing}")
            
        # 5. Verify No NaNs
        nan_count = df_regimes.isna().sum().sum()
        if nan_count == 0:
            print("SUCCESS: No NaN values in output.")
        else:
            print(f"FAILURE: {nan_count} NaN values found!")

        # 6. Verify Current Regime & Duration
        status = get_current_regime(df_regimes)
        print("\n--- Current Status ---")
        print(f"Current Regime: {status['current_regime']}")
        print(f"Days in Regime: {status['days_in_regime']}")
        print(f"Regime History (Tail): {status['regime_history']}")
        
        if status['days_in_regime'] > 0:
            print("SUCCESS: Duration calculation confirmed.")
        else:
            print("FAILURE: Duration is 0.")

    else:
        print("Failed to detect regimes (DataFrame empty).")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
