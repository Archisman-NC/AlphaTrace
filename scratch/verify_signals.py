import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv
    from app.quant.signals import compute_signals, REQUIRED_SIGNAL_COLUMNS
    
    ticker = "RELIANCE.NS"
    print(f"--- Testing Signal Engine: {ticker} ---")
    
    # 1. Fetch real data
    print(f"Fetching 1y data for {ticker}...")
    df_raw = fetch_ohlcv(ticker, period="1y")
    
    if df_raw.empty:
        print("Failed to fetch raw data.")
        sys.exit(1)
        
    print(f"Raw data shape: {df_raw.shape}")
    
    # 2. Compute signals
    print("\nComputing signals...")
    df_signals = compute_signals(df_raw.copy())
    
    if not df_signals.empty:
        print(f"Success! Result shape: {df_signals.shape}")
        
        # 3. Verify Schema Contract
        print("\nVerifying Schema Contract...")
        missing = [col for col in REQUIRED_SIGNAL_COLUMNS if col not in df_signals.columns]
        if not missing:
            print("SUCCESS: All REQUIRED_SIGNAL_COLUMNS present.")
        else:
            print(f"FAILURE: Missing columns: {missing}")
            
        # 4. Verify No NaNs
        nan_count = df_signals.isna().sum().sum()
        if nan_count == 0:
            print("SUCCESS: No NaN values in output.")
        else:
            print(f"FAILURE: {nan_count} NaN values found!")
            
        # 5. Verify Signal Values
        signals = df_signals["signal"].unique()
        print(f"Unique signals found: {signals}")
        invalid_signals = [s for s in signals if s not in [-1, 0, 1]]
        if not invalid_signals:
            print("SUCCESS: Signal values are restricted to {-1, 0, 1}.")
        else:
            print(f"FAILURE: Invalid signals found: {invalid_signals}")
            
        # 6. Verify Dtypes
        if df_signals["signal"].dtype in ['int64', 'int32']:
            print("SUCCESS: 'signal' column is integer type.")
        else:
            print(f"FAILURE: 'signal' column type is {df_signals['signal'].dtype}")
            
        print("\nFirst 5 rows of signals and indicators:\n", df_signals.head(5))
        
        # 7. Verify OHLCV preservation
        ohlcv = ["Open", "High", "Low", "Close", "Volume"]
        if all(c in df_signals.columns for c in ohlcv):
            print("SUCCESS: OHLCV columns preserved.")
        else:
            print("FAILURE: OHLCV columns lost!")

    else:
        print("Failed to compute signals (DataFrame empty).")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
