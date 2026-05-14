import sys
import os
import pandas as pd
import numpy as np
import json
from dataclasses import asdict

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv
    from app.quant.signals import compute_signals
    from app.quant.signal_generator import generate_trading_signal, generate_portfolio_signals

    ticker = "RELIANCE.NS"
    print(f"--- Testing Signal Generator: {ticker} ---")

    # 1. Fetch and Compute Indicators
    df_raw = fetch_ohlcv(ticker, period="1y")
    df_indicators = compute_signals(df_raw)
    print(f"Base indicators computed. Data length: {len(df_indicators)}")

    # 2. Test Case A: Real Market Signal
    print("\nGenerating signal for current market state...")
    current_signal = generate_trading_signal(df_indicators, regime="Bull", ticker=ticker)
    print(f"Ticker: {current_signal.ticker}")
    print(f"Direction: {current_signal.direction}")
    print(f"Confidence: {current_signal.confidence:.2f}")
    print(f"Reason: {current_signal.causal_reason}")

    # 3. Test Case B: Simulated LONG Signal (Oversold)
    print("\nSimulating Oversold (LONG) condition...")
    oversold_df = df_indicators.copy()
    # Force indicators for LONG: RSI < 35, Close near bb_lower, MACD improving
    last_idx = oversold_df.index[-1]
    prev_idx = oversold_df.index[-2]
    
    oversold_df.at[last_idx, "rsi"] = 25
    oversold_df.at[last_idx, "bb_lower"] = 2500
    oversold_df.at[last_idx, "bb_upper"] = 2600
    oversold_df.at[last_idx, "Close"] = 2510 # %B = 0.1
    oversold_df.at[last_idx, "macd"] = 5
    oversold_df.at[last_idx, "macd_signal"] = 0 # improving
    oversold_df.at[prev_idx, "macd"] = 0
    oversold_df.at[prev_idx, "macd_signal"] = 0
    
    long_signal = generate_trading_signal(oversold_df, ticker="LONG_TEST")
    print(f"Direction: {long_signal.direction} (Expected: LONG)")
    print(f"Stop Loss: {long_signal.stop_loss:.2f}")
    print(f"Take Profit: {long_signal.take_profit:.2f}")
    
    if long_signal.direction == "LONG":
        print("SUCCESS: LONG signal correctly generated.")
    else:
        print("FAILURE: LONG signal logic mismatch.")

    # 4. Test Case C: Simulated SHORT Signal (Overbought)
    print("\nSimulating Overbought (SHORT) condition...")
    overbought_df = df_indicators.copy()
    overbought_df.at[last_idx, "rsi"] = 75
    overbought_df.at[last_idx, "bb_lower"] = 2500
    overbought_df.at[last_idx, "bb_upper"] = 2600
    overbought_df.at[last_idx, "Close"] = 2590 # %B = 0.9
    overbought_df.at[last_idx, "macd"] = -5
    overbought_df.at[last_idx, "macd_signal"] = 0 # weakening
    overbought_df.at[prev_idx, "macd"] = 0
    overbought_df.at[prev_idx, "macd_signal"] = 0
    
    short_signal = generate_trading_signal(overbought_df, ticker="SHORT_TEST")
    print(f"Direction: {short_signal.direction} (Expected: SHORT)")
    
    if short_signal.direction == "SHORT":
        print("SUCCESS: SHORT signal correctly generated.")
    else:
        print("FAILURE: SHORT signal logic mismatch.")

    # 5. Portfolio Signal Aggregation
    print("\nTesting portfolio-wide signal generation...")
    portfolio_data = {
        "RELIANCE.NS": df_indicators,
        "LONG_STOCK": oversold_df,
        "SHORT_STOCK": overbought_df
    }
    portfolio_signals = generate_portfolio_signals(portfolio_data)
    print(f"Total signals generated: {len(portfolio_signals)}")
    for s in portfolio_signals:
        print(f"- {s.ticker}: {s.direction} ({s.confidence:.2f})")

    # 6. Verify JSON Serializability
    try:
        sig_json = json.dumps([asdict(s) for s in portfolio_signals], indent=2)
        print("\nSUCCESS: Signals are JSON serializable.")
    except TypeError as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
