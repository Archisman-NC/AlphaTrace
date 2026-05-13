import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv
    from app.quant.signals import compute_signals
    from app.quant.regime_detector import detect_regimes_hmm
    from app.quant.backtester import run_backtest
    from app.quant.diagnostics import (
        explain_signals, 
        analyze_regime_performance, 
        detect_strategy_weaknesses,
        generate_strategy_diagnostics
    )
    
    ticker = "RELIANCE.NS"
    print(f"--- Testing Diagnostics Layer: {ticker} ---")
    
    # 1. Pipeline execution
    print("Executing full quant pipeline...")
    df_raw = fetch_ohlcv(ticker, period="2y")
    df_signals = compute_signals(df_raw)
    
    nifty_raw = fetch_ohlcv("^NSEI", period="2y")
    df_regimes = detect_regimes_hmm(nifty_raw)
    
    backtest_results = run_backtest(df_signals)
    eq_curve = backtest_results["equity_curve"]
    
    # 2. Explain Signals
    print("\nExplaining signals...")
    explanations = explain_signals(df_signals, df_regimes)
    if not explanations.empty:
        print(f"Success! {len(explanations)} explanations generated.")
        print(f"Sample explanation: {explanations['signal_reason'].iloc[10]}")
    else:
        print("FAILURE: Explanations empty.")
        
    # 3. Analyze Regime Performance
    print("\nAnalyzing regime performance...")
    regime_perf = analyze_regime_performance(eq_curve, df_regimes)
    if not regime_perf.empty:
        print("Success! Regime performance table generated.")
        print(regime_perf)
        
        # Verify Exposure Sums to ~1
        exposure_sum = regime_perf["exposure"].sum()
        print(f"Total Exposure Sum: {exposure_sum:.4f}")
        if abs(exposure_sum - 1.0) < 0.01:
            print("SUCCESS: Exposure calculation verified.")
        else:
            print("FAILURE: Exposure does not sum to 1.0")
    else:
        print("FAILURE: Regime performance empty.")
        
    # 4. Generate Full Diagnostics
    print("\nGenerating full diagnostics...")
    diagnostics = generate_strategy_diagnostics(backtest_results, regime_perf, explanations)
    
    print(f"Summary: {diagnostics['summary']}")
    print(f"Latest Signal: {diagnostics['latest_signal_explanation']}")
    print(f"Strengths: {diagnostics['strengths']}")
    print(f"Weaknesses: {diagnostics['weaknesses']}")
    
    # 5. Verify JSON Serializability
    import json
    try:
        json.dumps(diagnostics)
        print("\nSUCCESS: Diagnostics fully JSON serializable.")
    except Exception as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
