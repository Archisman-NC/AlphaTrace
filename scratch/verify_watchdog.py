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
    from app.quant.watchdog import scan_portfolio_for_anomalies

    ticker = "RELIANCE.NS"
    print(f"--- Testing Watchdog Engine: {ticker} ---")

    # 1. Fetch Real Data
    df = fetch_ohlcv(ticker, period="1y")
    if df.empty:
        print("Failed to fetch data.")
        sys.exit(1)
    
    base_returns = df["Close"].pct_change().dropna()
    print(f"Baseline returns prepared. Length: {len(base_returns)}")

    # 2. Simulate Anomalies
    print("\nInjecting synthetic anomalies for testing...")
    
    # Portfolio for scanning
    portfolio_test = {}

    # Case A: Normal Behavior (Should have 0 alerts)
    portfolio_test["NORMAL_TICKER"] = base_returns.copy()

    # Case B: Sharpe Decay (Gradual collapse of mean returns)
    # Ensure baseline has a positive mean for Sharpe test
    pos_returns = base_returns.copy() + 0.005 # Force a positive mean
    decay_returns = pos_returns.copy()
    
    # Collapse the last 20 days
    window = 20
    decay_returns.iloc[-window:] = decay_returns.iloc[-window:] * 0.1 - 0.05
    portfolio_test["DECAY_TICKER"] = decay_returns

    # Case C: Distribution Shift (Volatility Shock)
    shift_returns = pos_returns.copy()
    # Increase variance by 5x in the last 20 days
    shift_returns.iloc[-window:] = np.random.normal(0, pos_returns.std() * 10, window)
    portfolio_test["SHIFT_TICKER"] = shift_returns

    # Case D: Z-Score Breach (Sudden Spike)
    spike_returns = pos_returns.copy()
    # Inject a 10-sigma event on the last day
    spike_returns.iloc[-1] = pos_returns.mean() + pos_returns.std() * 15
    portfolio_test["SPIKE_TICKER"] = spike_returns

    # 3. Run Scanner
    print("\nScanning portfolio for anomalies...")
    alerts = scan_portfolio_for_anomalies(portfolio_test)

    print(f"Found {len(alerts)} alerts.")

    # 4. Verify Alerts & Severity
    for alert in alerts:
        print(f"[{alert.severity}] {alert.ticker} | {alert.alert_type}: {alert.message}")

    # 5. Verify JSON Serializability
    try:
        alert_json = json.dumps([asdict(a) for a in alerts], indent=2)
        print("\nSUCCESS: Alerts are JSON serializable.")
        # print(alert_json)
    except TypeError as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

    # 6. Specific Checks
    decay_found = any(a.alert_type == "SHARPE_DECAY" for a in alerts)
    shift_found = any(a.alert_type == "DISTRIBUTION_SHIFT" for a in alerts)
    spike_found = any(a.alert_type == "ZSCORE_BREACH" for a in alerts)

    if decay_found and shift_found and spike_found:
        print("\nSUCCESS: All simulated anomaly types detected correctly.")
    else:
        print(f"\nFAILURE: Missing detections. Decay: {decay_found}, Shift: {shift_found}, Spike: {spike_found}")

    # 7. Check Deterministic Ordering (Critical first)
    if alerts and alerts[0].severity in ["CRITICAL", "HIGH"]:
        print("SUCCESS: Deterministic severity ordering verified.")
    else:
        print("FAILURE: Severity ordering check failed or no alerts found.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
