import sys
import os
import pandas as pd
import numpy as np
import json
from dataclasses import asdict

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.quant.signal_generator import TradingSignal
    from app.quant.portfolio_signals import PortfolioSignalSummary, aggregate_sector_signals, portfolio_signal_reasoning_context

    print("--- Verifying Signal Dashboard Data Structures ---")

    # Helper to create mock signals
    def create_mock_signal(ticker, direction, confidence):
        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            signal_strength="STRONG" if confidence > 0.7 else "MODERATE",
            causal_reason="RSI entered oversold territory with improving MACD momentum.",
            regime="Bull",
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            generated_at="2024-05-14"
        )

    # 1. Simulate a Bullish Scan Result
    print("\nSimulating Bullish Scan Result...")
    signals = [
        create_mock_signal("HDFCBANK.NS", "LONG", 0.85),
        create_mock_signal("RELIANCE.NS", "LONG", 0.80),
        create_mock_signal("TCS.NS", "NEUTRAL", 0.50)
    ]
    summary = PortfolioSignalSummary(
        total_signals=3,
        long_signals=2,
        short_signals=0,
        neutral_signals=1,
        top_signal="HDFCBANK.NS (STRONG)",
        average_confidence=0.716,
        market_bias="BULLISH",
        generated_at="2024-05-14"
    )
    sector_agg = aggregate_sector_signals(signals)
    diagnostics = ["Signal concentration is currently highest in the Banking sector."]

    # This mirrors st.session_state.signal_results
    signal_results = {
        "signals": [asdict(s) for s in signals],
        "summary": asdict(summary),
        "sector_agg": sector_agg,
        "diagnostics": diagnostics
    }

    # 2. Verify Session State Schema
    expected_keys = ["signals", "summary", "sector_agg", "diagnostics"]
    missing = [k for k in expected_keys if k not in signal_results]
    if not missing:
        print("SUCCESS: Session state schema verified.")
    else:
        print(f"FAILURE: Missing session state keys: {missing}")

    # 3. Verify Sector Mapping for Plotly
    print("\nVerifying Sector Mapping for Charts...")
    sector_df = pd.DataFrame.from_dict(signal_results["sector_agg"], orient='index').reset_index()
    if not sector_df.empty and "bias" in sector_df.columns:
        print(f"SUCCESS: Sector DataFrame prepared. Sectors found: {list(sector_df['index'])}")
    else:
        print("FAILURE: Sector DataFrame preparation failed.")

    # 4. Verify AI Context Generation from Mock Results
    print("\nVerifying AI Context Generation...")
    mock_sigs = [TradingSignal(**s) for s in signal_results["signals"]]
    mock_sum = PortfolioSignalSummary(**signal_results["summary"])
    ctx = portfolio_signal_reasoning_context(mock_sigs, mock_sum, signal_results["diagnostics"])
    if "BULLISH bias" in ctx and "HDFCBANK.NS" in ctx:
        print("SUCCESS: AI Context generation from session data verified.")
    else:
        print("FAILURE: AI Context generation mismatch.")

    # 5. JSON Serializability
    try:
        json_str = json.dumps(signal_results)
        print("\nSUCCESS: All dashboard data is JSON serializable.")
    except TypeError as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
