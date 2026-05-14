import sys
import os
import pandas as pd
import numpy as np
import json
from dataclasses import asdict

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..'))))

try:
    from app.quant.signal_generator import TradingSignal
    from app.quant.portfolio_signals import (
        scan_portfolio_signals,
        generate_portfolio_signal_summary,
        aggregate_sector_signals,
        generate_signal_diagnostics,
        portfolio_signal_reasoning_context
    )

    print("--- Testing Portfolio Signal Intelligence ---")

    # Helper to generate mock signals
    def create_mock_signal(ticker, direction, confidence, generated_at="2024-05-14"):
        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            signal_strength="STRONG" if confidence > 0.7 else "MODERATE",
            causal_reason="Mock reason",
            regime="Bull",
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            generated_at=generated_at
        )

    # 1. Test Case A: Bullish Portfolio
    print("\n[Case A] Simulating Bullish Portfolio...")
    bull_signals = [
        create_mock_signal("HDFCBANK.NS", "LONG", 0.85),
        create_mock_signal("ICICIBANK.NS", "LONG", 0.80),
        create_mock_signal("RELIANCE.NS", "LONG", 0.75),
        create_mock_signal("TCS.NS", "NEUTRAL", 0.50)
    ]
    bull_summary = generate_portfolio_signal_summary(bull_signals)
    print(f"Bias: {bull_summary.market_bias} (Expected: BULLISH)")
    print(f"Top Signal: {bull_summary.top_signal}")

    # 2. Test Case B: Bearish Portfolio
    print("\n[Case B] Simulating Bearish Portfolio...")
    bear_signals = [
        create_mock_signal("INFY.NS", "SHORT", 0.90),
        create_mock_signal("TCS.NS", "SHORT", 0.80),
        create_mock_signal("SBIN.NS", "SHORT", 0.70),
        create_mock_signal("LT.NS", "NEUTRAL", 0.50)
    ]
    bear_summary = generate_portfolio_signal_summary(bear_signals)
    print(f"Bias: {bear_summary.market_bias} (Expected: BEARISH)")

    # 3. Test Case C: Mixed Portfolio (Consensus Conflict)
    print("\n[Case C] Simulating Mixed Portfolio...")
    mixed_signals = [
        create_mock_signal("HDFCBANK.NS", "LONG", 0.90),
        create_mock_signal("INFY.NS", "SHORT", 0.85),
        create_mock_signal("RELIANCE.NS", "LONG", 0.40),
        create_mock_signal("ITC.NS", "SHORT", 0.35)
    ]
    mixed_summary = generate_portfolio_signal_summary(mixed_signals)
    print(f"Bias: {mixed_summary.market_bias} (Expected: MIXED)")

    # 4. Sector Aggregation & Clustering
    print("\nVerifying Sector Aggregation...")
    sector_agg = aggregate_sector_signals(bull_signals)
    for sector, data in sector_agg.items():
        print(f"Sector: {sector} | Bias: {data['bias']} | Avg Conf: {data['avg_confidence']:.2f}")

    # 5. Diagnostics & Reasoning Context
    print("\nGenerating Diagnostics...")
    diagnostics = generate_signal_diagnostics(bull_signals, bull_summary, sector_agg)
    for d in diagnostics:
        print(f"- {d}")

    print("\nAI Reasoning Context:")
    ctx = portfolio_signal_reasoning_context(bull_signals, bull_summary, diagnostics)
    print(ctx)

    # 6. Verify Deterministic Ranking (scan_portfolio_signals)
    print("\nVerifying Deterministic Ranking...")
    # Mock data
    dates = pd.date_range("2024-01-01", periods=10)
    mock_df = pd.DataFrame({"Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1000}, index=dates)
    # Add indicators for LONG signal (RSI < 35, etc. - based on signals.py)
    mock_df["rsi"] = 30
    mock_df["macd"] = 5
    mock_df["macd_signal"] = 0
    mock_df["bb_upper"] = 110
    mock_df["bb_lower"] = 100
    mock_df["signal"] = 1
    
    ticker_data = {
        "TCS.NS": mock_df,
        "INFY.NS": mock_df,
        "RELIANCE.NS": mock_df
    }
    
    ranked_signals = scan_portfolio_signals(ticker_data)
    print("Ranked Tickers (Alphabetical tie-break expected if conf identical):")
    for s in ranked_signals:
        print(f"- {s.ticker} ({s.confidence:.2f})")

    # 7. JSON Serializability
    try:
        json_summary = json.dumps(asdict(bull_summary))
        print("\nSUCCESS: Portfolio summary is JSON serializable.")
    except Exception as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
