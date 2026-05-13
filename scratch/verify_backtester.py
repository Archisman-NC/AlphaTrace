import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv
    from app.quant.signals import compute_signals
    from app.quant.backtester import run_backtest, REQUIRED_BACKTEST_COLUMNS
    
    ticker = "RELIANCE.NS"
    print(f"--- Testing Backtester Engine: {ticker} ---")
    
    # 1. Fetch and Signal
    print(f"Generating signals for {ticker}...")
    df_raw = fetch_ohlcv(ticker, period="1y")
    df_signals = compute_signals(df_raw)
    
    # 2. Run Backtest
    print("\nRunning backtest...")
    initial_capital = 100000.0
    results = run_backtest(df_signals, initial_capital=initial_capital)
    
    if results["metrics"]["status"] == "success":
        metrics = results["metrics"]
        equity_df = results["equity_curve"]
        
        print(f"Success! Final Equity: {equity_df['equity_curve'].iloc[-1]:.2f}")
        
        # 3. Verify Metrics
        print("\nVerifying Metrics:")
        print(f"- Total Return: {metrics['total_return']:.2%}")
        print(f"- Buy & Hold: {metrics['buy_hold_return']:.2%}")
        print(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"- Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"- Win Rate: {metrics['win_rate']:.2%}")
        print(f"- Exposure Ratio: {metrics['exposure_ratio']:.2%}")
        print(f"- Trades: {metrics['num_trades']}")
        
        # 4. Verify Lookahead Bias Prevention
        # Check that on the first signal day in the results, position is 0
        valid_signals = df_signals.loc[equity_df.index]
        first_signal_idx = valid_signals[valid_signals["signal"] != 0].index[0]
        pos_on_signal_day = equity_df.loc[first_signal_idx, "position"]
        if pos_on_signal_day == 0:
            print("\nSUCCESS: Lookahead bias prevention verified (pos=0 on first signal day).")
        else:
            print("\nFAILURE: Potential lookahead bias! Position active on signal day.")
            
        # 5. Verify Compounding
        # Manual check: (1 + last_strategy_return) * prev_equity
        last_row = equity_df.iloc[-1]
        prev_row = equity_df.iloc[-2]
        expected_equity = prev_row["equity_curve"] * (1 + last_row["strategy_return"])
        if abs(expected_equity - last_row["equity_curve"]) < 1e-6:
            print("SUCCESS: Compounding math verified.")
        else:
            print(f"FAILURE: Compounding math mismatch! Expected {expected_equity}, got {last_row['equity_curve']}")
            
        # 6. Verify Schema
        missing = [col for col in REQUIRED_BACKTEST_COLUMNS if col not in equity_df.columns]
        if not missing:
            print("SUCCESS: All REQUIRED_BACKTEST_COLUMNS present.")
        else:
            print(f"FAILURE: Missing columns: {missing}")

    else:
        print("Backtest failed.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
