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
        
        # 4. Verify Lookahead Bias Prevention (Transition-Based Validation)
        # Requirement: A NEW position must NOT open on the SAME BAR that generated the signal.
        # Methodology: 
        #   1. Compute position_change = position.diff()
        #   2. Confirm that position_change at T depends on signal at T-1, not T.
        
        # Align signals with backtest timeframe (including 1-day lookback for transition check)
        expected_position_full = df_signals["signal"].shift(1).fillna(0)
        expected_position = expected_position_full.loc[equity_df.index]
        
        # Calculate Position Change
        equity_df["pos_change"] = equity_df["position"].diff().fillna(0)
        
        print("\nVerifying Lookahead Bias Prevention...")
        
        # Rigorous check: Position at T must exactly match Signal at T-1
        if (equity_df["position"] == expected_position).all():
            print("SUCCESS: No lookahead bias detected. Execution alignment confirmed (Position_T = Signal_T-1).")
        else:
            mismatches = equity_df[equity_df["position"] != expected_position]
            print(f"FAILURE: Potential lookahead leakage found at {len(mismatches)} points.")
            # Show first mismatch
            first_m = mismatches.index[0]
            print(f"Mismatch at {first_m.date()}: Signal={equity_df.loc[first_m, 'signal']}, Position={equity_df.loc[first_m, 'position']}")

        # 5. Explicit Temporal Alignment Diagnostics
        print("\nTemporal Alignment Diagnostics (Detailed View):")
        # Find first non-zero signal and its subsequent execution
        signal_days = equity_df[equity_df["signal"] != 0].index
        if not signal_days.empty:
            first_sig_idx = signal_days[0]
            sig_loc = equity_df.index.get_loc(first_sig_idx)
            
            # Show a 5-day window starting from signal
            start_loc = max(0, sig_loc - 1)
            end_loc = min(len(equity_df), sig_loc + 4)
            diag_window = equity_df.iloc[start_loc:end_loc].copy()
            
            print(f"{'Date':<12} | {'Signal':<6} | {'Position':<8} | {'Change':<6} | {'Asset Ret':<10} | {'Strat Ret':<10}")
            print("-" * 70)
            for idx, row in diag_window.iterrows():
                print(f"{str(idx.date()):<12} | {int(row['signal']):>6} | {row['position']:>8.1f} | {row['pos_change']:>6.1f} | {row['daily_return']:>10.4%} | {row['strategy_return']:>10.4%}")
            
            # Mathematical Confirmation of Signal(T) -> Position(T+1)
            t_idx = diag_window.index[1] # The signal day
            t_plus_1_idx = diag_window.index[2] # The execution day
            
            sig_t = diag_window.loc[t_idx, "signal"]
            pos_t_plus_1 = diag_window.loc[t_plus_1_idx, "position"]
            
            if sig_t == pos_t_plus_1:
                print(f"\nSUCCESS: Signal at {t_idx.date()} ({sig_t}) transitioned to Position at {t_plus_1_idx.date()} ({pos_t_plus_1}).")
            else:
                print(f"\nFAILURE: Alignment break! Signal at {t_idx.date()} does not match Position at {t_plus_1_idx.date()}.")
        
        # 6. Verify Compounding & Return Alignment
        print("\nVerifying Return Alignment & Compounding:")
        # Requirement: R_t = position_t * r_t
        sample_row = equity_df.iloc[-1]
        calculated_strat_ret = sample_row["position"] * sample_row["daily_return"]
        
        if abs(calculated_strat_ret - sample_row["strategy_return"]) < 1e-7:
            print("SUCCESS: Return alignment verified (R_t = pos_t * r_t).")
        else:
            print(f"FAILURE: Return alignment mismatch! Expected {calculated_strat_ret}, got {sample_row['strategy_return']}")

        # Manual Compounding check
        prev_equity = equity_df.iloc[-2]["equity_curve"]
        curr_equity = equity_df.iloc[-1]["equity_curve"]
        expected_equity = prev_equity * (1 + sample_row["strategy_return"])
        
        if abs(expected_equity - curr_equity) < 1e-6:
            print("SUCCESS: Compounding math verified.")
        else:
            print(f"FAILURE: Compounding math mismatch!")

        # 7. Verify Schema
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
