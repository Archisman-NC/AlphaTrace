import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.portfolio_builder import PORTFOLIOS
    from app.reasoning.proactive_engine import get_watchdog_insights
    from app.data.market_data import fetch_ohlcv

    print("--- Verifying Watchdog Dashboard Data Structures ---")

    # 1. Verify Portfolio Access
    portfolio_id = "PORTFOLIO_001"
    portfolio_data = PORTFOLIOS.get(portfolio_id, {})
    tickers = list(portfolio_data.get("holdings", {}).keys())
    print(f"Portfolio {portfolio_id} tickers: {tickers}")

    # 2. Verify Return Fetching Logic
    # Testing with a single ticker to ensure OHLCV -> Returns pipeline is stable
    test_ticker = tickers[0]
    print(f"Fetching returns for {test_ticker}...")
    raw_df = fetch_ohlcv(test_ticker, period="2y")
    if not raw_df.empty:
        returns = raw_df["Close"].pct_change().dropna()
        print(f"SUCCESS: Fetched {len(returns)} days of returns for {test_ticker}")
    else:
        print(f"FAILURE: Could not fetch returns for {test_ticker}")

    # 3. Verify Watchdog Analysis Pipeline
    print("\nRunning simulated watchdog analysis...")
    # Setup mock portfolio returns
    mock_returns = {
        "STRAT_A": pd.Series(np.random.normal(0.0005, 0.01, 200)),
        "STRAT_B": pd.Series(np.random.normal(0.0005, 0.01, 200))
    }
    # Inject anomaly in STRAT_B
    mock_returns["STRAT_B"].iloc[-1] = 0.5 # Extreme spike
    
    insights = get_watchdog_insights(mock_returns)
    print(f"Analysis Status: {insights['status']}")
    print(f"Number of Alerts: {len(insights['critical_alerts']) + len(insights['high_alerts'])}")

    # 4. Verify Chart Data Preparation
    print("\nVerifying Chart Data Structures...")
    if not raw_df.empty:
        returns = raw_df["Close"].pct_change().dropna()
        rolling_sharpe = (returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
        print(f"Rolling Sharpe computed. Samples: {len(rolling_sharpe.dropna())}")
        
        # Verify baseline calculation
        baseline = (returns.mean() / returns.std()) * np.sqrt(252)
        print(f"Trailing Baseline Sharpe: {baseline:.2f}")
        
        if not np.isnan(baseline):
            print("SUCCESS: Chart data preparation verified.")
        else:
            print("FAILURE: Baseline Sharpe is NaN.")

    # 5. Verify Session State Schema
    # This simulates what Streamlit sees in session_state.watchdog_results
    expected_keys = ["status", "critical_alerts", "high_alerts", "medium_alerts", "low_alerts", "summary", "top_risk", "suggested_actions", "timestamp"]
    missing_keys = [k for k in expected_keys if k not in insights]
    
    if not missing_keys:
        print("\nSUCCESS: Watchdog result schema is stable.")
    else:
        print(f"\nFAILURE: Missing schema keys: {missing_keys}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
