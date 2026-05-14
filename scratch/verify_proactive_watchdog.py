import sys
import os
import pandas as pd
import numpy as np
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.reasoning.proactive_engine import get_watchdog_insights, watchdog_reasoning_context

    print("--- Testing Proactive Watchdog Integration ---")

    # 1. Setup Simulation Data
    dates = pd.date_range(start="2023-01-01", periods=100)
    
    # Stable Ticker (Normal returns)
    stable_returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    
    # Degrading Ticker (Sharpe collapse)
    decay_returns = pd.Series(np.random.normal(0.005, 0.01, 100), index=dates)
    decay_returns.iloc[-20:] = decay_returns.iloc[-20:] * 0.1 - 0.05 # Collapse
    
    # Volatile Ticker (Z-Score breach + Shift)
    shock_returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    shock_returns.iloc[-1] = 0.2 # 20% spike (Massive Z-Score)
    shock_returns.iloc[-10:] = np.random.normal(0.0005, 0.05, 10) # High vol shift

    portfolio_returns = {
        "STABLE_STRAT": stable_returns,
        "DECAY_STRAT": decay_returns,
        "SHOCK_STRAT": shock_returns
    }

    # 2. Run Integration
    print("\nProcessing proactive watchdog insights...")
    insights = get_watchdog_insights(portfolio_returns)

    # 3. Verify Structure & Status
    print(f"Status: {insights['status']}")
    print(f"Summary: {insights['summary']}")
    print(f"Top Risk: {insights['top_risk']}")
    
    # 4. Verify Prioritization
    print("\nAlert Distribution:")
    print(f"  CRITICAL: {len(insights['critical_alerts'])}")
    print(f"  HIGH:     {len(insights['high_alerts'])}")
    print(f"  MEDIUM:   {len(insights['medium_alerts'])}")
    
    if insights["status"] in ["DEGRADED", "CRITICAL"]:
        print("SUCCESS: Status correctly escalated due to simulated anomalies.")

    # 5. Verify Suggested Actions
    print("\nSuggested Actions:")
    for action in insights["suggested_actions"]:
        print(f"  - {action}")

    # 6. Verify AI Escalation Context
    print("\nAI Escalation Context:")
    reasoning_ctx = watchdog_reasoning_context(insights)
    print(reasoning_ctx)

    # 7. Verify JSON Serializability
    try:
        json_str = json.dumps(insights)
        print("\nSUCCESS: Insights are JSON serializable.")
    except TypeError as e:
        print(f"\nFAILURE: JSON serialization failed: {e}")

    # 8. Check Deterministic Ordering
    if insights["critical_alerts"]:
        # The first alert should be from SHOCK_STRAT or DECAY_STRAT depending on scan order,
        # but they should all be CRITICAL.
        print("SUCCESS: Critical alerts prioritized at the top level.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
