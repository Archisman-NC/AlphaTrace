import sys
import os
import pandas as pd
import numpy as np
from dataclasses import asdict

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.rl.research_evaluator import (
        compute_metrics,
        evaluate_research_pipeline,
        generate_research_summary
    )
    from app.quant.regime_detector import REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS

    print("--- Verifying Comparative Research Evaluator ---")

    # 1. Prepare Mock Data
    dates = pd.date_range("2024-01-01", periods=100)
    ppo_rets = pd.Series(np.random.normal(0.001, 0.015, 100), index=dates)
    rsi_rets = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    bh_rets = pd.Series(np.random.normal(0.0002, 0.012, 100), index=dates)
    
    # Mock regimes
    regime_choices = [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]
    regimes = pd.Series(np.random.choice(regime_choices, 100), index=dates)

    ppo_results = {"returns": ppo_rets, "total_trades": 15}
    rsi_results = {"returns": rsi_rets, "total_trades": 8}
    bh_results = {"returns": bh_rets}

    # 2. Test Metric Computation
    print("\nTesting Metric Computation...")
    eval_obj = compute_metrics(ppo_rets, "Test_PPO", regimes)
    print(f"SUCCESS: PPO Sharpe={eval_obj.sharpe_ratio:.2f}")
    print(f"Regime Performance Keys: {list(eval_obj.regime_performance.keys())}")

    # 3. Test Comparative Pipeline
    print("\nTesting Comparative Pipeline...")
    comparison = evaluate_research_pipeline(ppo_results, rsi_results, bh_results, regimes)
    print(f"SUCCESS: Comparison summary generated.")
    print(f"Summary Snippet: {comparison.summary[:80]}...")

    # 4. Verify JSON Serializability
    try:
        json_data = asdict(comparison)
        print("SUCCESS: Research results are JSON serializable.")
    except Exception as e:
        print(f"FAILURE: Serialization failed: {e}")

    # 5. Verify Regime-Aware Logic
    if REGIME_BULL in comparison.ppo_metrics.regime_performance:
        print(f"SUCCESS: Regime-aware performance validated for {REGIME_BULL}.")
    else:
        print(f"FAILURE: Missing regime performance data.")

    print("\n--- Research Evaluator Verification Complete ---")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
