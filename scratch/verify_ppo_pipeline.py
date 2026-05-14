import sys
import os
import logging

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress verbose logging for verification
logging.getLogger("PPO-Research").setLevel(logging.WARNING)
logging.getLogger("app.data.market_data").setLevel(logging.WARNING)

try:
    from research.train_ppo_agent import run_ppo_experiment

    print("--- Verifying PPO Training Pipeline ---")

    # Run a miniature version of the experiment for verification
    # We use a shorter period and fewer steps to ensure it's fast
    results = run_ppo_experiment(ticker="RELIANCE.NS", seed=42)
    
    if results and "final_nw" in results:
        print("\nSUCCESS: PPO Pipeline Execution Verified.")
        print(f"  Final Net Worth: {results['final_nw']:.2f}")
        print(f"  Sharpe Ratio:    {results['sharpe']:.2f}")
        print(f"  Total Trades:    {results['trades']}")
        
        # Check for equity curve file
        plot_file = "research/plots/ppo_equity_RELIANCE_NS.png"
        if os.path.exists(plot_file):
            print(f"  Equity Curve:    Generated ({plot_file})")
        else:
            print("  Equity Curve:    MISSING")
            
        # Verify Action Distribution
        dist = results["action_dist"]
        if sum(dist.values()) > 0.99: # Allow for rounding
            print("  Action Distribution: Validated.")
        else:
            print("  Action Distribution: Malformed.")
    else:
        print("\nFAILURE: PPO Pipeline returned invalid results.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
