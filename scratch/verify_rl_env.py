import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.rl.trading_env import AlphaTraceTradingEnv
    from app.rl.feature_builder import build_rl_features

    print("--- Verifying RL Research Sandbox ---")

    # 1. Prepare Mock Data
    print("\nPreparing mock market data...")
    dates = pd.date_range("2024-01-01", periods=100)
    mock_df = pd.DataFrame({
        "Open": np.random.uniform(100, 110, size=100),
        "High": np.random.uniform(110, 120, size=100),
        "Low": np.random.uniform(90, 100, size=100),
        "Close": np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
        "Volume": np.random.uniform(1000, 5000, size=100),
        "rsi": np.random.uniform(30, 70, size=100),
        "macd": np.random.normal(0, 1, 100),
        "macd_signal": np.random.normal(0, 1, 100),
        "bb_upper": 160,
        "bb_lower": 90
    }, index=dates)

    # 2. Build Features
    print("Building RL features...")
    features_df = build_rl_features(mock_df)
    print(f"Features shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")

    # 3. Initialize Environment
    print("\nInitializing AlphaTraceTradingEnv...")
    env = AlphaTraceTradingEnv(features_df, mock_df["Close"])
    
    # 4. Test Reset
    obs, info = env.reset(seed=42)
    print(f"Reset Observation: {obs}")
    print(f"Observation Shape: {obs.shape} (Expected: {features_df.shape[1] + 1})")

    # 5. Run Random Agent Simulation
    print("\nRunning Random Agent Simulation (10 steps)...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.6f}, Net Worth={info['net_worth']:.2f}, Pos={info['position']}")
        
        if terminated or truncated:
            print("Episode Terminated.")
            break

    # 6. Test Deterministic Seeding
    print("\nTesting Deterministic Seeding...")
    env.reset(seed=123)
    action_1 = env.action_space.sample() # Action from seed 123
    
    env.reset(seed=123)
    action_2 = env.action_space.sample()
    
    if action_1 == action_2:
        print(f"SUCCESS: Seeding is deterministic (Action={action_1}).")
    else:
        print(f"FAILURE: Seeding is inconsistent (Action1={action_1}, Action2={action_2}).")

    print("\n--- RL Environment Verification Complete ---")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
