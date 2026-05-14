import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data.market_data import fetch_ohlcv
from app.quant.signals import compute_signals
from app.rl.trading_env import AlphaTraceTradingEnv
from app.rl.feature_builder import build_rl_features

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PPO-Research")

def run_ppo_experiment(ticker="RELIANCE.NS", seed=42):
    """
    Experimental PPO Training and Evaluation Workflow.
    """
    logger.info(f"Starting PPO Experiment for {ticker} (Seed: {seed})")
    
    # 1. Deterministic Seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 2. Data Preparation
    raw_df = fetch_ohlcv(ticker, period="2y")
    if raw_df.empty:
        logger.error("Failed to fetch data. Aborting.")
        return
    
    # Calculate indicators
    df = compute_signals(raw_df)
    features_df = build_rl_features(df)
    prices = df["Close"]
    
    # 3. Train/Eval Split (75/25 Temporal Split)
    split_idx = int(len(features_df) * 0.75)
    train_features = features_df.iloc[:split_idx]
    train_prices = prices.iloc[:split_idx]
    
    eval_features = features_df.iloc[split_idx:]
    eval_prices = prices.iloc[split_idx:]
    
    logger.info(f"Split: Train ({len(train_features)} days), Eval ({len(eval_features)} days)")
    
    # 4. Initialize Environments
    train_env = AlphaTraceTradingEnv(train_features, train_prices)
    eval_env = AlphaTraceTradingEnv(eval_features, eval_prices)
    
    # 5. Initialize PPO Model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu"
    )
    
    # 6. Training Phase
    logger.info("Training PPO Agent...")
    model.learn(total_timesteps=10000)
    logger.info("Training complete.")
    
    # 7. Evaluation Phase
    logger.info("Evaluating Policy...")
    obs, _ = eval_env.reset(seed=seed)
    
    ppo_net_worths = [eval_env.initial_balance]
    actions = []
    
    for _ in range(len(eval_features) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ppo_net_worths.append(info["net_worth"])
        actions.append(int(action))
        if terminated or truncated:
            break
            
    # 8. Baseline Comparison (Buy and Hold)
    bh_returns = (eval_prices / eval_prices.iloc[0]) * eval_env.initial_balance
    
    # 9. Performance Metrics
    ppo_returns = pd.Series(ppo_net_worths).pct_change().dropna()
    sharpe = (ppo_returns.mean() / ppo_returns.std()) * np.sqrt(252) if ppo_returns.std() != 0 else 0
    
    total_trades = eval_env.total_trades
    final_nw = eval_env.net_worth
    bh_final_nw = bh_returns.iloc[-1]
    
    logger.info(f"--- Evaluation Summary ({ticker}) ---")
    logger.info(f"Final Net Worth (PPO): {final_nw:,.2f}")
    logger.info(f"Final Net Worth (B&H): {bh_final_nw:,.2f}")
    logger.info(f"Sharpe Ratio (PPO):    {sharpe:.2f}")
    logger.info(f"Total Trades:          {total_trades}")
    
    # 10. Action Distribution Analysis
    action_counts = pd.Series(actions).value_counts(normalize=True).to_dict()
    logger.info(f"Action Distribution: 0(HOLD): {action_counts.get(0,0):.1%}, 1(BUY): {action_counts.get(1,0):.1%}, 2(SELL): {action_counts.get(2,0):.1%}")

    # 11. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_net_worths, label="PPO Agent", color="blue", linewidth=1.5)
    plt.plot(bh_returns.values, label="Buy & Hold Baseline", color="gray", linestyle="--", alpha=0.7)
    plt.title(f"AlphaTrace RL Research: PPO vs Baseline ({ticker})")
    plt.xlabel("Days (Evaluation Period)")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("research/plots", exist_ok=True)
    plot_path = f"research/plots/ppo_equity_{ticker.replace('.', '_')}.png"
    plt.savefig(plot_path)
    logger.info(f"Equity curve saved to {plot_path}")
    
    return {
        "final_nw": final_nw,
        "sharpe": sharpe,
        "trades": total_trades,
        "action_dist": action_counts
    }

if __name__ == "__main__":
    run_ppo_experiment()
