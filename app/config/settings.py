import os
import numpy as np

# --- 1. Market Data & Fetching ---
DEFAULT_LOOKBACK_PERIOD = "6mo"
INDICATOR_LOOKBACK = "1y" # For stable RSI/MACD/Regime calculation
TICKERS_FOR_REGIME = ["^NSEI"] # Nifty 50 as benchmark

# --- 2. Technical Signal Thresholds ---
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
BB_WINDOW = 20
BB_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- 3. Watchdog Monitoring Thresholds ---
WATCHDOG_ROLLING_WINDOW = 20
WATCHDOG_SHARPE_DECAY_THRESHOLD = 0.5 # Trigger if current Sharpe < 50% of historical
WATCHDOG_KS_CRITICAL = 0.01
WATCHDOG_KS_WARNING = 0.05
WATCHDOG_ZSCORE_CRITICAL = 4.0
WATCHDOG_ZSCORE_HIGH = 3.0

# --- 4. Reinforcement Learning (Experimental) ---
RL_INITIAL_BALANCE = 100000
RL_TRANSACTION_COST = 0.001 # 0.1% per trade
RL_TRAIN_STEPS = 10000
RL_EVAL_SPLIT = 0.75 # 75% train, 25% eval

# PPO Hyperparameters
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 128
PPO_BATCH_SIZE = 64
PPO_ENTROPY_COEF = 0.01

# --- 5. Dashboard Configuration ---
DASHBOARD_REFRESH_INTERVAL = 3600 # Cache TTL for market data
DASHBOARD_TITLE = "📊 AlphaTrace Intelligence"

# --- 6. Path Management ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEARCH_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "research", "artifacts")
PLOT_DIR = os.path.join(RESEARCH_ARTIFACTS_DIR, "plots")

# Ensure directories exist
for d in [RESEARCH_ARTIFACTS_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
