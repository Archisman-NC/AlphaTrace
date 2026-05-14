# RL Research Sandbox (Experimental)

The `app/rl/` package is a self-contained research sandbox for experimental reinforcement learning (RL) trading agents. 

> [!WARNING]
> **Experimental Infrastructure Only**.
> The RL module is strictly for quantitative research and educational demonstration. It is **NOT** integrated into the production reasoning layer, signal generation, or dashboard intelligence.

## 1. Research Philosophy

AlphaTrace maintains a strict separation between its **Deterministic Quant Stack** and its **Experimental RL Sandbox**. RL agents are used as a "Simulation Sandbox" to explore non-linear decision-making patterns, not for live execution or insight generation.

## 2. Environment Design: AlphaTraceTradingEnv

The environment uses `gymnasium` and follows a simplified Markov Decision Process (MDP) for trading.

### A. Action Space (`spaces.Discrete(3)`)
*   **0: HOLD** - Maintain current position.
*   **1: BUY** - Enter or maintain a LONG position.
*   **2: SELL** - Enter or maintain a SHORT position.

### B. Observation Space
The observation vector consists of 7 normalized features:
1.  **Log Return**: Daily price change.
2.  **Volatility**: 20-day rolling standard deviation.
3.  **RSI Norm**: Relative Strength Index (scaled 0-1).
4.  **MACD Norm**: MACD histogram relative to price.
5.  **Bollinger %B**: Price position relative to bands.
6.  **Log Vol Change**: Change in trading volume.
7.  **Position State**: Current agent state (-1, 0, 1).

### C. Reward Shaping
The reward function is designed to balance profitability with operational stability:
$$Reward = R_t - \text{Penalty}_{transaction}$$
*   **$R_t$**: Percentage change in net worth at time $t$.
*   **Penalty**: A 0.1% penalty is applied for every position change to discourage overtrading and account for slippage.

## 3. Position Logic

The agent manages a simplified directional state:
*   **Flat (0)**: No market exposure.
*   **Long (1)**: 100% long exposure to the asset.
*   **Short (-1)**: 100% short exposure to the asset.

## 4. Known Research Limitations

*   **No Microstructure**: The environment assumes perfect execution at closing prices.
*   **Simplified Liquidity**: Assumes no market impact from agent actions.
*   **Daily Granularity**: RL agents operate on daily bars, limiting their ability to respond to intraday volatility.
*   **Isolated Execution**: No connection to real-time data or broker APIs.

## 5. Usage in Research

The sandbox is used to:
1.  Test different reward shaping strategies (e.g., Sharpe-based rewards).
2.  Compare deterministic technical signals against RL agent policies.
3.  Simulate "Adversarial" market regimes to stress-test position transitions.
