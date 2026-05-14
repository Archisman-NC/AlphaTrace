# PPO Reinforcement Learning Experiment

This document details the training methodology, evaluation results, and failure analysis of the experimental Proximal Policy Optimization (PPO) agent within the AlphaTrace research sandbox.

## 1. Experiment Methodology

*   **Model**: `stable_baselines3.PPO` with Multi-Layer Perceptron (MLP) policy.
*   **Environment**: `AlphaTraceTradingEnv` (Isolated Research Sandbox).
*   **Asset**: `RELIANCE.NS` (2-year historical data).
*   **Training Configuration**:
    *   Steps: 10,000
    *   Learning Rate: 3e-4
    *   Batch Size: 64
    *   Entropy Coefficient: 0.01 (to encourage exploration)

## 2. Evaluation Discipline

To prevent temporal leakage and over-optimization, we utilized a strict **Temporal Split**:
*   **Train Set (75%)**: Used for policy gradient optimization.
*   **Eval Set (25%)**: Unseen data used for deterministic backtesting of the learned policy.

## 3. Comparative Baseline

The PPO agent is compared against a **Buy and Hold (B&H)** baseline. This is the minimum threshold for a trading agent to be considered potentially useful.

## 4. Observed Behaviors

### A. Action Distribution Analysis
Initial runs often show a "Holding Bias" or "Overtrading" depending on the reward shaping:
*   **Inertia**: Agents may become "stuck" in a HOLD state if transaction penalties are too high.
*   **Flipping**: If penalties are too low, agents may flip between LONG and SHORT daily to capture micro-noise.

### B. Exploitation of Indicators
The agent successfully learns to weight **RSI** and **Bollinger %B** as primary state drivers, confirming that the observation space provides sufficient signal.

## 5. Failure Analysis (CRITICAL)

Despite the ability of RL agents to optimize for training data, several fundamental instabilities were observed:

### A. Overfitting (Temporal Sensitivity)
The PPO agent frequently overfits to the specific volatility patterns of the training period. When moved to the evaluation period, small shifts in market "physics" (regime changes) often lead to significant drawdown.

### B. Reward Shaping Instability
Small changes in the transaction penalty ($0.1\%$ vs $0.2\%$) lead to radically different policy outcomes. This "Sensitivity to Hyperparameters" makes RL agents inherently less stable than deterministic signal engines.

### C. Exploration Noise
Unlike RSI signals which have a clear geometric meaning, PPO actions in low-confidence states can appear stochastic or irrational to a human researcher.

## 6. Comparison: Deterministic vs. RL

| Feature | Deterministic Signals (RSI/MACD) | PPO RL Agent |
| :--- | :--- | :--- |
| **Interpretability** | 100% (Rule-based) | Low (Neural weights) |
| **Stability** | High (Repeatable) | Low (Sensitive to seeds) |
| **Adaptability** | None (Static) | High (Data-driven) |
| **Operational Risk** | Low | High (Non-linear failures) |

## 7. Conclusion

The PPO agent remains a **Research Artifact**. While it demonstrates the ability to ingest technical features and form a position policy, the observed instabilities and regime-sensitivity reinforce the AlphaTrace philosophy: **Deterministic intelligence must remain the production standard, while RL serves as an experimental sandbox for non-linear exploration.**
