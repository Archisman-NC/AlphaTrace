# Comparative RL Research Framework

This document outlines the formal methodology used in AlphaTrace to evaluate **Reinforcement Learning (PPO)** agents against **Deterministic Signal Strategies** (RSI/MACD) and **Market Baselines**.

## 1. Evaluation Philosophy

AlphaTrace treats RL as an **Exploratory Research Methodology**, not a production alternative to deterministic quant logic. The core goal of this research layer is to answer:
1.  Does the RL agent learn behaviors that deterministic rules miss?
2.  How stable is the RL policy across structural market transitions (Regimes)?
3.  Is the added complexity of RL justified by a superior Sharpe ratio?

## 2. Comparative Metrics

We use a standardized suite of metrics to ensure "Apples-to-Apples" comparison:

| Metric | Purpose |
| :--- | :--- |
| **Sharpe Ratio** | Risk-adjusted efficiency comparison. |
| **Max Drawdown** | Stability and tail-risk evaluation. |
| **Regime Return** | Performance conditioned on Bull/Bear/Sideways states. |
| **Trade Density** | Analysis of overtrading vs. inactivity tendencies. |

## 3. Regime-Aware Comparison

By integrating the **HMM Regime Detector**, we can analyze where each approach excels:

*   **Deterministic Signals**: Usually exhibit high stability in Trending (Bull/Bear) regimes where momentum is clear but may struggle with "Whipsaws" in Sideways regimes.
*   **PPO Agents**: Often exhibit extreme instability in Bear regimes or volatility spikes due to "Policy Collapse" or lack of exploration in training.

## 4. Why Deterministic Rules Remain Production-Grade

Through rigorous comparative research, we consistently find that:
1.  **Interpretability is a Risk Control**: In a crash, a researcher can explain an RSI signal. A PPO policy's internal weights remain a black box.
2.  **Consensus Stability**: Deterministic rules provide a "Ground Truth" consensus. RL policies are highly sensitive to random seeds and training noise.
3.  **Operational Cost**: RL requires significant computational overhead and data-engineering complexity without a guaranteed alpha premium over simple, well-tuned signals.

## 5. Research Summaries

The system generates deterministic research summaries to synthesize findings:
*   *Example*: "Deterministic RSI signals demonstrated higher stability and lower drawdown than PPO across Bear regimes. PPO policy exhibited strong inactivity bias, limiting participation during trending periods."

## 6. Future Directions

*   **Multi-Agent Benchmarking**: Comparing different RL architectures (DQN vs. PPO vs. SAC).
*   **Regime-Adaptive Ensembles**: Researching if an ensemble of deterministic signals can be weighted by an RL "Meta-Controller."
*   **Adversarial Regimes**: Specifically training agents on "Crisis Data" to evaluate tail-risk mitigation.
