# AlphaTrace Demo Workflows

This document provides step-by-step walkthroughs for demonstrating the core capabilities of the AlphaTrace quantitative intelligence platform.

## 1. Operational Dashboard Walkthrough
**Goal**: Demonstrate the primary research and monitoring interface.
1.  **Launch**: Run `python3 run_dashboard.py`.
2.  **Portfolio Selection**: Choose "Rahul Sharma (Diversified)" to see a multi-asset setup.
3.  **Real-Time Analytics**: Show the performance metrics and regime detection headers.
4.  **Interactive Copilot**: Ask the AI: *"What is the current market regime and how is it affecting my portfolio?"* to demonstrate context-aware reasoning.

## 2. Signal Intelligence Demo
**Goal**: Surface actionable opportunities using deterministic technical indicators.
1.  Navigate to the **🎯 Signals** tab.
2.  Select a portfolio and click **🚀 Generate Signals**.
3.  **Surfacing Opportunities**: Show the "Top Opportunity Concentration" cards.
4.  **Causal Reasoning**: Highlight the `causal_reason` field in the signal inventory to show interpretability.
5.  **Sector Conviction**: Explain the sector bar chart and how it identifies clustering trends.

## 3. Statistical Watchdog Demo
**Goal**: Demonstrate operational strategy health monitoring.
1.  Navigate to the **🚨 Watchdog** tab.
2.  Select a portfolio and click **🚨 Run Health Scan**.
3.  **Escalation Logic**: Show how Sharpe decay and Z-score breaches are aggregated into a "Portfolio Status" (e.g., `WATCH` or `DEGRADED`).
4.  **AI Escalation**: Explain the "Escalation Context" preview to show what the AI assistant sees.

## 4. RL Research Sandbox Demo
**Goal**: Show the isolated environment for experimental trading agents.
1.  **Run Experiment**: Run `python3 run_research.py`.
2.  **Pipeline Visibility**: Show the training metrics (rewards, net worth) in the console.
3.  **Artifact Analysis**: Open `research/artifacts/plots/` to show the comparative equity curve (PPO vs. B&H).
4.  **Comparative Diagnostics**: Refer to `docs/ppo_experiment.md` to explain why the RL agent's instability is documented honestly.

## 5. Verification & Integrity Demo
**Goal**: Demonstrate the robustness of the quantitative stack.
1.  **Run Suite**: Run `python3 run_verification.py`.
2.  **Reproducibility**: Show the pass/fail summary to prove that every quant module is functioning as per the deterministic contract.
