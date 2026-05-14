# Research Methodology & Philosophy

AlphaTrace is designed as a **Research Infrastructure**, not a "Black-Box" trading bot. This document outlines the core quantitative philosophy guiding its development.

## 1. Deterministic Intelligence Philosophy

We believe that in quantitative research, **Interpretability > Prediction**. 

*   **Deterministic Logic**: Every signal, regime, and alert is the result of fixed, auditable mathematical rules.
*   **No Black-Boxes**: We intentionally avoid complex neural networks or non-deterministic ML models that cannot explain *why* a specific output was generated.
*   **Reproducibility**: Identical market states always produce identical intelligence outputs.

## 2. Interpretability-First Design

AlphaTrace is built to provide **Causal Intelligence**:
*   Every signal comes with a `causal_reason`.
*   Every portfolio status comes with a `diagnostic`.
*   Every AI insight is anchored in a pre-computed quantitative context.

## 3. Operational Monitoring vs. Forecasting

The platform focuses on **Operational Intelligence**—understanding the "Now" and the "State of Health":
*   **Regime Detection**: Identifying the current market environment (HMM).
*   **Watchdog**: Monitoring for strategy decay and structural shifts.
*   **Opportunity Surfacing**: Identifying where indicators are currently aligned.

## 4. What AlphaTrace is NOT

To maintain research-grade integrity, AlphaTrace explicitly avoids:
*   **Autonomous Execution**: No broker integrations or "one-click" trading.
*   **Opaque ML Scoring**: No "Confidence 98%" scores from non-interpretable models.
*   **Predictive Hype**: No claims of "Price will go to X."
*   **"AI Trader" Narratives**: The AI is a context-aware assistant, not the decision-maker.

## 5. Known Limitations & Future Directions

### Current Limitations
*   No transaction-cost or slippage modeling.
*   Static signal thresholds (not yet regime-adaptive).
*   No online learning or probabilistic calibration.
*   Single timeframe analysis (Daily).

### Future Research Directions
*   **Adaptive Thresholds**: Modifying signal triggers based on HMM volatility regimes.
*   **Probabilistic Calibration**: Using historical signal accuracy to weight confidence.
*   **Factor-Model Integration**: Adding macro-economic and fundamental factor layers.
*   **Meta-Strategy Selection**: Automatically suggesting the most robust strategy for the current regime.
