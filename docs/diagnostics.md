# Strategy Explainability & Diagnostics

AlphaTrace provides a deterministic diagnostics layer designed to provide full transparency into strategy behavior and performance attribution.

## Core Analytics

### 1. Signal Attribution
For every generated signal, the diagnostics engine produces a human-readable explanation.
- **Logic**: Combines technical triggers (RSI crosses) with the active market regime.
- **Example**: *"RSI (28.5) entered oversold territory during Bear regime."*

### 2. Regime-Conditioned Performance
The system slices standard backtest metrics across different market environments:
- **Annualized Sharpe**: Identifying the strategy's risk-adjusted edge in Bull vs. Bear markets.
- **Win Rate**: Tracking how consistently the strategy generates positive returns in different states.
- **Regime Exposure**: Calculating what percentage of the total backtest period the strategy spent in each regime.

### 3. Weakness Detection
A rules-based engine identifies structural risks:
- **Negative Alpha**: Flags regimes where the strategy has a negative Sharpe ratio.
- **Consistency Risks**: Detects win rates below critical thresholds (e.g., < 40%) in specific market states.

## AI Interpretation
The diagnostics layer synthesizes these findings into a concise, non-hype analytical summary. This summary serves as the "ground truth" for the AI reasoning engine, ensuring that the assistant's narrative is anchored in hard performance data rather than speculative chat.

## Engineering Constraint
All diagnostic outputs are **JSON-serializable** and free of Pandas/NumPy leakage, ensuring stable integration with downstream reasoning tools and UI components.
