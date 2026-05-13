# AlphaTrace Architecture

AlphaTrace is a modular, deterministic quantitative research platform designed to bridge the gap between hard analytical backtesting and contextual AI reasoning.

## System Layers

### 1. Market Data Layer (`app/data/`)
- **Responsibility**: Ingests live market data via `yfinance`.
- **Contracts**: Enforces strict OHLCV schemas and ensures clean `DatetimeIndex` alignment.
- **Components**: `market_data.py`, `portfolio_builder.py`.

### 2. Quantitative Engine (`app/quant/`)
- **Responsibility**: Feature engineering, signal generation, and vectorized backtesting.
- **Contracts**:
    - **Signals**: Produces Buy (1), Sell (-1), and Hold (0) labels with indicators (RSI, MACD, BB).
    - **Backtester**: Evaluates signals with lookahead bias prevention and geometric compounding.
- **Components**: `signals.py`, `backtester.py`.

### 3. Regime Detection Engine (`app/quant/regime_detector.py`)
- **Responsibility**: Probabilistic market state classification using Hidden Markov Models (HMM).
- **Logic**: Maps hidden states to deterministic labels (Bull, Bear, Sideways) based on mean returns and volatility.

### 4. Explainability & Diagnostics (`app/quant/diagnostics.py`)
- **Responsibility**: Deterministic attribution of strategy performance.
- **Logic**: Analyzes Sharpe ratios and win rates conditioned on market regimes; explains signal triggers using technical levels.

### 5. Reasoning Layer (`app/reasoning/`)
- **Responsibility**: Orchestrates quantitative context for the AI Copilot.
- **Contract**: Assembles a JSON-serializable "Context Snapshot" containing portfolio metrics, regime distribution, and analytical summaries.
- **Components**: `router.py`, `intent_classifier.py`.

### 6. UI Layer (`app/ui/`)
- **Responsibility**: Presentation and interaction via Streamlit.
- **Components**: `dashboard.py`.

## Data Flow

```
[ Market Data ] ──> [ Signal Engine ] ──> [ Backtester ]
      │                                         │
      │                                         │
[ Regime Detector ] ──────────┐                 │
      │                       │                 │
      ▼                       ▼                 ▼
[ AI Context Mixer ] <── [ Diagnostics ] <── [ Equity Curve ]
      │
      ▼
[ Streamlit UI ] <── [ LLM Response ]
```

## Deterministic Principles
- **No Random Seed Drift**: All models (HMM) use fixed `random_state`.
- **Lookahead Bias Prevention**: Signals are always shifted by one period before execution.
- **No NaN Leakage**: Strict `dropna` policies ensure mathematical integrity across vectorized operations.
