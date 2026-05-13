# AlphaTrace: Contextual Quantitative Intelligence

AlphaTrace is a professional-grade quantitative research platform that combines deterministic backtesting with regime-aware AI reasoning. It is designed for researchers who require high-fidelity analytical rigor paired with interpretable, context-specific market insights.

## Core Features

- **📈 Vectorized Backtesting**: High-performance strategy evaluation with lookahead bias protection and geometric compounding.
- **🧠 HMM Regime Detection**: Probabilistic market classification (Bull/Bear/Sideways) using Hidden Markov Models.
- **🔍 Deterministic Diagnostics**: Strategy explainability that attributes performance and signal triggers to specific market states.
- **💬 AI Copilot**: A reasoning engine that interprets quantitative snapshots to provide grounded, non-speculative market analysis.
- **📊 Interactive Terminal**: A Streamlit-based dashboard for rapid strategy iteration and visual performance audit.

## Architecture Overview

```text
       [ Market Ingestion ]
               ↓
    [ Signal & Indicator Engine ]
               ↓
    [ Vectorized Backtesting ] ──> [ Performance Metrics ]
               ↓                         ↓
    [ HMM Regime Detection ] ──> [ Strategy Diagnostics ]
               ↓                         ↓
      [ AI Context Mixer ] <─────────────┘
               ↓
      [ Streamlit Dashboard ]
```

## Repository Structure

- `app/data/`: Market ingestion and portfolio construction.
- `app/quant/`: The analytical core (Signals, Backtester, Regimes, Diagnostics).
- `app/reasoning/`: The AI orchestration layer and context loaders.
- `app/ui/`: Streamlit dashboard implementation.
- `docs/`: Technical architecture and implementation details.
- `scratch/`: Official component verification scripts.

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd AlphaTrace

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Project

To launch the interactive research terminal:

```bash
python run_app.py
```

## Quantitative Philosophy

AlphaTrace is built on **Deterministic Contracts**. Every analytical output—from Sharpe ratios to regime labels—is generated through reproducible, mathematically grounded pipelines. The AI layer acts as an interpretability bridge, never as a black-box signal generator.

---
*AlphaTrace: Interpretable Intelligence for Quantitative Research.*
