---
title: AlphaTrace
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: 3.12
pinned: false
---

# Autonomous Financial Reasoning Engine

### Explain portfolio movement using causal AI + deterministic pipelines

---

## Overview

The Autonomous Financial Reasoning Engine is a production-grade causal financial reasoning system designed to bridge the gap between raw market signals and human-readable insights. By synthesizing market data, financial news, and individual portfolio holdings, the system identifies not just *what* happened to a portfolio, but explains *why* it moved. It constructs verifiable reasoning chains that link macroeconomic triggers to sector-level impacts and granular stock performance.

---

## Architecture

The system follows a strict multi-stage pipeline architecture to ensure data integrity and reasoning accuracy:

```text
Market Data + News + Portfolio Holdings
↓
Phase 1: Market Intelligence (Signal Extraction & Categorization)
↓
Phase 2: Portfolio Intelligence (Exposure Calculation & Metric Normalization)
↓
Phase 3: Reasoning Engine (Causal Chain Construction & Conflict Detection)
↓
Outcome: Structured Explanation + AI Judge Evaluation + Confidence Scoring
```

---

## How It Works

The engine operates on a strict causal chain model:
**News → Sector → Stock → Portfolio**

* **Trigger Identification**: The system ingests financial news and identifies macroeconomic or regulatory triggers.
* **Sector Propagation**: Mapping news impact to specific industry sectors (e.g., Banking, IT, Energy).
* **Security Linkage**: Attributing sector-level sentiment to individual stock tickers based on correlation and exposure.
* **Portfolio Attribution**: Aggregating security-level performance into total portfolio daily change and risk metrics.

---

## Key Features

* **Deterministic Reasoning Pipeline**: Core computations and causal linking are handled via pure logic to eliminate LLM hallucinations.
* **Sector-Level Impact Analysis**: Quantitative calculation of sector contribution to total portfolio variance.
* **Conflict Detection**: Automated identification of diverging signals where sectoral trends and security performance disagree.
* **Groq-Powered Explanations**: Uses Llama-3.3-70b via Groq for high-latency, low-cost narrative synthesis.
* **Hybrid Evaluation Layer**: Combines an LLM-as-a-Judge with a deterministic rule-check layer for structural validation.
* **Quantitative Confidence Scoring**: Multi-factor confidence calculation based on signal strength and data completeness.
* **Observability Integration**: Full lifecycle tracing of LLM interactions using Langfuse.

---

## Sample Output

```text
[FINAL ADVISORY EXPLANATION]
Portfolio declined by 2.73%. Banking holdings contributed -1.84%, primarily driven by HDFCBANK, as hawkish RBI stance pressured lending outlook. Uncertainty remains regarding the concentrated exposure to Banking and Financial Services sectors, which poses risk to portfolio stability.

Top Drivers:
* Banking holdings contributed -1.84%, primarily driven by HDFCBANK, as hawkish RBI stance pressured lending outlook.
* Financial Services holdings contributed -0.37%, primarily driven by BAJFINANCE, as tight liquidity conditions weighed on NBFCs.

Risks:
* Concentrated exposure to Banking sector poses risk to portfolio stability.
* Combined exposure to Banking/Finance sectors poses risk to portfolio diversification.

Confidence: 0.90
AI Judge Score: 10.0 / 10
```

---

## How To Run

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
Create a `.env` file with `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`.

3. Execute the pipeline for the default portfolio:
```bash
python3 main.py
```

4. Execute for a specific portfolio ID:
```bash
python3 main.py --portfolio PORTFOLIO_002
```

---

## Tech Stack

* **Language**: Python 3.9+
* **Validation**: Pydantic v2
* **Inference**: Groq SDK (Llama 3.3 70B)
* **Observability**: Langfuse SDK v4
* **Environment**: python-dotenv

---

## Design Decisions

### Hybrid System: Deterministic Reasoning + LLM Interpretation
The most critical architectural decision was to separate **Reasoning** from **Interpretation**.
* **Deterministic Logic** performs all financial calculations, sector mappings, and causal linking. This ensures that the grounding of every explanation is mathematically verifiable.
* **LLM Layer** is strictly restricted to narrative synthesis and qualitative evaluation.
By preventing the LLM from performing math or link-building, the system effectively eliminates hallucination and ensures high-reliability performance in financial contexts.

---

## Observability

The system integrates **Langfuse** for production-grade observability. Every execution cycle is tracked with detailed traces:
* **Trace Lifecycle**: Captures the full context from input prompts to final JSON outputs.
* **Performance Metrics**: Monitors token usage, provider latency, and cost per execution.
* **Debugging**: Enables granular trace-level debugging for both the generation and evaluation phases.

---

## Known Limitations

* **LLM Judge calibration**: The evaluation model does not use a human-labeled benchmark, so scores tend to skew high (8–10). In a production setting, this would be calibrated against expert-graded explanations to ensure scoring rigor.

* **Mock data dependency**: The causal reasoning layer depends on the richness of input news. With sparse signals, the system falls back to quantitative inference, which reduces the narrative's explanatory depth.

* **Static sector mapping**: Stocks not present in the reference sector mapping are ignored during normalization. Integration with a live security data API would be required for broader universe coverage.

---

## What I’d extend next

* **Factor attribution**: Decompose portfolio movement into granular components, separating idiosyncratic stock-level performance from broad sector-level volatility.

* **Advanced Risk Modeling (VaR / CVaR)**: Implement rolling risk estimation using historical simulation or parametric models to provide forward-looking volatility projections.

* **Dynamic Signal Weighting**: Refine the confidence scoring model by weighting signals based on magnitude, temporal relevance, and cross-signal alignment, moving beyond the current heuristic-based thresholds.

* **Real-time Market Integration**: Transition from batch ingestion from mock JSON sources to real-time streams (e.g., via ZeroMQ or WebSocket) for low-latency portfolio monitoring.

---

## Web Interface (Hugging Face Spaces)

AlphaTrace includes a production-ready web interface built with Gradio, allowing users to trigger reasoning cycles and visualize terminal outputs directly in the browser.

### Features
* **Live Sandbox**: Run analysis on pre-configured portfolios (PORTFOLIO_001, 002, 003, or ALL).
* **Console Output Capture**: Displays the full internal trace, including P&L metrics, causal drivers, risks, and AI Judge scores.
* **Observability Integration**: Automatically links generated briefings to Langfuse traces if configured.

### Deployment to Spaces
1. **New Space**: Create a new Gradio Space on Hugging Face.
2. **Settings**: Add the following secret environment variables:
   * `GROQ_API_KEY`: Your Groq platform key.
   * `LANGFUSE_PUBLIC_KEY`: For telemetry tracing.
   * `LANGFUSE_SECRET_KEY`: For telemetry tracing.
3. **Upload**: Push the repository contents (including `app.py`, `main.py`, `data/`, and `app/`).

### Local Execution
To run the dashboard locally:
```bash
python app.py
```
The interface will be accessible at `http://localhost:7860`.
