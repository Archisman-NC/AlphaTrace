# Market Regime Detection Engine

AlphaTrace uses a Hidden Markov Model (HMM) to classify market states into probabilistic regimes. This allows the system to adjust its reasoning and expectations based on the broader market environment.

## Methodology

### 1. Feature Engineering
The engine operates on two primary features derived from Nifty 50 (`^NSEI`) data:
- **Log Returns**: Used for superior statistical properties (normality and additivity) compared to percentage returns.
- **Rolling Volatility**: 5-day standard deviation of log returns, helping the model distinguish between stable trends and high-fear states.

### 2. Probabilistic Modeling
- **Algorithm**: Gaussian Hidden Markov Model (`GaussianHMM`).
- **States**: Configured for 3 components by default.
- **Covariance**: Uses `full` covariance to capture dependencies between returns and volatility.

### 3. Deterministic Labeling
Since HMM hidden states are arbitrary, AlphaTrace implements a strict mapping logic:
1.  **State Statistics**: Compute the mean log-return for each hidden state.
2.  **Sorting**: Sort states by their mean return.
3.  **Mapping**:
    - **Lowest Mean Return** ──> `Bear`
    - **Middle Mean Return** ──> `Sideways`
    - **Highest Mean Return** ──> `Bull`

## Strategic Utility

### Persistence Analysis
The engine calculates "Days in Regime" to identify how long a trend has persisted. This is used by the AI to detect potential exhaustion points or trend maturity.

### AI Contextualization
The reasoning layer injects the current regime and its historical distribution (percentage of time spent in each state) into the AI's prompt. This prevents the LLM from making generic statements and forces it to acknowledge the current market volatility and trend.

## Performance Optimization
To ensure a snappy UI, the HMM fitting process is wrapped in **Streamlit Caching** (`@st.cache_data`). This prevents redundant re-computation during session refreshes while ensuring the model stays updated with the latest daily data.
