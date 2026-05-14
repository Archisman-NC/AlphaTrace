# Statistical Watchdog Framework

The Watchdog Engine is the operational health monitor for AlphaTrace, designed to detect strategy degradation and statistical anomalies in real-time.

## 1. Multi-Detector Strategy

A strategy violation can manifest in different statistical dimensions. AlphaTrace uses three distinct detectors to capture these variations:

### A. Sharpe Decay (`detect_sharpe_decay`)
*   **Metric**: Rolling 20-day Sharpe vs. Trailing 252-day Sharpe.
*   **Purpose**: Detects slow "alpha decay" or strategy degradation where the risk-adjusted edge is collapsing.
*   **Trigger**: Triggered when the recent Sharpe drops below a specific ratio of the historical baseline.

### B. Distribution Shift (`detect_distribution_shift`)
*   **Metric**: Kolmogorov-Smirnov (KS) Test $p$-value.
*   **Purpose**: Identifies structural changes in the return distribution (e.g., a sudden expansion in volatility or a "fat-tail" event).
*   **Trigger**: Triggered when the KS $p$-value falls below $0.05$ (Significant) or $0.01$ (Critical).

### C. Z-Score Breach (`detect_zscore_breach`)
*   **Metric**: Standardized return ($Z = \frac{R - \mu}{\sigma}$).
*   **Purpose**: Detects acute outlier events (crashes or spikes) that require immediate operational investigation.
*   **Trigger**: Triggered when $|Z| > 3.0$ (High) or $|Z| > 4.0$ (Critical).

## 2. Operational Escalation Flow

Alerts are prioritized and aggregated into a single "Operational Status" for the portfolio.

| Status | Trigger Condition |
| :--- | :--- |
| `CRITICAL` | Any `CRITICAL` severity alert (e.g., $Z > 4.0$ or KS $p < 0.001$). |
| `DEGRADED` | Multiple `HIGH` alerts or mixed `CRITICAL/HIGH` violations. |
| `WATCH` | Any `HIGH` severity alert (e.g., Sharpe ratio collapse). |
| `STABLE` | No statistical anomalies detected. |

## 3. Difference Between Anomalies

| Type | Temporal Horizon | Insight |
| :--- | :--- | :--- |
| **Z-Score** | High Frequency | Sudden shock or outlier. |
| **Dist Shift** | Medium Frequency | Change in "Market Physics" or regime. |
| **Sharpe Decay**| Low Frequency | Slow strategy obsolescence. |

## 4. Deterministic Summaries

The Watchdog generates human-readable but deterministic summaries to ensure clarity:
*   *Example*: "Critical Sharpe degradation detected in INFY.NS with accompanying return distribution instability."
