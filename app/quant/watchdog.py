import logging
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict

# Initialize logger
logger = logging.getLogger(__name__)

# Severity Ranking for deterministic sorting
SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3
}

@dataclass
class WatchdogAlert:
    alert_type: Literal["SHARPE_DECAY", "DISTRIBUTION_SHIFT", "ZSCORE_BREACH"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ticker: str
    message: str
    metric_value: float
    threshold: float
    triggered_at: str

def detect_sharpe_decay(
    returns: pd.Series,
    ticker: str,
    window: int = 20,
    decay_threshold: float = 0.5
) -> Optional[WatchdogAlert]:
    """
    Detect gradual degradation in strategy quality by comparing recent rolling Sharpe
    vs trailing historical Sharpe.
    """
    if len(returns) < window * 2:
        logger.debug(f"Insufficient history for Sharpe decay detection: {ticker}")
        return None

    try:
        # Calculate trailing historical Sharpe (excluding recent window)
        historical_returns = returns.iloc[:-window]
        recent_returns = returns.iloc[-window:]

        def calc_annualized_sharpe(r):
            std = r.std()
            if std == 0 or np.isnan(std): return 0.0
            return (r.mean() / std) * np.sqrt(252)

        trailing_sharpe = calc_annualized_sharpe(historical_returns)
        recent_sharpe = calc_annualized_sharpe(recent_returns)

        # Handle low signal environment
        if trailing_sharpe <= 0:
            return None

        ratio = recent_sharpe / trailing_sharpe

        if recent_sharpe < trailing_sharpe * decay_threshold:
            severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
            if ratio < 0.2: severity = "CRITICAL"
            elif ratio < 0.4: severity = "HIGH"

            return WatchdogAlert(
                alert_type="SHARPE_DECAY",
                severity=severity,
                ticker=ticker,
                message=f"Strategy Sharpe decayed to {recent_sharpe:.2f} (vs trailing {trailing_sharpe:.2f})",
                metric_value=float(recent_sharpe),
                threshold=float(trailing_sharpe * decay_threshold),
                triggered_at=str(returns.index[-1])
            )
    except Exception as e:
        logger.error(f"Error in detect_sharpe_decay for {ticker}: {e}")
    
    return None

def detect_distribution_shift(
    returns: pd.Series,
    ticker: str,
    window: int = 20,
    p_threshold: float = 0.05
) -> Optional[WatchdogAlert]:
    """
    Detect structural changes in return behavior using Kolmogorov-Smirnov test.
    """
    if len(returns) < window * 2:
        return None

    try:
        historical_returns = returns.iloc[:-window].dropna()
        recent_returns = returns.iloc[-window:].dropna()

        if len(historical_returns) < 10 or len(recent_returns) < 10:
            return None

        stat, p_value = stats.ks_2samp(historical_returns, recent_returns)

        if p_value < p_threshold:
            severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
            if p_value < 0.001: severity = "CRITICAL"
            elif p_value < 0.01: severity = "HIGH"

            return WatchdogAlert(
                alert_type="DISTRIBUTION_SHIFT",
                severity=severity,
                ticker=ticker,
                message=f"Structural shift detected in return distribution (p-value: {p_value:.4f})",
                metric_value=float(p_value),
                threshold=float(p_threshold),
                triggered_at=str(returns.index[-1])
            )
    except Exception as e:
        logger.error(f"Error in detect_distribution_shift for {ticker}: {e}")

    return None

def detect_zscore_breach(
    returns: pd.Series,
    ticker: str,
    window: int = 30,
    z_threshold: float = 2.5
) -> Optional[WatchdogAlert]:
    """
    Detect sudden abnormal return events (spikes or crashes).
    """
    if len(returns) < window:
        return None

    try:
        latest_return = returns.iloc[-1]
        historical_returns = returns.iloc[-window:-1]
        
        mean = historical_returns.mean()
        std = historical_returns.std()

        if std == 0 or np.isnan(std):
            return None

        z_score = (latest_return - mean) / std

        if abs(z_score) > z_threshold:
            severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
            if abs(z_score) > 4: severity = "CRITICAL"
            elif abs(z_score) > 3: severity = "HIGH"

            event_type = "spike" if z_score > 0 else "crash"
            return WatchdogAlert(
                alert_type="ZSCORE_BREACH",
                severity=severity,
                ticker=ticker,
                message=f"Abnormal return {event_type} detected (Z-Score: {z_score:.2f})",
                metric_value=float(z_score),
                threshold=float(z_threshold),
                triggered_at=str(returns.index[-1])
            )
    except Exception as e:
        logger.error(f"Error in detect_zscore_breach for {ticker}: {e}")

    return None

def scan_portfolio_for_anomalies(
    portfolio_returns: Dict[str, pd.Series]
) -> List[WatchdogAlert]:
    """
    Master scanner that aggregates and sorts alerts across a portfolio.
    """
    alerts = []
    
    for ticker, returns in portfolio_returns.items():
        if returns.empty:
            continue
            
        # Clean returns
        clean_returns = returns.dropna()
        
        # Run detectors
        alerts_found = [
            detect_sharpe_decay(clean_returns, ticker),
            detect_distribution_shift(clean_returns, ticker),
            detect_zscore_breach(clean_returns, ticker)
        ]
        
        alerts.extend([a for a in alerts_found if a is not None])

    # Deterministic sorting by severity then ticker
    alerts.sort(key=lambda x: (SEVERITY_RANK.get(x.severity, 99), x.ticker, x.alert_type))
    
    return alerts
