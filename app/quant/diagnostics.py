import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# Initialize logger
logger = logging.getLogger(__name__)

# Schema Contracts
REGIME_PERFORMANCE_COLUMNS = [
    "regime", "avg_return", "volatility", "sharpe", "win_rate", "num_periods", "exposure"
]

SIGNAL_EXPLANATION_COLUMNS = [
    "Close", "rsi", "signal", "signal_reason", "regime"
]

def explain_signals(
    signals_df: pd.DataFrame,
    regime_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate deterministic explanations for trading signals aligned with market regimes.
    """
    if signals_df.empty or regime_df.empty:
        return pd.DataFrame(columns=SIGNAL_EXPLANATION_COLUMNS)

    try:
        # 1. Alignment
        # Merge on index to get regime for each signal
        merged = pd.merge(
            signals_df[["Close", "rsi", "signal"]],
            regime_df[["regime"]],
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        # 2. Explanation Logic
        def get_reason(row):
            sig = row["signal"]
            reg = row["regime"]
            rsi = row["rsi"]
            
            if sig == 1: # Buy
                return f"RSI ({rsi:.1f}) entered oversold territory during {reg} regime."
            elif sig == -1: # Sell
                return f"RSI ({rsi:.1f}) entered overbought territory during {reg} regime."
            return "Hold"

        merged["signal_reason"] = merged.apply(get_reason, axis=1)
        
        return merged[SIGNAL_EXPLANATION_COLUMNS].copy()

    except Exception as e:
        logger.error(f"Error explaining signals: {e}")
        return pd.DataFrame(columns=SIGNAL_EXPLANATION_COLUMNS)

def analyze_regime_performance(
    backtest_df: pd.DataFrame,
    regime_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze strategy performance metrics broken down by market regime.
    """
    if backtest_df.empty or regime_df.empty:
        return pd.DataFrame(columns=REGIME_PERFORMANCE_COLUMNS)

    try:
        # Align backtest results with regimes
        merged = pd.merge(
            backtest_df[["strategy_return"]],
            regime_df[["regime"]],
            left_index=True,
            right_index=True,
            how="inner"
        )
        
        total_periods = len(merged)
        regime_stats = []
        
        for regime_name in merged["regime"].unique():
            subset = merged[merged["regime"] == regime_name]
            n = len(subset)
            if n == 0: continue
            
            avg_ret = subset["strategy_return"].mean()
            vol = subset["strategy_return"].std()
            sharpe = 0.0
            if vol > 0:
                sharpe = (avg_ret / vol) * np.sqrt(252)
                
            wins = (subset["strategy_return"] > 0).sum()
            win_rate = wins / n
            exposure = n / total_periods
            
            regime_stats.append({
                "regime": regime_name,
                "avg_return": float(avg_ret),
                "volatility": float(vol),
                "sharpe": float(sharpe),
                "win_rate": float(win_rate),
                "num_periods": int(n),
                "exposure": float(exposure)
            })
            
        return pd.DataFrame(regime_stats)[REGIME_PERFORMANCE_COLUMNS]

    except Exception as e:
        logger.error(f"Error analyzing regime performance: {e}")
        return pd.DataFrame(columns=REGIME_PERFORMANCE_COLUMNS)

def detect_strategy_weaknesses(regime_perf_df: pd.DataFrame) -> List[str]:
    """
    Identify deterministic strategy weaknesses based on regime performance.
    """
    weaknesses = []
    if regime_perf_df.empty: return ["Insufficient data for weakness detection."]

    for _, row in regime_perf_df.iterrows():
        reg = row["regime"]
        sharpe = row["sharpe"]
        win_rate = row["win_rate"]
        
        if sharpe < 0:
            weaknesses.append(f"Strategy exhibits negative risk-adjusted returns (Sharpe: {sharpe:.2f}) during {reg} regimes.")
        
        if win_rate < 0.4 and row["num_periods"] > 10:
            weaknesses.append(f"Low win rate ({win_rate:.1%}) observed in {reg} market states.")

    if not weaknesses:
        weaknesses.append("No significant structural weaknesses detected in current sample.")
        
    return weaknesses

def generate_strategy_diagnostics(
    backtest_results: Dict[str, Any],
    regime_perf_df: pd.DataFrame,
    explanations_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Synthesize all diagnostic data into a professional research summary.
    """
    try:
        # Extract Strengths
        strengths = []
        best_regime = regime_perf_df.sort_values("sharpe", ascending=False).iloc[0] if not regime_perf_df.empty else None
        if best_regime is not None and best_regime["sharpe"] > 0.5:
            strengths.append(f"Strongest performance in {best_regime['regime']} regimes (Sharpe: {best_regime['sharpe']:.2f}).")
        
        # Detect Weaknesses
        weaknesses = detect_strategy_weaknesses(regime_perf_df)
        
        # Latest Signal Explanation
        latest_explanation = "No recent signals."
        if not explanations_df.empty:
            last_sig_row = explanations_df[explanations_df["signal"] != 0].tail(1)
            if not last_sig_row.empty:
                date_str = last_sig_row.index[0].strftime('%Y-%m-%d')
                latest_explanation = f"Last Signal ({date_str}): {last_sig_row['signal_reason'].iloc[0]}"

        # Narrative Summary
        total_ret = backtest_results.get("metrics", {}).get("total_return", 0.0)
        summary = f"Strategy diagnostics confirm a total return of {total_ret:.2%}. "
        if best_regime is not None:
            summary += f"Primary edge detected in {best_regime['regime']} states with {best_regime['exposure']:.1%} market exposure."

        return {
            "regime_performance": regime_perf_df.to_dict(orient="records"),
            "strengths": strengths if strengths else ["Performance metrics remain within baseline parameters."],
            "weaknesses": weaknesses,
            "latest_signal_explanation": latest_explanation,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating strategy diagnostics: {e}")
        return {
            "regime_performance": [],
            "strengths": [],
            "weaknesses": ["Diagnostic generation failed."],
            "latest_signal_explanation": "Unknown",
            "summary": "Internal error in diagnostics layer."
        }
