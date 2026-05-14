import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from app.quant.regime_detector import REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS

logger = logging.getLogger(__name__)

@dataclass
class StrategyEvaluation:
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: float
    regime_performance: Dict[str, float] # Regime -> Return

@dataclass
class RLDiagnostics:
    action_distribution: Dict[int, float] # Action -> Ratio
    reward_volatility: float
    reward_skewness: float
    exploration_entropy: float
    overtrading_ratio: float # Trades per 100 steps
    hold_dominance: float

@dataclass
class BenchmarkComparison:
    ppo_metrics: StrategyEvaluation
    rsi_metrics: StrategyEvaluation
    bh_metrics: StrategyEvaluation
    random_metrics: StrategyEvaluation
    summary: str

def compute_metrics(returns: pd.Series, name: str, regime_series: Optional[pd.Series] = None) -> StrategyEvaluation:
    """
    Compute standard performance metrics for a return series.
    """
    if returns.empty:
        return StrategyEvaluation(name, 0, 0, 0, 0, 0, {})

    cum_ret = (1 + returns).prod() - 1
    ann_ret = (1 + cum_ret) ** (252 / len(returns)) - 1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    # Drawdown
    cum_equity = (1 + returns).cumprod()
    peak = cum_equity.expanding(min_periods=1).max()
    dd = (cum_equity / peak) - 1
    max_dd = float(dd.min())
    
    # Regime-aware returns
    regime_perf = {}
    if regime_series is not None:
        # Align returns and regimes
        common_idx = returns.index.intersection(regime_series.index)
        if not common_idx.empty:
            aligned_rets = returns.loc[common_idx]
            aligned_regs = regime_series.loc[common_idx]
            for r in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
                mask = (aligned_regs == r)
                if mask.any():
                    regime_perf[r] = float((1 + aligned_rets[mask]).prod() - 1)
                else:
                    regime_perf[r] = 0.0

    return StrategyEvaluation(
        name=name,
        total_return=float(cum_ret),
        annualized_return=float(ann_ret),
        sharpe_ratio=float(sharpe),
        max_drawdown=max_dd,
        total_trades=0, # To be filled by caller
        regime_performance=regime_perf
    )

def evaluate_research_pipeline(
    ppo_results: Dict[str, Any],
    rsi_results: Dict[str, Any],
    bh_results: Dict[str, Any],
    regimes: pd.Series
) -> BenchmarkComparison:
    """
    Aggregate comparative analytics for RL vs Deterministic vs B&H.
    """
    ppo_eval = compute_metrics(ppo_results["returns"], "PPO_Agent", regimes)
    ppo_eval.total_trades = ppo_results.get("total_trades", 0)
    
    rsi_eval = compute_metrics(rsi_results["returns"], "RSI_Strategy", regimes)
    rsi_eval.total_trades = rsi_results.get("total_trades", 0)
    
    bh_eval = compute_metrics(bh_results["returns"], "Buy_Hold", regimes)
    
    # Dummy random agent for context
    random_rets = pd.Series(np.random.normal(0, 0.01, len(ppo_results["returns"])), index=ppo_results["returns"].index)
    random_eval = compute_metrics(random_rets, "Random_Agent", regimes)

    summary = generate_research_summary(ppo_eval, rsi_eval, bh_eval)

    return BenchmarkComparison(
        ppo_metrics=ppo_eval,
        rsi_metrics=rsi_eval,
        bh_metrics=bh_eval,
        random_metrics=random_eval,
        summary=summary
    )

def generate_research_summary(ppo: StrategyEvaluation, rsi: StrategyEvaluation, bh: StrategyEvaluation) -> str:
    """
    Deterministic synthesis of comparative findings.
    """
    lines = []
    
    # 1. Performance Rank
    results = [ppo, rsi, bh]
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    lines.append(f"Comparative Leader: {results[0].name} (Sharpe: {results[0].sharpe_ratio:.2f})")
    
    # 2. Stability Analysis
    if ppo.max_drawdown < rsi.max_drawdown:
        lines.append("PPO demonstrated unexpectedly higher drawdown protection than the deterministic RSI strategy.")
    else:
        lines.append("Deterministic RSI signals maintained superior stability and lower drawdown than the PPO agent.")
        
    # 3. Regime Sensitivity
    bear_ppo = ppo.regime_performance.get(REGIME_BEAR, 0)
    bear_rsi = rsi.regime_performance.get(REGIME_BEAR, 0)
    if bear_ppo < bear_rsi and bear_ppo < 0:
        lines.append(f"PPO policy exhibited significant instability during {REGIME_BEAR} regimes, underperforming fixed rules.")
    
    # 4. Overtrading
    if ppo.total_trades > rsi.total_trades * 2:
        lines.append(f"PPO agent demonstrated overtrading tendency ({ppo.total_trades} trades) relative to RSI consensus.")

    return " ".join(lines)

def plot_comparative_equity(
    ppo_equity: pd.Series, 
    rsi_equity: pd.Series, 
    bh_equity: pd.Series,
    save_path: str
):
    """
    Lightweight research plotting for equity curve comparison.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_equity, label="PPO (RL)", color="blue")
    plt.plot(rsi_equity, label="RSI (Deterministic)", color="orange")
    plt.plot(bh_equity, label="B&H (Baseline)", color="gray", linestyle="--")
    plt.title("Comparative Research: RL Policy vs Deterministic Signals")
    plt.xlabel("Days")
    plt.ylabel("Normalized Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
