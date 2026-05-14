import logging
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Literal
from app.quant.signal_generator import TradingSignal, generate_trading_signal
from app.data.market_data import SECTOR_MAP

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class PortfolioSignalSummary:
    total_signals: int
    long_signals: int
    short_signals: int
    neutral_signals: int
    top_signal: Optional[str]
    average_confidence: float
    market_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]
    generated_at: str

def scan_portfolio_signals(
    ticker_data: Dict[str, pd.DataFrame],
    regimes: Optional[Dict[str, str]] = None
) -> List[TradingSignal]:
    """
    Generate structured signals for every ticker in the portfolio.
    """
    signals = []
    regimes = regimes or {}
    
    # Sort tickers to ensure deterministic processing order
    for ticker in sorted(ticker_data.keys()):
        df = ticker_data[ticker]
        if df.empty or len(df) < 2:
            logger.debug(f"Skipping signal generation for {ticker}: insufficient data.")
            continue
            
        regime = regimes.get(ticker)
        try:
            signal = generate_trading_signal(df, regime=regime, ticker=ticker)
            signals.append(signal)
        except Exception as e:
            logger.error(f"Failed to generate signal for {ticker}: {e}")

    # Confidence-Aware Ranking (Highest confidence first, deterministic tie-break by ticker)
    signals.sort(key=lambda x: (x.direction != "NEUTRAL", x.confidence, x.ticker), reverse=True)
    
    return signals

def generate_portfolio_signal_summary(signals: List[TradingSignal]) -> PortfolioSignalSummary:
    """
    Aggregate portfolio-level intelligence from a list of signals.
    """
    total = len(signals)
    longs = [s for s in signals if s.direction == "LONG"]
    shorts = [s for s in signals if s.direction == "SHORT"]
    neutrals = [s for s in signals if s.direction == "NEUTRAL"]
    
    avg_conf = sum(s.confidence for s in signals) / total if total > 0 else 0.0
    
    # Market Bias Logic
    bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"] = "NEUTRAL"
    if total > 0:
        long_ratio = len(longs) / total
        short_ratio = len(shorts) / total
        
        if long_ratio > 0.4 and short_ratio < 0.2:
            bias = "BULLISH"
        elif short_ratio > 0.4 and long_ratio < 0.2:
            bias = "BEARISH"
        elif long_ratio > 0.2 and short_ratio > 0.2:
            bias = "MIXED"
        else:
            bias = "NEUTRAL"

    top_signal = None
    if signals and signals[0].direction != "NEUTRAL":
        top_signal = f"{signals[0].ticker} ({signals[0].signal_strength})"

    return PortfolioSignalSummary(
        total_signals=total,
        long_signals=len(longs),
        short_signals=len(shorts),
        neutral_signals=len(neutrals),
        top_signal=top_signal,
        average_confidence=float(avg_conf),
        market_bias=bias,
        generated_at=signals[0].generated_at if signals else "N/A"
    )

def aggregate_sector_signals(signals: List[TradingSignal]) -> Dict[str, Dict]:
    """
    Compute sector-level signal concentration and confidence.
    """
    sector_agg = {}
    
    for s in signals:
        sector = SECTOR_MAP.get(s.ticker, "Other")
        if sector not in sector_agg:
            sector_agg[sector] = {"longs": 0, "shorts": 0, "total": 0, "conf_sum": 0.0}
            
        sector_agg[sector]["total"] += 1
        sector_agg[sector]["conf_sum"] += s.confidence
        if s.direction == "LONG":
            sector_agg[sector]["longs"] += 1
        elif s.direction == "SHORT":
            sector_agg[sector]["shorts"] += 1
            
    # Calculate averages and dominant bias per sector
    results = {}
    for sector, data in sector_agg.items():
        avg_conf = data["conf_sum"] / data["total"]
        bias = "Neutral"
        if data["longs"] > data["shorts"]: bias = "Bullish"
        elif data["shorts"] > data["longs"]: bias = "Bearish"
        
        results[sector] = {
            "avg_confidence": float(avg_conf),
            "bias": bias,
            "count": data["total"],
            "signal_ratio": (data["longs"] + data["shorts"]) / data["total"]
        }
        
    return results

def generate_signal_diagnostics(
    signals: List[TradingSignal], 
    summary: PortfolioSignalSummary,
    sector_agg: Dict[str, Dict]
) -> List[str]:
    """
    Generate analytical diagnostics about the portfolio signal structure.
    """
    diagnostics = []
    
    # 1. Concentration Analysis
    top_sectors = sorted(sector_agg.items(), key=lambda x: x[1]["count"], reverse=True)
    if top_sectors:
        main_sector = top_sectors[0][0]
        diagnostics.append(f"Signal concentration is currently highest in the {main_sector} sector.")
        
    # 2. Bias Analysis
    if summary.market_bias == "MIXED":
        diagnostics.append("Portfolio lacks a strong directional consensus; indicators show conflicting trends across assets.")
    elif summary.market_bias == "BULLISH":
        diagnostics.append(f"Broad bullish consensus detected across {summary.long_signals} assets.")
        
    # 3. Opportunity Zones (Clustering)
    bullish_sectors = [s for s, d in sector_agg.items() if d["bias"] == "Bullish" and d["avg_confidence"] > 0.6]
    if bullish_sectors:
        diagnostics.append(f"High-confidence LONG cluster detected in {', '.join(bullish_sectors)}.")
        
    bearish_sectors = [s for s, d in sector_agg.items() if d["bias"] == "Bearish" and d["avg_confidence"] > 0.6]
    if bearish_sectors:
        diagnostics.append(f"High-confidence SHORT cluster detected in {', '.join(bearish_sectors)}.")

    return diagnostics

def portfolio_signal_reasoning_context(
    signals: List[TradingSignal],
    summary: PortfolioSignalSummary,
    diagnostics: List[str]
) -> str:
    """
    Format portfolio signal intelligence for the AI reasoning layer.
    """
    lines = [
        f"Portfolio Signal Intelligence: {summary.market_bias} bias detected.",
        f"Summary: {summary.long_signals} LONG, {summary.short_signals} SHORT out of {summary.total_signals} tickers.",
        f"Average Confidence: {summary.average_confidence:.2f}"
    ]
    
    if summary.top_signal:
        lines.append(f"Top Opportunity: {summary.top_signal}")
        
    if diagnostics:
        lines.append(f"Diagnostics: {' '.join(diagnostics[:3])}")
        
    return "\n".join(lines)
