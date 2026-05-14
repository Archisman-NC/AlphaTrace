import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    ticker: str
    direction: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: float
    signal_strength: Literal["WEAK", "MODERATE", "STRONG"]
    causal_reason: str
    regime: Optional[str]
    entry_price: float
    stop_loss: float
    take_profit: float
    generated_at: str

def generate_trading_signal(
    df: pd.DataFrame,
    regime: Optional[str] = None,
    ticker: str = "UNKNOWN"
) -> TradingSignal:
    """
    Generate a structured trading signal from the latest market state.
    """
    if df.empty or len(df) < 2:
        logger.warning(f"Insufficient data for signal generation: {ticker}")
        return _get_neutral_fallback(ticker, regime)

    try:
        # Get latest data point
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        close = latest["Close"]
        rsi = latest.get("rsi", 50.0)
        macd = latest.get("macd", 0.0)
        macd_signal = latest.get("macd_signal", 0.0)
        bb_upper = latest.get("bb_upper", close * 1.02)
        bb_lower = latest.get("bb_lower", close * 0.98)
        
        # Calculate MACD momentum
        macd_improving = (macd > macd_signal) and (macd - macd_signal > prev.get("macd", 0) - prev.get("macd_signal", 0))
        macd_weakening = (macd < macd_signal) and (macd - macd_signal < prev.get("macd", 0) - prev.get("macd_signal", 0))
        
        # Calculate Bollinger %B
        bb_range = bb_upper - bb_lower
        bb_pct = (close - bb_lower) / bb_range if bb_range != 0 else 0.5

        # 1. Directional Logic
        direction: Literal["LONG", "SHORT", "NEUTRAL"] = "NEUTRAL"
        reasons = []
        
        # LONG: Oversold + Improving Momentum + Near Lower Band
        if rsi < 35 and macd_improving and bb_pct < 0.2:
            direction = "LONG"
            reasons.append(f"RSI entered oversold territory ({rsi:.1f})")
            reasons.append("MACD shows improving bullish momentum")
            reasons.append(f"Price is testing lower Bollinger support (%B: {bb_pct:.1%})")
            
        # SHORT: Overbought + Weakening Momentum + Near Upper Band
        elif rsi > 65 and macd_weakening and bb_pct > 0.8:
            direction = "SHORT"
            reasons.append(f"RSI entered overbought territory ({rsi:.1f})")
            reasons.append("MACD shows weakening momentum")
            reasons.append(f"Price is testing upper Bollinger resistance (%B: {bb_pct:.1%})")
            
        else:
            direction = "NEUTRAL"
            reasons.append("Indicators show mixed signals or lack of clear directional edge")

        # 2. Confidence Scoring (0.0 -> 1.0)
        confidence = 0.0
        if direction != "NEUTRAL":
            # RSI Component (higher confidence as it gets more extreme)
            rsi_conf = abs(50 - rsi) / 50.0 
            
            # BB Component (higher confidence as it gets closer to edges)
            bb_conf = abs(0.5 - bb_pct) * 2.0
            
            # MACD Component (simple spread)
            macd_conf = min(abs(macd - macd_signal) / (close * 0.01), 1.0) if close != 0 else 0.0
            
            confidence = (rsi_conf * 0.4) + (bb_conf * 0.4) + (macd_conf * 0.2)
            confidence = min(max(confidence, 0.0), 1.0)
        else:
            confidence = 0.5 # Baseline for neutral

        # 3. Signal Strength Classification
        strength: Literal["WEAK", "MODERATE", "STRONG"] = "WEAK"
        if direction != "NEUTRAL":
            if confidence > 0.75: strength = "STRONG"
            elif confidence > 0.5: strength = "MODERATE"
        else:
            strength = "WEAK"

        # 4. Causal Explanation
        causal_reason = " ".join(reasons)
        if regime:
            causal_reason = f"[{regime} Regime] {causal_reason}"

        # 5. ATR-like Risk Framing (Simple Volatility Proxy)
        # Use BB range as a proxy for volatility if ATR isn't in columns
        vol_proxy = bb_range if bb_range > 0 else close * 0.02
        
        if direction == "LONG":
            stop_loss = close - (vol_proxy * 1.5)
            take_profit = close + (vol_proxy * 3.0)
        elif direction == "SHORT":
            stop_loss = close + (vol_proxy * 1.5)
            take_profit = close - (vol_proxy * 3.0)
        else:
            stop_loss = close * 0.95
            take_profit = close * 1.05

        return TradingSignal(
            ticker=ticker,
            direction=direction,
            confidence=float(confidence),
            signal_strength=strength,
            causal_reason=causal_reason,
            regime=regime,
            entry_price=float(close),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            generated_at=str(df.index[-1])
        )

    except Exception as e:
        logger.error(f"Error generating trading signal for {ticker}: {e}")
        return _get_neutral_fallback(ticker, regime)

def _get_neutral_fallback(ticker: str, regime: Optional[str]) -> TradingSignal:
    return TradingSignal(
        ticker=ticker,
        direction="NEUTRAL",
        confidence=0.0,
        signal_strength="WEAK",
        causal_reason="Insufficient data or calculation error.",
        regime=regime,
        entry_price=0.0,
        stop_loss=0.0,
        take_profit=0.0,
        generated_at="N/A"
    )

def generate_portfolio_signals(
    ticker_data: Dict[str, pd.DataFrame],
    regimes: Optional[Dict[str, str]] = None
) -> List[TradingSignal]:
    """
    Generate structured signals across a portfolio, sorted by confidence.
    """
    signals = []
    regimes = regimes or {}
    
    for ticker, df in ticker_data.items():
        regime = regimes.get(ticker)
        sig = generate_trading_signal(df, regime=regime, ticker=ticker)
        signals.append(sig)
        
    # Sort by confidence (highest first), then by strength
    signals.sort(key=lambda x: (x.direction != "NEUTRAL", x.confidence), reverse=True)
    
    return signals
