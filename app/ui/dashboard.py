import os
import sys

# --- Path Stabilization Sentinel (Part 1) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Disable Problematic Hot-Reload (Part 1) ---
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

print("🚀 STABLE IMPORT MODE ACTIVE")

import json
import logging
import time
from dataclasses import asdict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Quant & Data Imports
from app.data.market_data import fetch_ohlcv, NIFTY50_TICKERS
from app.quant.signals import compute_signals, SIGNAL_BUY, SIGNAL_SELL
from app.quant.regime_detector import detect_regimes_hmm
from app.quant.backtester import run_backtest
from app.quant.diagnostics import (
    explain_signals,
    analyze_regime_performance,
    generate_strategy_diagnostics
)
from app.data.portfolio_builder import PORTFOLIOS
from app.quant.portfolio_signals import (
    scan_portfolio_signals,
    generate_portfolio_signal_summary,
    aggregate_sector_signals,
    generate_signal_diagnostics,
    portfolio_signal_reasoning_context
)

# Base Utilities (Safe to keep at top)
from app.utils.helpers import safe_float

# --- Direct Core Imports (Removing shields) ---
from app.evaluation.llm_evaluator import evaluate_response
from app.reasoning.memory_engine import normalize_memory_turn, extract_relevant_memory
from app.reasoning.proactive_engine import (
    generate_proactive_insight,
    get_watchdog_insights,
    watchdog_reasoning_context
)

# --- Lazy-Load Wrappers (Part 3) ---
def get_resolve_context():
    from app.reasoning.context_resolver import resolve_context
    return resolve_context

def get_execute_intents():
    from app.reasoning.router import execute_intents
    return execute_intents

def get_stream_final_response():
    from app.reasoning.response_generator import stream_final_response
    return stream_final_response

def get_polish_response():
    from app.reasoning.response_polisher import polish_response
    return polish_response

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AlphaTrace AI Copilot", page_icon="📊", layout="wide")

# --- Session Initialization ---
if "memory" not in st.session_state: st.session_state.memory = []
if "messages" not in st.session_state: st.session_state.messages = []
if "current_portfolio" not in st.session_state: st.session_state.current_portfolio = "PORTFOLIO_001"
if "last_tool_data" not in st.session_state: st.session_state.last_tool_data = None
if "proactive_metadata" not in st.session_state: st.session_state.proactive_metadata = None
if "last_insight_topic" not in st.session_state: st.session_state.last_insight_topic = None
if "last_insight_turn" not in st.session_state: st.session_state.last_insight_turn = -2
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None
if "watchdog_results" not in st.session_state: st.session_state.watchdog_results = None
if "signal_results" not in st.session_state: st.session_state.signal_results = None

PORTFOLIO_MAPPING = {
    "Rahul Sharma (Diversified)": "PORTFOLIO_001",
    "Priya Patel (Sector Concentrated)": "PORTFOLIO_002",
    "Arun Krishnamurthy (Conservative)": "PORTFOLIO_003"
}

def interpret_conf(c):
    # Fix 4: Safe confidence interpretation
    val = safe_float(c)
    if val > 0.8: return "High"
    if val > 0.6: return "Moderate"
    return "Low"

# --- Sidebar ---
with st.sidebar:
    st.title("📊 AlphaTrace Hub")
    selected_label = st.selectbox(
        "Active Context", 
        options=list(PORTFOLIO_MAPPING.keys()),
        index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0
    )
    
    new_pid = PORTFOLIO_MAPPING[selected_label]
    if new_pid != st.session_state.current_portfolio:
        st.session_state.current_portfolio = new_pid
        st.session_state.memory = []; st.session_state.messages = []
        st.session_state.last_tool_data = None
        st.rerun()

    if st.session_state.last_tool_data:
        st.divider()
        data = st.session_state.last_tool_data
        metrics = {}
        for tool_res in data.values():
            if isinstance(tool_res, dict): metrics.update(tool_res.get("metrics", {}))
        
        # Resolve Tiered Confidence (Parts 1-5)
        conf = data.get("global_metrics", {}).get("confidence")
        if conf is None:
            conf = st.session_state.get("last_confidence", 0.5)
        
        conf = safe_float(conf)
        if conf > 0.75:
            st.metric("Confidence", "High")
        elif conf > 0.5:
            st.metric("Confidence", "Moderate")
        else:
            # Mask low confidence in sidebar
            st.info("Limited signal strength — insights based on available sectors")
        
        exposure = metrics.get("sector_exposure", {})
        if exposure:
            df_exp = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            df_exp["Allocation"] = df_exp["Allocation"].apply(safe_float)
            fig = px.pie(df_exp, values="Allocation", names="Sector", hole=0.4, height=180)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            # Fix 9: Streamlit stretch layout (Part 8)
            st.plotly_chart(fig, width="stretch")

# --- Main Content Tabs ---
tab_ai, tab_backtest, tab_watchdog, tab_signals = st.tabs(["💬 AI Copilot", "📊 Backtest Terminal", "🚨 Watchdog", "🎯 Signals"])

with tab_ai:
    # --- Chat Display ---
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and i == len(st.session_state.messages) - 1 and st.session_state.proactive_metadata:
                meta = st.session_state.proactive_metadata
                if st.button(f"🔍 Analyze signal: {meta['type'].title()}", key="proactive_btn"):
                    st.session_state.last_insight_turn = len(st.session_state.memory)
                    st.session_state.pending_prompt = meta['followup_query']
                    st.rerun()

    user_input = st.chat_input("Analyze portfolio...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_prompt = user_input
        st.rerun()

    # --- Reasoning ---
    if st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        with st.chat_message("assistant"):
            try:
                with st.spinner("Reasoning..."):
                    recent_mem = st.session_state.memory[::-1][:3]
                    session_wrapped = {"current_portfolio": st.session_state.current_portfolio, "memory": recent_mem}
                    
                    resolution = get_resolve_context()(active_prompt, session_wrapped)
                    classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_mem)
                    validation = validate_and_route(resolution["resolved_query"], classification)

                    if validation["action"] != "execute":
                        res_path = validation.get('reason', 'Could you clarify that?')
                        st.markdown(res_path); st.session_state.messages.append({"role": "assistant", "content": res_path})
                    else:
                        intent = validation.get("validated_intent")
                        # (Intent normalization logic preserved)
                        if not intent or intent == ["full_analysis"]:
                            q = active_prompt.lower()
                            if any(x in q for x in ["why", "reason", "cause"]): intent = ["explanation"]
                            elif any(x in q for x in ["compare", "vs", "difference"]): intent = ["comparison"]
                            elif any(x in q for x in ["what should", "advice", "suggest"]): intent = ["advice"]
                            else: intent = ["full_analysis"]
                        
                        execution_results = get_execute_intents()({
                            "intent": intent,
                            "portfolio_id": validation.get("portfolio_id", st.session_state.current_portfolio),
                            "confidence": validation.get("confidence", 0.5)
                        }, {"current_portfolio": st.session_state.current_portfolio})
                        
                        st.session_state.current_portfolio = execution_results["portfolio_id"]
                        tool_data = {res["type"]: res for res in execution_results["results"]}
                        st.session_state.last_tool_data = tool_data

                        # Proactive
                        if (len(st.session_state.memory) - st.session_state.last_insight_turn) >= 2:
                            proactive = generate_proactive_insight(tool_data, active_prompt, st.session_state.memory, st.session_state.last_insight_topic)
                            if proactive:
                                st.session_state.proactive_metadata = proactive
                                st.session_state.last_insight_topic = proactive["topic"]
                                st.session_state.last_insight_turn = len(st.session_state.memory)

                        # Narrative
                        stream_gen = get_stream_final_response()(resolution["resolved_query"], intent, execution_results["portfolio_id"], tool_data, extract_relevant_memory(active_prompt, st.session_state.memory))
                        full_narrative = st.write_stream(stream_gen)
                        
                        conf_val = 0.5
                        if "__CONFIDENCE__:" in full_narrative:
                            parts = full_narrative.split("__CONFIDENCE__:")
                            display_text = parts[0]
                            conf_val = safe_float(parts[1].strip())
                            st.session_state["last_confidence"] = conf_val
                        else:
                            display_text = full_narrative
                            st.session_state["last_confidence"] = conf_val

                        if conf_val > 0.75: st.caption(f"Reasoning Fidelity: :green[High] ({int(conf_val*100)}%)")
                        elif conf_val > 0.5: st.caption(f"Reasoning Fidelity: :orange[Moderate] ({int(conf_val*100)}%)")
                        else: st.info("Limited signal strength — reasoning based on current analytical snapshots.")
                        
                        if st.session_state.proactive_metadata:
                            st.markdown(f"\n\n{st.session_state.proactive_metadata['text']}")
                            display_text += f"\n\n{st.session_state.proactive_metadata['text']}"

                        final_brief = get_polish_response()(display_text, intent, {}, conf_val)
                        memory_obj = normalize_memory_turn(st.session_state.current_portfolio, active_prompt, validation["validated_intent"], final_brief, tool_data)
                        st.session_state.memory.append(memory_obj)
                        st.session_state.messages.append({"role": "assistant", "content": final_brief})
                        st.rerun()

            except Exception as e:
                logger.error(f"Execution Fault: {e}")
                st.error("I've encountered a temporary analytical hurdle. Please re-state your query.")

with tab_backtest:
    st.header("📊 Strategy Backtest Terminal")
    st.caption("Evaluate RSI Mean-Reversion signals on historical data.")
    
    # Context-Aware Ticker Selection
    active_p_data = PORTFOLIOS.get(st.session_state.current_portfolio, {})
    active_tickers = list(active_p_data.get("holdings", {}).keys())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.selectbox(
            "Select Ticker", 
            options=active_tickers + [t for t in NIFTY50_TICKERS if t not in active_tickers],
            help="Tickers from your active portfolio are shown first."
        )
    with col2:
        period = st.select_slider("Period", options=["3mo", "6mo", "1y", "2y"], value="1y")
    with col3:
        st.write("") # Spacer
        run_btn = st.button("🚀 Run Backtest", use_container_width=True)

    if run_btn:
        with st.spinner(f"Backtesting {ticker}..."):
            # Step A: Fetch
            df_raw = fetch_ohlcv(ticker, period=period)
            if df_raw.empty:
                st.error(f"Could not fetch data for {ticker}. Please try again.")
            else:
                # Step B: Signals
                df_signals = compute_signals(df_raw)
                
                # Step C: Backtest
                results = run_backtest(df_signals)
                
                if results["metrics"]["status"] == "success":
                    m = results["metrics"]
                    eq = results["equity_curve"]
                    
                    # 1. Metric Grid
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Return", f"{m['total_return']:.2%}")
                    c2.metric("Buy & Hold", f"{m['buy_hold_return']:.2%}")
                    c3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
                    c4.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Win Rate", f"{m['win_rate']:.2%}")
                    c2.metric("Exposure", f"{m['exposure_ratio']:.2%}")
                    c3.metric("Trades", m['num_trades'])
                    c4.metric("Status", "Stable", delta="Verified")

                    st.divider()
                    
                    # 2. Equity Curve Chart
                    st.subheader("Equity Curve (Compounded)")
                    fig_eq = px.line(
                        eq, 
                        y=["equity_curve", "buy_hold_curve"],
                        labels={"value": "Equity (INR)", "Date": "Timeline", "variable": "Strategy"},
                        color_discrete_map={"equity_curve": "#00FFCC", "buy_hold_curve": "#888888"}
                    )
                    fig_eq.update_layout(template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    # 3. Signal Visualization (Price + Markers)
                    st.subheader("Price & Execution Signals")
                    
                    # Optimized Plotting: Construct compact frames
                    price_trace = go.Scatter(x=eq.index, y=eq["Close"], name="Price", line=dict(color="#FFFFFF", width=1))
                    
                    buy_signals = eq[eq["signal"] == SIGNAL_BUY]
                    sell_signals = eq[eq["signal"] == SIGNAL_SELL]
                    
                    buy_trace = go.Scatter(
                        x=buy_signals.index, y=buy_signals["Close"],
                        mode='markers', name='Buy',
                        marker=dict(symbol='triangle-up', size=10, color='#00FF00')
                    )
                    
                    sell_trace = go.Scatter(
                        x=sell_signals.index, y=sell_signals["Close"],
                        mode='markers', name='Sell',
                        marker=dict(symbol='triangle-down', size=10, color='#FF0000')
                    )
                    
                    fig_sig = go.Figure(data=[price_trace, buy_trace, sell_trace])
                    fig_sig.update_layout(
                        template="plotly_dark", 
                        hovermode="x unified",
                        showlegend=True,
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=400
                    )
                    st.plotly_chart(fig_sig, use_container_width=True)

                    # Step D: Diagnostics & Explainability
                    st.divider()
                    with st.spinner("Generating strategy diagnostics..."):
                        # Use Nifty for general market regime context
                        nifty_df = fetch_ohlcv("^NSEI", period=period)
                        reg_df = detect_regimes_hmm(nifty_df)
                        
                        explanations = explain_signals(df_signals, reg_df)
                        regime_perf = analyze_regime_performance(eq, reg_df)
                        diagnostics = generate_strategy_diagnostics(results, regime_perf, explanations)
                        
                        st.subheader("🔍 Strategy Diagnostics & Explainability")
                        st.info(diagnostics["summary"])
                        st.write(f"**💡 Insight:** {diagnostics['latest_signal_explanation']}")
                        
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            st.markdown("**Strengths**")
                            for s in diagnostics["strengths"]: st.success(s)
                        with dcol2:
                            st.markdown("**Potential Weaknesses**")
                            for w in diagnostics["weaknesses"]: st.warning(w)
                        
                        st.markdown("**Regime Performance Breakdown**")
                        if not regime_perf.empty:
                            display_perf = regime_perf.copy()
                            display_perf["avg_return"] = display_perf["avg_return"].apply(lambda x: f"{x:+.2%}")
                            display_perf["volatility"] = display_perf["volatility"].apply(lambda x: f"{x:.2%}")
                            display_perf["sharpe"] = display_perf["sharpe"].apply(lambda x: f"{x:.2f}")
                            display_perf["win_rate"] = display_perf["win_rate"].apply(lambda x: f"{x:.1%}")
                            display_perf["exposure"] = display_perf["exposure"].apply(lambda x: f"{x:.1%}")
                            st.table(display_perf)
                        else:
                            st.write("Insufficient regime data for breakdown.")
                else:
                    st.error("Backtest failed to produce valid results.")

with tab_watchdog:
    st.header("🚨 Strategy Watchdog Engine")
    st.caption("Operational strategy-health monitoring and statistical anomaly detection.")
    
    # 1. Monitoring Controls
    wcol1, wcol2, wcol3 = st.columns([2, 1, 1])
    with wcol1:
        selected_portfolio_label = st.selectbox(
            "Select Portfolio to Scan",
            options=list(PORTFOLIO_MAPPING.keys()),
            index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0,
            key="watchdog_portfolio_select"
        )
        selected_pid = PORTFOLIO_MAPPING[selected_portfolio_label]
    
    with wcol2:
        monitor_window = st.selectbox(
            "Monitoring Window",
            options=["30d", "90d", "1y"],
            index=2,
            help="Window for statistical comparison (recent vs trailing)."
        )
    
    with wcol3:
        st.write("") # Spacer
        scan_btn = st.button("🔍 Run Health Scan", use_container_width=True)

    # 2. Scan Execution
    if scan_btn:
        with st.spinner("Scanning portfolio for statistical anomalies..."):
            try:
                portfolio_data = PORTFOLIOS.get(selected_pid, {})
                tickers = list(portfolio_data.get("holdings", {}).keys())
                
                if not tickers:
                    st.warning("No tickers found in the selected portfolio.")
                else:
                    portfolio_returns = {}
                    for t in tickers:
                        # Use a 2y period to ensure enough trailing history for 1y monitor
                        raw_df = fetch_ohlcv(t, period="2y")
                        if not raw_df.empty:
                            portfolio_returns[t] = raw_df["Close"].pct_change().dropna()
                    
                    if portfolio_returns:
                        # Step 1: Run Watchdog Detectors & Proactive Aggregation
                        insights = get_watchdog_insights(portfolio_returns)
                        st.session_state.watchdog_results = insights
                    else:
                        st.error("Could not fetch return history for portfolio tickers.")
            except Exception as e:
                logger.error(f"Watchdog scan failure: {e}")
                st.error("Failed to complete health scan. Check logs for details.")

    # 3. Rendering Results
    if st.session_state.watchdog_results:
        res = st.session_state.watchdog_results
        
        # Operational Status Header
        st.divider()
        status = res["status"]
        if status == "STABLE":
            st.success(f"🟢 STATUS: {status}")
        elif status == "WATCH":
            st.warning(f"🟠 STATUS: {status}")
        elif status in ["DEGRADED", "CRITICAL"]:
            st.error(f"🔴 STATUS: {status}")
            
        # Alert Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Critical Alerts", len(res["critical_alerts"]))
        m2.metric("High Alerts", len(res["high_alerts"]))
        m3.metric("Total Anomalies", len(res["critical_alerts"]) + len(res["high_alerts"]) + len(res["medium_alerts"]))
        m4.metric("Risk Profile", status.title())

        # Operational Summary & Suggested Actions
        st.info(f"**Operational Summary:** {res['summary']}")
        
        if res["suggested_actions"]:
            with st.expander("💡 Suggested Investigations", expanded=True):
                for action in res["suggested_actions"]:
                    st.write(f"- {action}")

        # Alert Table
        st.subheader("📋 Active Statistical Alerts")
        all_alerts = res["critical_alerts"] + res["high_alerts"] + res["medium_alerts"] + res["low_alerts"]
        if all_alerts:
            alert_df = pd.DataFrame(all_alerts)
            # Reorder for UI
            cols = ["ticker", "alert_type", "severity", "metric_value", "threshold", "triggered_at"]
            st.dataframe(alert_df[cols], use_container_width=True, hide_index=True)
        else:
            st.write("No active alerts detected.")

        # Rolling Sharpe Visualization
        if res["critical_alerts"] or res["high_alerts"]:
            st.divider()
            st.subheader("📉 Strategy Decay Visualization")
            
            # Find the top risk ticker
            top_risk_alert = (res["critical_alerts"] + res["high_alerts"])[0]
            target_ticker = top_risk_alert["ticker"]
            
            # Fetch data for plotting
            plot_df = fetch_ohlcv(target_ticker, period="1y")
            if not plot_df.empty:
                returns = plot_df["Close"].pct_change().dropna()
                # Simple rolling sharpe
                rolling_sharpe = (returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
                
                fig_decay = go.Figure()
                fig_decay.add_trace(go.Scatter(
                    x=rolling_sharpe.index, y=rolling_sharpe,
                    name="Rolling 20d Sharpe",
                    line=dict(color="#00FFCC", width=2)
                ))
                # Add baseline
                baseline = (returns.mean() / returns.std()) * np.sqrt(252)
                fig_decay.add_hline(y=baseline, line_dash="dash", line_color="#888888", annotation_text="Trailing Baseline")
                
                fig_decay.update_layout(
                    title=f"Sharpe Decay Analysis: {target_ticker}",
                    template="plotly_dark",
                    yaxis_title="Sharpe Ratio",
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_decay, use_container_width=True)

        # AI Escalation Context Preview
        with st.expander("🤖 AI Escalation Context (Internal View)"):
            st.caption("This is the structured operational context provided to the reasoning engine.")
            ctx = watchdog_reasoning_context(res)
            st.code(ctx, language="text")
    else:
        st.info("Select a portfolio and run a health scan to evaluate operational risks.")

with tab_signals:
    st.header("🎯 Live Signal Intelligence")
    st.caption("Context-aware portfolio signal intelligence and actionable opportunity surfacing.")
    
    # 1. Signal Scan Controls
    scol1, scol2, scol3 = st.columns([2, 1, 1])
    with scol1:
        signal_p_label = st.selectbox(
            "Portfolio for Signal Scan",
            options=list(PORTFOLIO_MAPPING.keys()),
            index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0,
            key="signal_portfolio_select"
        )
        signal_pid = PORTFOLIO_MAPPING[signal_p_label]
        
    with scol2:
        signal_window = st.selectbox(
            "Indicator Lookback",
            options=["3mo", "6mo", "1y", "2y"],
            index=1,
            help="Window for calculating technical indicators (RSI, MACD, etc.)."
        )
        
    with scol3:
        st.write("") # Spacer
        sig_scan_btn = st.button("🚀 Generate Signals", use_container_width=True)

    # 2. Pipeline Execution
    if sig_scan_btn:
        with st.spinner("Generating structured portfolio signals..."):
            try:
                p_data = PORTFOLIOS.get(signal_pid, {})
                tickers = list(p_data.get("holdings", {}).keys())
                
                if not tickers:
                    st.warning("No tickers found in the selected portfolio.")
                else:
                    ticker_data = {}
                    for t in tickers:
                        raw_df = fetch_ohlcv(t, period=signal_window)
                        if not raw_df.empty:
                            ticker_data[t] = compute_signals(raw_df)
                    
                    if ticker_data:
                        # Run Portfolio Intelligence Pipeline
                        signals = scan_portfolio_signals(ticker_data)
                        summary = generate_portfolio_signal_summary(signals)
                        sector_agg = aggregate_sector_signals(signals)
                        diagnostics = generate_signal_diagnostics(signals, summary, sector_agg)
                        
                        st.session_state.signal_results = {
                            "signals": [asdict(s) for s in signals],
                            "summary": asdict(summary),
                            "sector_agg": sector_agg,
                            "diagnostics": diagnostics
                        }
                    else:
                        st.error("Could not fetch market data for portfolio signals.")
            except Exception as e:
                logger.error(f"Signal scan failure: {e}")
                st.error("Failed to generate signals. Check logs for details.")

    # 3. Rendering Results
    if st.session_state.signal_results:
        res = st.session_state.signal_results
        sum_data = res["summary"]
        
        # Portfolio Bias Header
        st.divider()
        bias = sum_data["market_bias"]
        if bias == "BULLISH":
            st.success(f"🟢 Portfolio Bias: {bias}")
        elif bias == "BEARISH":
            st.error(f"🔴 Portfolio Bias: {bias}")
        elif bias == "MIXED":
            st.warning(f"🟠 Portfolio Bias: {bias}")
        else:
            st.info(f"⚪ Portfolio Bias: {bias}")
            
        # Signal Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Confidence", f"{sum_data['average_confidence']:.2%}", help="Average confidence across all tickers.")
        m2.metric("LONG Signals", sum_data["long_signals"])
        m3.metric("SHORT Signals", sum_data["short_signals"])
        m4.metric("Active Scan", f"{sum_data['total_signals']} Tickers")

        # Opportunity Cards (Top 3)
        st.subheader("💡 Top Opportunity Concentration")
        valid_signals = [s for s in res["signals"] if s["direction"] != "NEUTRAL"]
        if valid_signals:
            cols = st.columns(min(len(valid_signals), 3))
            for i, sig in enumerate(valid_signals[:3]):
                with cols[i]:
                    color = "green" if sig["direction"] == "LONG" else "red"
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; border:1px solid #444; background-color:#1e1e1e;">
                        <h3 style="margin-top:0; color:{color};">{sig['ticker']}</h3>
                        <p style="font-size:0.9em; color:#aaa;">{sig['direction']} | {sig['signal_strength']}</p>
                        <p style="font-size:1.1em; font-weight:bold;">{sig['confidence']:.0%} Confidence</p>
                        <p style="font-size:0.85em; margin-bottom:10px;">{sig['causal_reason'][:100]}...</p>
                        <div style="font-family:monospace; font-size:0.8em; color:#00FFCC;">
                            TP: {sig['take_profit']:.2f}<br>
                            SL: {sig['stop_loss']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No high-confidence directional signals detected.")

        # Sector Conviction Visualization
        st.divider()
        st.subheader("🏢 Sector Conviction & Bias")
        sector_df = pd.DataFrame.from_dict(res["sector_agg"], orient='index').reset_index()
        sector_df.rename(columns={'index': 'Sector'}, inplace=True)
        
        if not sector_df.empty:
            fig_sector = px.bar(
                sector_df, 
                x="Sector", y="avg_confidence", color="bias",
                color_discrete_map={"Bullish": "#00FF00", "Bearish": "#FF0000", "Neutral": "#888888"},
                title="Average Signal Confidence by Sector",
                labels={"avg_confidence": "Avg Confidence", "bias": "Dominant Bias"}
            )
            fig_sector.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Signal Table & Diagnostics
        st.divider()
        dcol1, dcol2 = st.columns([2, 1])
        with dcol1:
            st.subheader("📋 Full Signal Inventory")
            all_sig_df = pd.DataFrame(res["signals"])
            # Reorder for UI
            table_cols = ["ticker", "direction", "confidence", "signal_strength", "entry_price", "stop_loss", "take_profit"]
            st.dataframe(all_sig_df[table_cols], use_container_width=True, hide_index=True)
        
        with dcol2:
            st.subheader("🔍 Signal Diagnostics")
            for diag in res["diagnostics"]:
                st.info(diag)
                
        # AI Opportunity Context Preview
        with st.expander("🤖 AI Opportunity Context (Internal View)"):
            st.caption("This is the structured intelligence provided to the reasoning layer for cross-asset analysis.")
            # Convert back to TradingSignal objects for the helper
            from app.quant.signal_generator import TradingSignal
            from app.quant.portfolio_signals import PortfolioSignalSummary
            
            mock_sigs = [TradingSignal(**s) for s in res["signals"]]
            mock_sum = PortfolioSignalSummary(**sum_data)
            ctx = portfolio_signal_reasoning_context(mock_sigs, mock_sum, res["diagnostics"])
            st.code(ctx, language="text")
    else:
        st.info("Select a portfolio and generate signals to identify current market opportunities.")
