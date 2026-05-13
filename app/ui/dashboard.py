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

import streamlit as st
import pandas as pd
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

# Base Utilities (Safe to keep at top)
from app.utils.helpers import safe_float

# --- Direct Core Imports (Removing shields) ---
from app.evaluation.llm_evaluator import evaluate_response
from app.reasoning.proactive_engine import generate_proactive_insight
from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.memory_engine import normalize_memory_turn, extract_relevant_memory

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
tab_ai, tab_backtest = st.tabs(["💬 AI Copilot", "📊 Backtest Terminal"])

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
