import streamlit as st
import os
import json
import logging
import time
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Import Reasoning Stack
from app.reasoning.context_resolver import resolve_context
from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.router import execute_intents
from app.reasoning.response_generator import stream_advisory_response
from app.reasoning.response_polisher import polish_response
from app.reasoning.memory_engine import normalize_memory_turn, extract_relevant_memory

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AlphaTrace AI Copilot", page_icon="📊", layout="wide")

# --- Session Initialization ---
if "memory" not in st.session_state:
    st.session_state.memory = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = "PORTFOLIO_001"

if "last_tool_data" not in st.session_state:
    st.session_state.last_tool_data = None

PORTFOLIO_MAPPING = {
    "Rahul Sharma (Diversified)": "PORTFOLIO_001",
    "Priya Patel (Sector Concentrated)": "PORTFOLIO_002",
    "Arun Krishnamurthy (Conservative)": "PORTFOLIO_003",
    "Master View (Combined)": "ALL_PORTFOLIOS"
}

def get_portfolio_context(pid):
    try:
        with open("data/mock/portfolios.json", "r") as f:
            data = json.load(f)
            if pid == "ALL_PORTFOLIOS":
                return {"risk_tolerance": "medium", "experience_level": "advanced", "name": "Master View"}
            p = data["portfolios"].get(pid, {})
            return {
                "risk_tolerance": p.get("risk_profile", "medium").lower(),
                "experience_level": "intermediate", 
                "name": p.get("user_name", "User")
            }
    except Exception:
        return {"risk_tolerance": "medium", "experience_level": "intermediate", "name": "User"}

def interpret_conf(c):
    if c > 0.8: return "High"
    if c > 0.6: return "Moderate"
    return "Low"

# --- Sidebar: Portfolio Intelligence Panel ---
with st.sidebar:
    st.title("📊 AlphaTrace Hub")
    
    # Portfolio Control
    selected_label = st.selectbox(
        "Context Portfolio", 
        options=list(PORTFOLIO_MAPPING.keys()),
        index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0
    )
    
    new_pid = PORTFOLIO_MAPPING[selected_label]
    if new_pid != st.session_state.current_portfolio:
        st.session_state.current_portfolio = new_pid
        st.session_state.memory = []
        st.session_state.messages = []
        st.session_state.last_tool_data = None
        st.rerun()

    # Visual Analytics Logic
    if st.session_state.last_tool_data:
        st.divider()
        st.subheader("Portfolio Intelligence")
        
        data = st.session_state.last_tool_data
        full_analysis = data.get("full_analysis", {})
        
        # 1. Confidence Indicator with Interpretation
        conf = data.get("metrics", {}).get("confidence", 0.0)
        st.metric("Reasoning Confidence", f"{conf:.2f} ({interpret_conf(conf)})")
        st.progress(conf)
        
        st.divider()

        # 2. Sector Exposure (Donut Chart)
        exposure = full_analysis.get("sector_exposure", {})
        if exposure:
            st.caption("Sector exposure of your portfolio")
            top_sector = max(exposure, key=exposure.get)
            st.write(f"**Top Exposure:** {top_sector}")
            
            df_exposure = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            fig = px.pie(df_exposure, values="Allocation", names="Sector", hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), showlegend=False, height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()

        # 3. Holdings Table (Sorted & Styled)
        holdings = full_analysis.get("ranked_holdings", [])
        if holdings:
            st.markdown("**Top Holdings (by Allocation)**")
            df_holdings = pd.DataFrame(holdings)
            df_holdings = df_holdings.sort_values("allocation", ascending=False)
            
            def color_change(val):
                if val > 0: return "color: #00ff00"
                if val < 0: return "color: #ff4b4b"
                return "color: grey"

            st.dataframe(
                df_holdings[["ticker", "allocation", "daily_change"]].style.applymap(color_change, subset=["daily_change"]),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No current portfolio data available. Ask a question to start analysis.")

    # Passive Memory Indicators
    if st.session_state.memory:
        st.divider()
        latest = st.session_state.memory[-1]
        with st.expander("📌 Recent Drivers", expanded=False):
            for d in latest["drivers"]:
                st.markdown(f"**{d['sector']}**: {d['cause']}")
    
    st.divider()
    st.caption("Temporal Reasoning: ACTIVE")

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Reasoning Cycle ---
if prompt := st.chat_input("Analyze my portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # ANALYTICAL PHASE
            with st.spinner("Synthesizing temporal signals..."):
                recent_memory = st.session_state.memory[-3:]
                session_wrapped = {"current_portfolio": st.session_state.current_portfolio, "memory": recent_memory}
                
                resolution = resolve_context(prompt, session_wrapped)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_memory)
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    st.markdown(f"Clarification: {validation.get('reason', 'I need more context.')}")
                else:
                    execution_results = execute_intents({
                        "intent": validation["validated_intent"],
                        "portfolio_id": validation["portfolio_id"],
                        "confidence": validation["confidence"]
                    }, {"current_portfolio": st.session_state.current_portfolio})
                    
                    st.session_state.current_portfolio = execution_results["portfolio_id"]
                    tool_data = {res["type"]: res["data"] for res in execution_results["results"]}
                    
                    # Update Visual Cache
                    st.session_state.last_tool_data = {
                        "full_analysis": tool_data.get("full_analysis", {}),
                        "metrics": {"confidence": validation["confidence"]},
                        "sector_exposure": tool_data.get("full_analysis", {}).get("sector_exposure", {})
                    }
                    prof = get_portfolio_context(st.session_state.current_portfolio)

            # NARRATIVE PHASE
            if validation["action"] == "execute":
                memory_ctx = extract_relevant_memory(prompt, st.session_state.memory)
                stream_gen = stream_advisory_response(
                    resolution["resolved_query"],
                    validation["validated_intent"],
                    execution_results["portfolio_id"],
                    tool_data,
                    prof,
                    memory_context=memory_ctx
                )
                
                final_response = st.write_stream(stream_gen)
                
                # Temporal Trend
                if len(st.session_state.memory) >= 1:
                    prev = st.session_state.memory[-1]["metrics"].get("portfolio_change", 0.0)
                    curr = tool_data.get("full_analysis", {}).get("daily_change_percent", 0.0)
                    delta = curr - prev
                    if any(kw in prompt.lower() for kw in ["trend", "before", "worse"]):
                        trend = f"\n\n**Temporal Insight:** Market health has {'improved' if delta >= 0 else 'worsened'} by {abs(delta):.2f}% since last check."
                        st.markdown(trend)
                        final_response += trend

                final_briefing = polish_response(final_response, validation["validated_intent"], prof, validation["confidence"])

                memory_turn = normalize_memory_turn(st.session_state.current_portfolio, prompt, validation["validated_intent"], final_briefing, tool_data)
                st.session_state.memory.append(memory_turn)
                st.session_state.messages.append({"role": "assistant", "content": final_briefing})
                st.rerun()

        except Exception as e:
            logger.error(f"UI Error: {e}")
            st.error("Engine Synch Issue.")
