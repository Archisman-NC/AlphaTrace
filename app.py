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
from app.reasoning.proactive_engine import generate_proactive_insight # Proactive Upgrade

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

if "proactive_follow_up" not in st.session_state:
    st.session_state.proactive_follow_up = None

if "last_insight_msg" not in st.session_state:
    st.session_state.last_insight_msg = None

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

# --- Sidebar: Hub ---
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
        st.session_state.memory = []
        st.session_state.messages = []
        st.session_state.last_tool_data = None
        st.rerun()

    if st.session_state.last_tool_data:
        st.divider()
        data = st.session_state.last_tool_data
        full_analysis = data.get("full_analysis", {})
        conf = data.get("metrics", {}).get("confidence", 0.0)
        st.metric("Reasoning Confidence", f"{conf:.2f} ({interpret_conf(conf)})")
        st.progress(conf)
        
        exposure = full_analysis.get("sector_exposure", {})
        if exposure:
            st.caption("Sector exposure")
            top_sector = max(exposure, key=exposure.get)
            st.write(f"**Top:** {top_sector}")
            df_exposure = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            fig = px.pie(df_exposure, values="Allocation", names="Sector", hole=0.4, height=180)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Proactive Monitoring: ACTIVE")

# --- Chat Display ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # UI Hook for Proactive Suggestion (only on the last assistant message)
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1 and st.session_state.proactive_follow_up:
            if st.button(f"🔍 Perform Follow-up: {st.session_state.proactive_follow_up}", key="proactive_btn"):
                # Simulate user sending the recommended follow-up
                follow_up_query = st.session_state.proactive_follow_up
                st.session_state.proactive_follow_up = None # Consume it
                # We can't easily auto-trigger chat input from here, but we can append to messages and rerun
                # Let's set a flag to auto-process on rerun
                st.session_state.auto_prompt = follow_up_query
                st.rerun()

# --- Auto Prompt Handling ---
if "auto_prompt" in st.session_state and st.session_state.auto_prompt:
    prompt = st.session_state.auto_prompt
    st.session_state.auto_prompt = None
else:
    prompt = st.chat_input("Ask AlphaTrace about your portfolio...")

# --- Reasoning Cycle ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if not hasattr(st.session_state, "auto_prompt"): st.rerun() # Ensure UI updates before processing

    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE
            with st.spinner("Executing Intelligence Pipeline..."):
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
                    
                    st.session_state.last_tool_data = {
                        "full_analysis": tool_data.get("full_analysis", {}),
                        "metrics": {"confidence": validation["confidence"]},
                        "sector_exposure": tool_data.get("full_analysis", {}).get("sector_exposure", {})
                    }
                    prof = get_portfolio_context(st.session_state.current_portfolio)

                    # 3. PROACTIVE ENGINE Hook
                    proactive = generate_proactive_insight(tool_data, prompt, st.session_state.last_insight_msg)
                    if proactive:
                        st.session_state.proactive_follow_up = proactive["follow_up"]
                        st.session_state.last_insight_msg = proactive["insight"]
                    else:
                        st.session_state.proactive_follow_up = None

            # 2. NARRATIVE PHASE
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
                
                # Append Proactive Insight to response
                if st.session_state.proactive_follow_up:
                    insight_block = f"\n\n**💡 Proactive Insight:** {st.session_state.last_insight_msg} Want me to investigate?"
                    st.markdown(insight_block)
                    final_response += insight_block

                final_briefing = polish_response(final_response, validation["validated_intent"], prof, validation["confidence"])

                memory_turn = normalize_memory_turn(st.session_state.current_portfolio, prompt, validation["validated_intent"], final_briefing, tool_data)
                st.session_state.memory.append(memory_turn)
                st.session_state.messages.append({"role": "assistant", "content": final_briefing})
                st.rerun()

        except Exception as e:
            logger.error(f"Execution Error: {e}")
            st.error("Engine Sync Failure. Attempting recovery...")
