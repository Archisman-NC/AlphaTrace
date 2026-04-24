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
from app.reasoning.proactive_engine import generate_proactive_insight

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

if "proactive_metadata" not in st.session_state:
    st.session_state.proactive_metadata = None

if "last_insight_topic" not in st.session_state:
    st.session_state.last_insight_topic = None

if "last_insight_turn" not in st.session_state:
    st.session_state.last_insight_turn = -2 # Offset to allow first turn trigger

PORTFOLIO_MAPPING = {
    "Rahul Sharma (Diversified)": "PORTFOLIO_001",
    "Priya Patel (Sector Concentrated)": "PORTFOLIO_002",
    "Arun Krishnamurthy (Conservative)": "PORTFOLIO_003",
    "Master View (Combined)": "ALL_PORTFOLIOS"
}

# --- Sidebar: Hub ---
with st.sidebar:
    st.title("📊 AlphaTrace Hub")
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
        st.session_state.rerun()

    if st.session_state.last_tool_data:
        st.divider()
        data = st.session_state.last_tool_data
        full_analysis = data.get("full_analysis", {})
        conf = data.get("metrics", {}).get("confidence", 0.0)
        st.metric("Confidence", f"{conf:.2f}")
        st.progress(conf)
        
        exposure = full_analysis.get("sector_exposure", {})
        if exposure:
            top_sector = max(exposure, key=exposure.get)
            st.write(f"**Top:** {top_sector}")
            df_exposure = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            fig = px.pie(df_exposure, values="Allocation", names="Sector", hole=0.4, height=180)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Proactive Reasoning: ACTIVE")

# --- Chat Display ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # UI Button for Proactive Follow-up (Last turn focus)
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1 and st.session_state.proactive_metadata:
            meta = st.session_state.proactive_metadata
            if st.button(f"🔍 Analyze this signal: {meta['type'].title()}", key="proactive_btn"):
                st.session_state.auto_prompt = meta['followup_query']
                st.session_state.proactive_metadata = None 
                st.rerun()

# --- Auto Prompt Handling ---
if "auto_prompt" in st.session_state and st.session_state.auto_prompt:
    prompt = st.session_state.auto_prompt
    st.session_state.auto_prompt = None
else:
    prompt = st.chat_input("Ask AlphaTrace...")

# --- Reasoning Cycle ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if not hasattr(st.session_state, "auto_prompt"): st.rerun()

    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE
            with st.spinner("Analyzing signals..."):
                current_turn = len(st.session_state.memory)
                recent_memory = st.session_state.memory[-3:]
                session_wrapped = {"current_portfolio": st.session_state.current_portfolio, "memory": recent_memory}
                
                resolution = resolve_context(prompt, session_wrapped)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_memory)
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    st.markdown(f"Clarification: {validation.get('reason', 'Need more context.')}")
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

                    # 3. THROTTLED PROACTIVE ENGINE (2-Turn Cooldown)
                    if (current_turn - st.session_state.last_insight_turn) >= 2:
                        proactive = generate_proactive_insight(
                            tool_data, 
                            prompt, 
                            st.session_state.memory, 
                            st.session_state.last_insight_topic
                        )
                        if proactive:
                            st.session_state.proactive_metadata = proactive
                            st.session_state.last_insight_topic = proactive["topic"]
                            st.session_state.last_insight_turn = current_turn
                        else:
                            st.session_state.proactive_metadata = None
                    else:
                        st.session_state.proactive_metadata = None

            # 2. NARRATIVE PHASE
            if validation["action"] == "execute":
                memory_ctx = extract_relevant_memory(prompt, st.session_state.memory)
                stream_gen = stream_advisory_response(
                    resolution["resolved_query"],
                    validation["validated_intent"],
                    execution_results["portfolio_id"],
                    tool_data,
                    {}, # Profile
                    memory_context=memory_ctx
                )
                
                final_response = st.write_stream(stream_gen)
                
                if st.session_state.proactive_metadata:
                    insight_block = f"\n\n{st.session_state.proactive_metadata['text']}"
                    st.markdown(insight_block)
                    final_response += insight_block

                final_briefing = polish_response(final_response, validation["validated_intent"], {}, validation["confidence"])

                memory_turn = normalize_memory_turn(st.session_state.current_portfolio, prompt, validation["validated_intent"], final_briefing, tool_data)
                st.session_state.memory.append(memory_turn)
                st.session_state.messages.append({"role": "assistant", "content": final_briefing})
                st.rerun()

        except Exception as e:
            logger.error(f"Execution Error: {e}")
            st.error("Engine Connectivity Error.")
