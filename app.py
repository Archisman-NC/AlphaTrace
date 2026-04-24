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
from app.reasoning.response_generator import stream_final_response
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
if "memory" not in st.session_state: st.session_state.memory = []
if "messages" not in st.session_state: st.session_state.messages = []
if "current_portfolio" not in st.session_state: st.session_state.current_portfolio = "PORTFOLIO_001"
if "last_tool_data" not in st.session_state: st.session_state.last_tool_data = None
if "proactive_metadata" not in st.session_state: st.session_state.proactive_metadata = None
if "last_insight_topic" not in st.session_state: st.session_state.last_insight_topic = None
if "last_insight_turn" not in st.session_state: st.session_state.last_insight_turn = -2

PORTFOLIO_MAPPING = {
    "Rahul Sharma (Diversified)": "PORTFOLIO_001",
    "Priya Patel (Sector Concentrated)": "PORTFOLIO_002",
    "Arun Krishnamurthy (Conservative)": "PORTFOLIO_003",
    "Master View (Combined)": "ALL_PORTFOLIOS"
}

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
        
        # Pull enriched metrics from across tool results
        metrics = {}
        for tool_type, tool_res in data.items():
            metrics.update(tool_res.get("metrics", {}))
        
        conf = data.get("global_metrics", {}).get("confidence", 0.0)
        st.metric("Confidence", f"{conf:.2f} ({interpret_conf(conf)})")
        st.progress(conf)
        
        exposure = metrics.get("sector_exposure", {})
        if exposure:
            st.caption("Sector highlights")
            df_exp = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            # Remove non-numeric values for pie chart
            df_exp = df_exp[df_exp['Allocation'].apply(lambda x: isinstance(x, (int, float)))]
            fig = px.pie(df_exp, values="Allocation", names="Sector", hole=0.4, height=180)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# --- Chat Display ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1 and st.session_state.proactive_metadata:
            meta = st.session_state.proactive_metadata
            if st.button(f"🔍 Analyze this signal: {meta['type'].title()}", key="proactive_btn"):
                st.session_state.last_insight_turn = len(st.session_state.memory)
                st.session_state.auto_prompt = meta['followup_query']
                st.session_state.proactive_metadata = None 
                st.rerun()

# --- Auto Prompt Handling ---
if "auto_prompt" in st.session_state and st.session_state.auto_prompt:
    prompt = st.session_state.auto_prompt
    st.session_state.auto_prompt = None
else:
    prompt = st.chat_input("Analyze portfolio...")

# --- Reasoning Cycle ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if not hasattr(st.session_state, "auto_prompt"): st.rerun()

    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE
            with st.spinner("Executing Intelligence pipeline..."):
                current_turn = len(st.session_state.memory)
                recent_mem = st.session_state.memory[-3:]
                session_wrapped = {"current_portfolio": st.session_state.current_portfolio, "memory": recent_mem}
                
                resolution = resolve_context(prompt, session_wrapped)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_mem)
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    st.markdown(f"Clarification: {validation.get('reason', 'Need context.')}")
                else:
                    execution_results = execute_intents({
                        "intent": validation["validated_intent"],
                        "portfolio_id": validation["portfolio_id"],
                        "confidence": validation["confidence"]
                    }, {"current_portfolio": st.session_state.current_portfolio})
                    
                    st.session_state.current_portfolio = execution_results["portfolio_id"]
                    
                    # STANDARDIZED DATA AGGREGATION
                    tool_data = {res["type"]: res for res in execution_results["results"]}
                    tool_data["global_metrics"] = {"confidence": validation["confidence"]}
                    
                    st.session_state.last_tool_data = tool_data

                    # PROACTIVE ENGINE (Throttled & Standardized)
                    if (current_turn - st.session_state.last_insight_turn) >= 2:
                        proactive = generate_proactive_insight(tool_data, prompt, st.session_state.memory, st.session_state.last_insight_topic)
                        if proactive:
                            st.session_state.proactive_metadata = proactive
                            st.session_state.last_insight_topic = proactive["topic"]
                            st.session_state.last_insight_turn = current_turn
                        else: st.session_state.proactive_metadata = None
                    else: st.session_state.proactive_metadata = None

            # 2. NARRATIVE PHASE (Gated)
            if validation["action"] == "execute":
                memory_ctx = extract_relevant_memory(prompt, st.session_state.memory)
                
                stream_gen = stream_final_response(
                    user_query=resolution["resolved_query"],
                    intents=validation["validated_intent"],
                    portfolio_id=execution_results["portfolio_id"],
                    tool_outputs=tool_data,
                    memory_context=memory_ctx
                )
                
                final_res = st.write_stream(stream_gen)
                
                if st.session_state.proactive_metadata:
                    insight_text = f"\n\n{st.session_state.proactive_metadata['text']}"
                    st.markdown(insight_text)
                    final_res += insight_text

                final_brief = polish_response(final_res, validation["validated_intent"], {}, validation["confidence"])

                memory_obj = normalize_memory_turn(st.session_state.current_portfolio, prompt, validation["validated_intent"], final_brief, tool_data)
                st.session_state.memory.append(memory_obj)
                st.session_state.messages.append({"role": "assistant", "content": final_brief})
                st.rerun()

        except Exception as e:
            logger.error(f"Execution Error: {e}")
            st.error("Engine Fault. Re-initializing...")
