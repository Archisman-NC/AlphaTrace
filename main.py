import os
import sys

# --- Path Stabilization Sentinel (Part 1) ---
ROOT = os.path.dirname(os.path.abspath(__file__))
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
from dotenv import load_dotenv

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
        
        conf = data.get("global_metrics", {}).get("confidence", 0.1)
        st.metric("Confidence", f"{conf:.2f} ({interpret_conf(conf)})")
        
        exposure = metrics.get("sector_exposure", {})
        if exposure:
            df_exp = pd.DataFrame(list(exposure.items()), columns=["Sector", "Allocation"])
            df_exp["Allocation"] = df_exp["Allocation"].apply(safe_float)
            fig = px.pie(df_exp, values="Allocation", names="Sector", hole=0.4, height=180)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            # Fix 9: Streamlit stretch layout (Part 8)
            st.plotly_chart(fig, width="stretch")

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
                
                # Diagnostic Sentinels (Part 1)
                print(f"[QUERY] {active_prompt}")
                resolution = get_resolve_context()(active_prompt, session_wrapped)
                print(f"[RESOLVED] {resolution['resolved_query']}")
                
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_mem)
                validation = validate_and_route(resolution["resolved_query"], classification)
                print(f"[INTENT BEFORE] {validation.get('validated_intent')}")

                if validation["action"] != "execute":
                    res_path = validation.get('reason', 'Could you clarify that?')
                    st.markdown(res_path); st.session_state.messages.append({"role": "assistant", "content": res_path})
                else:
                    # FIX DEFAULT INTENT (Part 2)
                    intent = validation.get("validated_intent")
                    if not intent or intent == ["full_analysis"]:
                        q = active_prompt.lower()
                        if any(x in q for x in ["why", "reason", "cause"]):
                            intent = ["explanation"]
                        elif any(x in q for x in ["compare", "vs", "difference"]):
                            intent = ["comparison"]
                        elif any(x in q for x in ["what should", "advice", "suggest"]):
                            intent = ["advice"]
                        else:
                            intent = ["full_analysis"]
                    
                    print(f"[INTENT FINAL] {intent}")

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

                    # Narrative (Part 3)
                    stream_gen = get_stream_final_response()(resolution["resolved_query"], intent, execution_results["portfolio_id"], tool_data, extract_relevant_memory(active_prompt, st.session_state.memory))
                    
                    # Capture stream output for parsing
                    full_narrative = st.write_stream(stream_gen)
                    
                    # Extract confidence metadata marker
                    conf_val = 0.5
                    if "__CONFIDENCE__:" in full_narrative:
                        parts = full_narrative.split("__CONFIDENCE__:")
                        display_text = parts[0]
                        conf_val = safe_float(parts[1].strip())
                    else:
                        display_text = full_narrative

                    # Fidelity Labeling (Part 7)
                    if conf_val > 0.75: label, color = "High", "green"
                    elif conf_val > 0.5: label, color = "Medium", "orange"
                    else: label, color = "Low (Best Effort)", "gray"
                    
                    st.caption(f"Reasoning Fidelity: :{color}[{label}] ({int(conf_val*100)}%)")
                    
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
            st.error("I've encountered a temporary analytical hurdle. Please re-state your query or try selecting a different portfolio.")
