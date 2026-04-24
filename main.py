import sys
import os
import time

# --- Deterministic Module Reset (Step 2) ---
# Prevents Streamlit hot-reload from corrupting the 'app' namespace
for m in list(sys.modules.keys()):
    if m.startswith("app."):
        del sys.modules[m]

# --- Path Stabilization Sentinel ---
# Ensures absolute imports resolve to the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import os
import json
import logging
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from app.utils.helpers import safe_float
# --- Namespace Integrity Check ---
try:
    import app
    print(f"[BOOT] Package 'app' resolved to: {app.__file__}")
except Exception as e:
    print(f"[CRITICAL] AlphaTrace structural failure: {e}")

# --- Global Import Shield ---
try:
    from app.evaluation.llm_evaluator import evaluate_response
except Exception:
    def evaluate_response(*args, **kwargs): return {"score": 7.0, "confidence": 0.5, "details": {}}

try:
    from app.reasoning.proactive_engine import generate_proactive_insight
except Exception:
    def generate_proactive_insight(*args, **kwargs): return None

# Standard Imports
from app.reasoning.context_resolver import resolve_context
from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.router import execute_intents
from app.reasoning.response_generator import stream_final_response
from app.reasoning.response_polisher import polish_response
from app.reasoning.memory_engine import normalize_memory_turn, extract_relevant_memory

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
    val = float(c) if isinstance(c, (int, float)) else 0.5
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
            # Fix 9: Streamlit stretch layout
            st.plotly_chart(fig, use_container_width=True)

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
                
                resolution = resolve_context(active_prompt, session_wrapped)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_mem)
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    res_path = validation.get('reason', 'Could you clarify that?')
                    st.markdown(res_path); st.session_state.messages.append({"role": "assistant", "content": res_path})
                else:
                    execution_results = execute_intents({
                        "intent": validation.get("validated_intent", ["full_analysis"]),
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
                    stream_gen = stream_final_response(resolution["resolved_query"], validation["validated_intent"], execution_results["portfolio_id"], tool_data, extract_relevant_memory(active_prompt, st.session_state.memory))
                    final_res = st.write_stream(stream_gen)
                    
                    if st.session_state.proactive_metadata:
                        st.markdown(f"\n\n{st.session_state.proactive_metadata['text']}")
                        final_res += f"\n\n{st.session_state.proactive_metadata['text']}"

                    final_brief = polish_response(final_res, validation.get("validated_intent", ["full_analysis"]), {}, validation.get("confidence", 0.5))
                    memory_obj = normalize_memory_turn(st.session_state.current_portfolio, active_prompt, validation["validated_intent"], final_brief, tool_data)
                    st.session_state.memory.append(memory_obj)
                    st.session_state.messages.append({"role": "assistant", "content": final_brief})
                    st.rerun()

        except Exception as e:
            logger.error(f"Execution Fault: {e}")
            st.error("I've encountered a temporary analytical hurdle. Please re-state your query or try selecting a different portfolio.")
