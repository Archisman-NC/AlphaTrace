import streamlit as st
import os
import json
import logging
import time
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

# --- Sidebar: Active Hub ---
with st.sidebar:
    st.title("📊 AlphaTrace Hub")
    selected_label = st.selectbox(
        "Active Portfolio", 
        options=list(PORTFOLIO_MAPPING.keys()),
        index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0
    )
    
    new_pid = PORTFOLIO_MAPPING[selected_label]
    if new_pid != st.session_state.current_portfolio:
        st.session_state.current_portfolio = new_id
        st.session_state.memory = []
        st.session_state.messages = []
        st.rerun()

    if st.session_state.memory:
        st.divider()
        latest = st.session_state.memory[-1]
        with st.expander("📌 Last Causal Drivers", expanded=True):
            for d in latest["drivers"]:
                st.markdown(f"**{d['sector']}**: {d['cause']} ({d['impact']:.2f}%)")
        with st.expander("⚠️ Active Risks", expanded=False):
            for r in latest["risks"]:
                color = "red" if r['severity'] > 0.7 else "orange"
                st.markdown(f":{color}[**{r['type']}**]: {r['description']}")

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask about your portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE
            with st.spinner("Analyzing temporal signals..."):
                recent_memory = st.session_state.memory[-3:]
                
                # Logic Chain
                session_wrapped = {"current_portfolio": st.session_state.current_portfolio, "memory": recent_memory}
                resolution = resolve_context(prompt, session_wrapped)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_memory)
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    response_text = f"Clarification: {validation.get('reason', 'I need more context.')}"
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Execute
                    execution_results = execute_intents({
                        "intent": validation["validated_intent"],
                        "portfolio_id": validation["portfolio_id"],
                        "confidence": validation["confidence"]
                    }, {"current_portfolio": st.session_state.current_portfolio})
                    
                    st.session_state.current_portfolio = execution_results["portfolio_id"]
                    tool_data = {res["type"]: res["data"] for res in execution_results["results"]}
                    prof = get_portfolio_context(st.session_state.current_portfolio)

            # 2. NARRATIVE PHASE (Streaming + Temporal Reasoning)
            if validation["action"] == "execute":
                # Active Memory Prioritization for Generator
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
                
                # --- TEMPORAL WOW MOMENT ---
                # Check for trend compared to previous memory turn
                if len(st.session_state.memory) >= 1:
                    prev_change = st.session_state.memory[-1]["metrics"].get("portfolio_change", 0.0)
                    curr_change = tool_data.get("full_analysis", {}).get("daily_change_percent", 0.0)
                    delta = curr_change - prev_change
                    
                    trend_line = ""
                    if any(kw in prompt.lower() for kw in ["worse", "better", "trend", "change", "before"]):
                        trend_line = f"\n\n**Temporal Insight:** Performance has {'improved' if delta >= 0 else 'worsened'} by {abs(delta):.2f}% since our last check."
                        st.markdown(trend_line)
                        final_response += trend_line

                final_briefing = polish_response(final_response, validation["validated_intent"], prof, validation["confidence"])

                # Update Memory
                memory_turn = normalize_memory_turn(
                    portfolio_id=st.session_state.current_portfolio,
                    user_query=prompt,
                    intents=validation["validated_intent"],
                    summary=final_briefing,
                    tool_data=tool_data
                )
                
                st.session_state.memory.append(memory_turn)
                st.session_state.messages.append({"role": "assistant", "content": final_briefing})

        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            st.error("System Error: Please check connection.")
