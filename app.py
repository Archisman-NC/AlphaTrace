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

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AlphaTrace AI Copilot", page_icon="📊", layout="wide")

# --- Memory System & Session State ---
if "memory" not in st.session_state:
    st.session_state.memory = [] # Array of MemoryTurn objects

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = "PORTFOLIO_001"

def get_recent_memory(k=3):
    """Retrieves the last K structured turns for context windowing."""
    return st.session_state.memory[-k:]

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

# --- Custom Styling ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("📊 AlphaTrace AI Copilot")
st.markdown("*Intelligent, multi-turn financial reasoning engine.*")

# --- Sidebar ---
with st.sidebar:
    st.header("Active Context")
    selected_label = st.selectbox(
        "Active Portfolio", 
        options=list(PORTFOLIO_MAPPING.keys()),
        index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0
    )
    
    new_pid = PORTFOLIO_MAPPING[selected_label]
    if new_pid != st.session_state.current_portfolio:
        st.session_state.current_portfolio = new_pid
        st.session_state.memory = [] # Reset memory on portfolio switch
        st.session_state.messages = []
        st.rerun()

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Reasoning Cycle ---
if prompt := st.chat_input("Ask about your portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE (With Spinner)
            with st.spinner("Analyzing context and executing tools..."):
                recent_memory = get_recent_memory(k=3)
                
                # Context Resolution
                session_wrapped = {
                    "current_portfolio": st.session_state.current_portfolio,
                    "memory": recent_memory
                }
                resolution = resolve_context(prompt, session_wrapped)
                
                # Intent Classification (Memory-aware)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], recent_memory)
                
                # Validation & Routing
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    response_text = f"Could you clarify? {validation.get('reason', 'I need more context.')}"
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Execute Tools
                    execution_results = execute_intents({
                        "intent": validation["validated_intent"],
                        "portfolio_id": validation["portfolio_id"],
                        "confidence": validation["confidence"]
                    }, {"current_portfolio": st.session_state.current_portfolio})
                    
                    st.session_state.current_portfolio = execution_results["portfolio_id"]
                    tool_data = {res["type"]: res["data"] for res in execution_results["results"]}
                    prof = get_portfolio_context(st.session_state.current_portfolio)

            # 2. NARRATIVE PHASE (Streaming)
            if validation["action"] == "execute":
                stream_gen = stream_advisory_response(
                    resolution["resolved_query"],
                    validation["validated_intent"],
                    execution_results["portfolio_id"],
                    tool_data,
                    prof
                )
                
                final_response = st.write_stream(stream_gen)
                
                # Optional Premium Polish (Background Sync)
                final_briefing = polish_response(final_response, validation["validated_intent"], prof, validation["confidence"])

                # --- 3. MEMORY CONSOLIDATION ---
                # Extract drivers/risks from tool_data safely for memory turn
                reason_data = tool_data.get("reason", {})
                risk_data = tool_data.get("risk", {})
                
                memory_turn = {
                    "portfolio_id": st.session_state.current_portfolio,
                    "user_query": prompt,
                    "intents": validation["validated_intent"],
                    "summary": final_briefing,
                    "drivers": reason_data.get("chains", [])[:3], # Top 3 drivers for context
                    "risks": risk_data.get("risks", [])[:3],      # Top 3 risks for context
                    "metrics": tool_data.get("full_analysis", {}),
                    "timestamp": time.time()
                }
                
                st.session_state.memory.append(memory_turn)
                st.session_state.messages.append({"role": "assistant", "content": final_briefing})

        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            st.error("I encountered an issue processing your request.")
