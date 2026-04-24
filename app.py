import streamlit as st
import os
import json
import logging
from dotenv import load_dotenv

# Import Reasoning Stack
from app.reasoning.context_resolver import resolve_context
from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.router import execute_intents
from app.reasoning.response_generator import stream_advisory_response # Updated to streaming
from app.reasoning.response_polisher import polish_response

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AlphaTrace AI Copilot", page_icon="📊", layout="wide")

# --- Session Initialization ---
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = "PORTFOLIO_001"

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# --- Custom Styling ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; }
    .stChatInputContainer { padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("📊 AlphaTrace AI Copilot")
st.markdown("*Analyze your portfolio conversationally using causal AI.*")

# --- Sidebar info ---
with st.sidebar:
    st.header("Select Context")
    selected_label = st.selectbox(
        "Active Portfolio", 
        options=list(PORTFOLIO_MAPPING.keys()),
        index=list(PORTFOLIO_MAPPING.values()).index(st.session_state.current_portfolio) if st.session_state.current_portfolio in PORTFOLIO_MAPPING.values() else 0
    )
    
    new_pid = PORTFOLIO_MAPPING[selected_label]
    if new_pid != st.session_state.current_portfolio:
        st.session_state.current_portfolio = new_pid
        st.session_state.messages = [] # Reset on switch
        st.session_state.last_analysis = None
        st.rerun()

    st.divider()
    st.info("AlphaTrace is reasoning using hybrid Llama-3.3 (Logic) and GPT-4o-mini (Polish) strategies.")

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Reasoning Cycle ---
if prompt := st.chat_input("Ask about your portfolio..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reasoning Cycle
    with st.chat_message("assistant"):
        try:
            # 1. ANALYTICAL PHASE (Spinner)
            with st.spinner("Analyzing market signals and portfolio data..."):
                session_context = {
                    "current_portfolio": st.session_state.current_portfolio,
                    "last_analysis": st.session_state.last_analysis
                }
                
                # Context & Intent logic
                resolution = resolve_context(prompt, session_context)
                classification = classify_intent(resolution["resolved_query"], resolution["portfolio_id"], st.session_state.messages[:-1])
                validation = validate_and_route(resolution["resolved_query"], classification)
                
                if validation["action"] != "execute":
                    response_text = f"Could you clarify? {validation.get('reason', 'I need more context to be precise.')}"
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

            # 2. NARRATIVE PHASE (Streaming - No Spinner)
            if validation["action"] == "execute":
                stream_gen = stream_advisory_response(
                    resolution["resolved_query"],
                    validation["validated_intent"],
                    execution_results["portfolio_id"],
                    tool_data,
                    prof
                )
                
                # Stream to UI and capture final text
                final_response = st.write_stream(stream_gen)
                
                # Update State
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.session_state.last_analysis = {"summary": final_response}

        except Exception as e:
            logger.error(f"Pipeline Error: {e}")
            err_msg = "I encountered an issue processing your request. Please check my status logs."
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
