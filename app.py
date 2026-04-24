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
from app.reasoning.response_generator import generate_advisory_response

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AlphaTrace AI Copilot", page_icon="📊", layout="wide")

# --- Session Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_portfolio" not in st.session_state:
    st.session_state.current_portfolio = "PORTFOLIO_001"

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# --- Custom Logic: Reasoning Pipeline ---
def get_alpha_trace_response(user_input: str) -> str:
    """
    Executes the full AlphaTrace reasoning cycle from query to narrative synthesis.
    """
    try:
        # Step 0: Context Resolution (Memory)
        session_context = {
            "current_portfolio": st.session_state.current_portfolio,
            "last_analysis": st.session_state.last_analysis
        }
        resolution = resolve_context(user_input, session_context)
        resolved_query = resolution["resolved_query"]
        portfolio_id = resolution["portfolio_id"]
        
        # Step 1: Intent Classification
        classification = classify_intent(resolved_query, portfolio_id, st.session_state.messages)
        
        # Step 2: Intent Validation
        validation = validate_and_route(resolved_query, classification)
        
        if validation["action"] == "fallback":
            return f"I'm sorry, I couldn't clearly understand the request. {validation.get('reason', '')}"
        
        if validation["action"] == "clarify":
            return f"I think I understand, but could you clarify? {validation.get('reason', '')}"

        # Step 3: Execution Routing
        # Note: In production, we'd pull real tool outputs here. 
        # For now, we simulate execution based on validated intents.
        execution_results = execute_intents({
            "intent": validation["validated_intent"],
            "portfolio_id": validation["portfolio_id"],
            "confidence": validation["confidence"]
        }, {"current_portfolio": st.session_state.current_portfolio})
        
        # Update State if portfolio switched
        st.session_state.current_portfolio = execution_results["portfolio_id"]

        # Aggregate data for generator
        tool_data = {}
        for result in execution_results["results"]:
            tool_data[result["type"]] = result["data"]

        # Step 4: Narrative Synthesis
        raw_response = generate_advisory_response(
            resolved_query,
            validation["validated_intent"],
            execution_results["portfolio_id"],
            tool_data,
            {"risk_tolerance": "high", "experience_level": "advanced"} # Mock profile
        )
        
        # Step 5: Premium Polish (Hybrid Strategy)
        from app.reasoning.response_polisher import polish_response
        final_response = polish_response(
            raw_response, 
            validation["validated_intent"], 
            {"risk_tolerance": "high", "experience_level": "advanced"},
            validation["confidence"]
        )

        # Save analysis for next turn context
        st.session_state.last_analysis = {"summary": final_response}
        
        return final_response

    except Exception as e:
        logger.error(f"Reasoning Pipeline Error: {e}")
        return f"System Error: I encountered an issue processing your request. Please check my status logs."

# --- UI Layout ---
st.title("📊 AlphaTrace AI Copilot")
st.markdown("*Analyze your portfolio conversationally using causal AI.*")

# --- Sidebar info ---
with st.sidebar:
    st.header("Session Status")
    st.info(f"**Active Portfolio:** {st.session_state.current_portfolio}")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.last_analysis = None
        st.rerun()

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("How can I help with your portfolio today?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process & Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing causal chains..."):
            response = get_alpha_trace_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
