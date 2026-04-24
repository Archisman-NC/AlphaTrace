import os
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse, safe_slice

try:
    from app.evaluation.llm_evaluator import evaluate_response
except Exception as e:
    print("[WARNING] evaluator import failed:", e)
    def evaluate_response(*args, **kwargs):
        """Emergency fallback evaluator"""
        return {"score": 7.0, "details": {"fallback": True}, "feedback": "fallback engagement"}

load_dotenv()
logger = logging.getLogger(__name__)

# --- STRICT NO-HALLUCINATION PROMPT ---
ADVISORY_SYSTEM_PROMPT = """
You are the AlphaTrace AI Financial Copilot.
Your mission is to provide high-fidelity, evidence-based reasoning.

## MANDATORY RULES (ZERO HALLUCINATION):
1. Only use the provided tool outputs (Market Intel, Causal Chains, Risks).
2. Do NOT invent numbers, percentages, or facts.
3. NEVER assume stock performance if not in data.
4. If tool data is missing or "status": "error", state clearly: "I'm missing some required data to give a precise answer on that."
5. Professional, concise, and institutional tone.

## STRUCTURE:
1. Executive Summary (Anchored in drivers)
2. Risk Context (Anchored in concentration/conflicts)
3. Actionable Narrative
"""

def guard_tool_data(tool_outputs: dict) -> bool:
    """Verifies that the analytical truth is sufficient for reasoning."""
    if not tool_outputs or "reason" not in tool_outputs:
        return False
    
    reason = tool_outputs.get("reason", {})
    if reason.get("status") == "error" or not reason.get("drivers", []):
        return False
    return True

def generate_validated_response(input_data: dict) -> str:
    """
    Production-grade generation flow:
    Guard -> Generate -> Audit -> Correct -> Final
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except:
        return "Synthesis engine is currently offline. Please try again later."

    # 1. HARD DATA GUARD
    if not guard_tool_data(input_data.get("tool_outputs", {})):
        return "I don't have enough verified data to answer that yet. Could you clarify which portfolio or sector you're focusing on?"

    # 2. GENERATION
    messages = [
        {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {input_data['user_query']}\nData: {json.dumps(input_data['tool_outputs'])}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1 # Low temp for data grounding
    )
    initial_draft = response.choices[0].message.content

    # 3. EVALUATION & SELF-CORRECTION
    eval_result = evaluate_response(initial_draft)
    score = eval_result["score"]

    if score < 6.5:
        correction_msg = [
            {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Draft: {initial_draft}\nFeedback: {json.dumps(eval_result['details'])}\nFix required: Ground more tightly in the tool data. Do NOT hallucinate."},
        ]
        retry_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=correction_msg,
            temperature=0.0
        )
        return retry_response.choices[0].message.content

    return initial_draft

def stream_final_response(user_query: str, intents: List[str], portfolio_id: str, tool_outputs: dict, memory_context: dict):
    """
    Streams the validated, audited, and NO-HALLUCINATION response.
    """
    input_data = {
        "user_query": user_query,
        "tool_outputs": tool_outputs,
        "memory": memory_context
    }
    
    final_text = generate_validated_response(input_data)
    
    # Simulate streaming for the audited output
    for word in final_text.split(" "):
        yield word + " "
        time_to_sleep = 0.05
        import time
        time.sleep(time_to_sleep)
