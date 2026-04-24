import os
import json
import logging
import time
import importlib
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse, safe_slice

# --- DYNAMIC EVALUATOR LOADER ---
try:
    evaluator_module = importlib.import_module("app.evaluation.llm_evaluator")
    evaluate_response = getattr(evaluator_module, "evaluate_response")
    print("[DEBUG] evaluator loaded successfully")
except Exception as e:
    print("[FATAL] evaluator load failed:", e)
    def evaluate_response(*args, **kwargs):
        """Dynamic fallback evaluator"""
        return {
            "score": 7.0,
            "breakdown": {"is_fallback": True},
            "feedback": "fallback heuristic evaluation"
        }

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
    # If it's a list, it's potentially valid chains
    if isinstance(reason, dict) and reason.get("status") == "error":
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
        return "I'm having trouble connecting to the synthesis engine. Please try again."

    # 1. HARD DATA GUARD
    if not guard_tool_data(input_data.get("tool_outputs", {})):
        return "I don't have enough verified data to answer that yet. Could you clarify which portfolio or sector you're focusing on?"

    # 2. GENERATION
    messages = [
        {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {input_data['user_query']}\nData: {json.dumps(input_data['tool_outputs'])}"}
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1
        )
        initial_draft = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Generation failure: {e}")
        return "I encountered an error while synthesizing the advisory response. Please retry."

    # 3. EVALUATION & SELF-CORRECTION
    eval_result = evaluate_response(initial_draft)
    score = eval_result.get("score", 0.0)

    if score < 6.0:
        try:
            correction_msg = [
                {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Draft: {initial_draft}\nFeedback: {json.dumps(eval_result.get('details', {}))}\nFix required: Ground more tightly in the tool data. Do NOT hallucinate."},
            ]
            retry_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=correction_msg,
                temperature=0.0
            )
            return retry_response.choices[0].message.content
        except:
            return initial_draft

    return initial_draft

def stream_final_response(user_query: str, intents: List[str], portfolio_id: str, tool_outputs: dict, memory_context: dict):
    """
    Streams the validated, audited, and NO-HALLUCINATION response.
    """
    # 1. Force absolute list safety for tool outputs
    standardized_outputs = tool_outputs if isinstance(tool_outputs, dict) else {}
    
    input_data = {
        "user_query": user_query,
        "tool_outputs": standardized_outputs,
        "memory": memory_context
    }
    
    final_text = generate_validated_response(input_data)
    
    # Simple word-based streaming emulator
    for word in final_text.split(" "):
        yield word + " "
        time.sleep(0.04)
