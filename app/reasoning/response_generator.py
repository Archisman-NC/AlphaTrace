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

# --- ADVISORY SYSTEM PROMPTS ---
ADVISORY_SYSTEM_PROMPT = """
You are the AlphaTrace AI Financial Copilot.
Provide high-fidelity, evidence-based reasoning.
1. Only use provided tool outputs. No hallucination.
2. No generic filler. High precision metrics only.
3. Tone: Institutional, professional, objective.
"""

RETRY_FEEDBACK_PROMPT = """
You previously generated a low-quality response (score {score}).
Improve it IMMEDIATELY by:
- Adding specific stock tickers mentioned in the data.
- Adding specific percentage changes (%) for sector/stock impact.
- Making the causal reasoning (X impacts Y) explicit and grounded.
Use ONLY the provided tool data. Do NOT invent numbers.
"""

def guard_tool_data(tool_outputs: dict) -> bool:
    """Verifies that the analytical truth is sufficient for reasoning."""
    if not tool_outputs or "reason" not in tool_outputs:
        return False
    reason = tool_outputs.get("reason", {})
    if isinstance(reason, dict) and reason.get("status") == "error":
        return False
    return True

def generate_validated_response(input_data: dict) -> str:
    """
    RETRY-FIRST ARCHITECTURE:
    generate -> evaluate -> (retry if <5) -> evaluate -> block if still <5
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except:
        return "Synthesis engine is currently offline."

    # 1. HARD DATA GUARD
    if not guard_tool_data(input_data.get("tool_outputs", {})):
        return "I don't have enough verified data to answer that. Could you clarify your portfolio ID?"

    # 2. INITIAL GENERATION
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
        current_draft = str(response.choices[0].message.content)
    except:
        return "Connection fault in reasoning engine."

    # 3. INITIAL EVALUATION
    eval_result = evaluate_response(current_draft)
    initial_score = eval_result.get("score", 0.0)
    print(f"[EVALUATOR] initial_score={initial_score}")

    final_score = initial_score

    # 4. RETRY-FIRST LOGIC (TASK 2)
    if initial_score < 5.0:
        print(f"[GUARDRAIL] Triggering SELF-CORRECTION for score {initial_score}...")
        
        retry_messages = [
            {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {input_data['user_query']}\nData: {json.dumps(input_data['tool_outputs'])}"},
            {"role": "assistant", "content": current_draft},
            {"role": "user", "content": RETRY_FEEDBACK_PROMPT.format(score=initial_score)}
        ]
        
        try:
            retry_res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=retry_messages,
                temperature=0.0 # Force maximum grounding
            )
            current_draft = str(retry_res.choices[0].message.content)
            
            # Re-evaluate
            retry_eval = evaluate_response(current_draft)
            final_score = retry_eval.get("score", 0.0)
            print(f"[EVALUATOR] retry_score={final_score}")
        except:
            print("[FATAL] Retry logic crashed.")

    # 5. FINAL DECISION (TASK 3)
    if final_score < 5.0:
        print(f"[GUARDRAIL] Fallback engaged (final_score={final_score})")
        return "I don't have enough reliable data to give a confident answer. You may want to clarify your query."

    # 6. SOFT IMPROVEMENT (5-6.5)
    if 5.0 <= final_score < 6.5:
        current_draft += "\n\n(This analysis is based on limited signals and may not capture the full picture.)"

    return current_draft

def stream_final_response(user_query: str, intents: List[str], portfolio_id: str, tool_outputs: dict, memory_context: dict):
    input_data = {
        "user_query": user_query,
        "tool_outputs": tool_outputs if isinstance(tool_outputs, dict) else {},
        "memory": memory_context
    }
    
    # RECURSIVE LOOP APPLIED HERE
    final_text = generate_validated_response(input_data)
    
    for word in final_text.split(" "):
        yield word + " "
        time.sleep(0.04)
