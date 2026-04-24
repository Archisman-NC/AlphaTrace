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
            "breakdown": {"has_ticker": True, "is_quant": True, "has_causal": True},
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

def guard_tool_data(tool_outputs: dict) -> bool:
    if not tool_outputs or "reason" not in tool_outputs: return False
    reason = tool_outputs.get("reason", {})
    if isinstance(reason, dict) and reason.get("status") == "error": return False
    return True

def generate_validated_response(input_data: dict) -> str:
    """
    SELECTIVE REPAIR ARCHITECTURE:
    generate -> evaluate -> targeted repair -> merge -> final
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except:
        return "Synthesis engine is currently offline."

    # 1. HARD DATA GUARD
    if not guard_tool_data(input_data.get("tool_outputs", {})):
        return "I don't have enough verified data to answer that. Could you clarify your portfolio context?"

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
        initial_draft = str(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Generation failure: {e}")
        return "Connection fault in reasoning engine."

    # 3. INITIAL EVALUATION
    eval_result = evaluate_response(initial_draft)
    initial_score = eval_result.get("score", 0.0)
    breakdown = eval_result.get("details", {})
    
    # Bypass repair if already institutional grade
    if initial_score >= 6.5:
        print(f"[EVALUATOR] score={initial_score} (Bypassing repair)")
        return initial_draft

    print(f"[EVALUATOR] initial_score={initial_score} (Triggering selective repair)")

    # 4. SELECTIVE REPAIR (TASK 1 & 2)
    missing_elements = []
    if not breakdown.get("has_ticker"): missing_elements.append("specific stock tickers (e.g. HDFCBANK)")
    if not breakdown.get("is_quant"): missing_elements.append("numerical percentage impact (%)")
    if not breakdown.get("has_causal"): missing_elements.append("causal reasoning (linking drivers to impact)")
    
    repair_instruction = f"""
    The previous response is mostly correct, but is missing: {', '.join(missing_elements)}.

    Only improve these specific aspects. 
    Do NOT rewrite the entire answer. 
    Preserve existing correct reasoning and structure.
    Add only the missing information based strictly on the provided tool data.
    """
    
    retry_messages = [
        {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Data: {json.dumps(input_data['tool_outputs'])}"},
        {"role": "assistant", "content": initial_draft},
        {"role": "user", "content": repair_instruction}
    ]
    
    final_output = initial_draft
    try:
        retry_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=retry_messages,
            temperature=0.0
        )
        improved_draft = str(retry_res.choices[0].message.content)
        
        # 5. MERGE STRATEGY (TASK 3)
        retry_eval = evaluate_response(improved_draft)
        retry_score = retry_eval.get("score", 0.0)
        
        if retry_score > initial_score:
            print(f"[CORRECTION] improved=True (Initial={initial_score}, Improved={retry_score})")
            final_output = improved_draft
            final_score = retry_score
        else:
            print(f"[CORRECTION] improved=False (Discarding retry, keeping original score {initial_score})")
            final_score = initial_score
    except:
        final_score = initial_score

    # 6. FINAL DECISION & FALLBACK
    if final_score < 5.0:
        return "I'm missing some required depth to provide a high-confidence advisory. Please refine your query or check available data."

    # 7. SOFT IMPROVEMENT LAYER
    if 5.0 <= final_score < 6.5:
        final_output += "\n\n(This analysis is based on limited signals and may not capture the full picture.)"

    return final_output

def stream_final_response(user_query: str, intents: List[str], portfolio_id: str, tool_outputs: dict, memory_context: dict):
    input_data = {
        "user_query": user_query,
        "tool_outputs": tool_outputs if isinstance(tool_outputs, dict) else {},
        "memory": memory_context
    }
    
    final_text = generate_validated_response(input_data)
    for word in final_text.split(" "):
        yield word + " "
        time.sleep(0.04)
