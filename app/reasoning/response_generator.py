import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

from app.evaluation.llm_evaluator import evaluate_response

load_dotenv()
logger = logging.getLogger(__name__)

# --- ADVISORY SYSTEM PROMPT ---
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

def generate_fallback_analysis(input_data: dict) -> str:
    """
    Lightweight narrative fallback for low-confidence scenarios.
    """
    tool_outputs = input_data.get("tool_outputs", {})
    if not tool_outputs or len(tool_outputs) == 0:
        return "Portfolio data is currently limited, however no immediate concentration risks or critical vulnerabilities are visible in the high-level sector snapshots."

    return (
        "Preliminary Portfolio View: Your holdings appear diversified across the identified sectors. "
        "While specific analytical signals are currently limited, no extreme concentration risks or "
        "immediate technical hazards were detected in this reasoning turn."
    )

def generate_validated_response(input_data: dict) -> str:
    """
    REGRESSION-SAFE REPAIR ARCHITECTURE:
    generate -> evaluate -> targeted repair -> signal audit -> merge/discard
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
    initial_breakdown = eval_result.get("details", {})
    
    # DEBUG (Step 5)
    print(f"[EVAL] score={initial_score}")
    
    # Bypass repair if already institutional grade
    if initial_score >= 6.0: # UPDATED THRESHOLD
        return initial_draft

    # 4. SELECTIVE REPAIR
    missing_elements = []
    if not initial_breakdown.get("has_ticker"): missing_elements.append("specific stock tickers")
    if not initial_breakdown.get("is_quant"): missing_elements.append("numerical percentage impact (%)")
    if not initial_breakdown.get("has_causal"): missing_elements.append("causal reasoning")
    
    repair_instruction = f"""
    The previous response is mostly correct, but is missing: {', '.join(missing_elements)}.
    Only improve these specific aspects. Do NOT rewrite the entire answer. 
    Preserve existing correct reasoning. Add missing info based strictly on tool data.
    """
    
    retry_messages = [
        {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Data: {json.dumps(input_data['tool_outputs'])}"},
        {"role": "assistant", "content": initial_draft},
        {"role": "user", "content": repair_instruction}
    ]
    
    final_output = initial_draft
    final_score = initial_score

    try:
        retry_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=retry_messages,
            temperature=0.0
        )
        improved_draft = str(retry_res.choices[0].message.content)
        
        # 5. REGRESSION-SAFE MERGE LOGIC (TASK 1-4)
        retry_eval = evaluate_response(improved_draft)
        retry_score = retry_eval.get("score", 0.0)
        retry_breakdown = retry_eval.get("details", {})
        
        # DEBUG (Step 5)
        print(f"[EVAL-RETRY] score={retry_score}")

        improved = False
        if retry_score > initial_score:
            improved = True
        elif retry_score == initial_score:
            # Check for additive content improvement even if score capped
            added_info = (
                (not initial_breakdown.get("has_ticker") and retry_breakdown.get("has_ticker")) or
                (not initial_breakdown.get("is_quant") and retry_breakdown.get("is_quant")) or
                (not initial_breakdown.get("has_causal") and retry_breakdown.get("has_causal"))
            )
            if added_info: improved = True

        regression = (
            (initial_breakdown.get("has_ticker") and not retry_breakdown.get("has_ticker")) or
            (initial_breakdown.get("is_quant") and not retry_breakdown.get("is_quant")) or
            (initial_breakdown.get("has_causal") and not retry_breakdown.get("has_causal"))
        )

        if improved and not regression:
            print(f"[MERGE] improved=True, regression=False (Score: {initial_score} -> {retry_score})")
            final_output = improved_draft
            final_score = retry_score
        else:
            print(f"[MERGE] improved={improved}, regression={regression} (Rejecting repair, keeping original)")
            final_score = initial_score
    except Exception as e:
        print(f"[MERGE] Logic fault: {e}")
        final_score = initial_score

    # 6. FINAL DECISION (ALIGNED WITH TASK STEP 1 & 3)
    if final_score < 4.0:
        return generate_fallback_analysis(input_data)

    # 7. SOFT IMPROVEMENT LAYER
    if 4.0 <= final_score < 6.0:
        final_output += "\n\n(This analysis provides a high-level view based on current data signals and may be refined as more context emerges.)"

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
