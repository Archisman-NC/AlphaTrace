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

# --- ADAPTIVE RESPONSE ARCHITECTURE ---
STRUCTURES = {
    "FULL_ANALYSIS": "Use a structured format with sections for ### Key Insight, ### Top Drivers, ### Risks, and ### Actions.",
    "EXPLANATION": "Use a natural, paragraph-based format with clear, data-backed reasoning. Avoid rigid bullet points.",
    "COMPARISON": "Identify clear differences using ### Comparison, ### Pros, ### Cons, and ### Recommendation.",
    "GENERAL": "Provide a direct, concise response based on available data."
}

def classify_query(query: str) -> str:
    """Lightweight intent classifier for structural selection."""
    q = str(query).lower()
    if any(x in q for x in ["analyze", "portfolio", "review", "status", "check"]):
        return "FULL_ANALYSIS"
    elif any(x in q for x in ["why", "explain", "reason", "because", "how"]):
        return "EXPLANATION"
    elif any(x in q for x in ["compare", "vs", "versus", "which is better", "difference"]):
        return "COMPARISON"
    else:
        return "GENERAL"

def validate_structure(response: str, intent: str):
    """Informal check to ensure analysis queries maintain structure."""
    if intent == "FULL_ANALYSIS" and "###" not in response:
        print(f"[WARN] Analysis intent detected but structure is missing.")

# --- ADVISORY SYSTEM PROMPT ---
ADVISORY_SYSTEM_PROMPT = """
You are the AlphaTrace AI Financial Analyst. You deliver sharp, data-driven intelligence WITHOUT exception.

CORE SIGNAL RULES:
You are provided with pre-computed signals (key_insight, top_drivers, risks).
1. STRICT FIDELITY: Use these signals EXACTLY. Do not invent new insights or ignore provided drivers.
2. NO GENERALIZATION: Avoid generic sector summaries (e.g., "The sector shows diversification"). 
3. ANALYST PERSONA: Be sharp, decisive, and explain the "So What?" for every data point.

FLEXIBLE STRUCTURE RULE:
Structure your response according to the following intended format:
{structure_guideline}

OUTPUT QUALITY RULES:
1. DECISIVE LANGUAGE: BANNED WORDS: "appears", "may", "suggests", "could", "likely".
   Use: "is driving", "is impacting", "is the primary reason", "has triggered".
2. DATA MINIMUMS: Every response MUST include at least 1-2 tickers and numeric values (% or weight).
3. PRIORITIZE: Focus only on what the system identifies as the most important drivers or risks.
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
    ADAPTIVE RESPONSE PIPELINE:
    intent -> structure -> generate -> evaluate -> repair -> validate
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except:
        return "Synthesis engine is currently offline."

    # 1. INTENT & STRUCTURE SELECTION
    user_query = input_data.get("user_query", "")
    intent = classify_query(user_query)
    structure_guideline = STRUCTURES.get(intent, STRUCTURES["GENERAL"])
    
    print(f"[INTENT] {intent}")
    print(f"[STRUCTURE] {structure_guideline}")

    # 2. INITIAL GENERATION
    dynamic_system_prompt = ADVISORY_SYSTEM_PROMPT.format(structure_guideline=structure_guideline)
    
    messages = [
        {"role": "system", "content": dynamic_system_prompt},
        {"role": "user", "content": f"Query: {user_query}\nData: {json.dumps(input_data['tool_outputs'])}"}
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
    
    print(f"[EVAL] score={initial_score}")
    
    # Check for structural anomalies early
    validate_structure(initial_draft, intent)

    # Bypass repair if already institutional grade
    if initial_score >= 6.0:
        final_output = initial_draft
        final_score = initial_score
    else:
        # 4. SELECTIVE REPAIR
        missing_elements = []
        if not initial_breakdown.get("has_ticker"): missing_elements.append("specific stock tickers")
        if not initial_breakdown.get("is_quant"): missing_elements.append("numerical percentage impact (%)")
        if not initial_breakdown.get("has_causal"): missing_elements.append("causal reasoning")
        
        repair_instruction = f"""
        The previous response is missing mandatory elements: {', '.join(missing_elements)}.
        Improve these aspects while strictly following the required guideline: {structure_guideline}
        Do NOT rewrite the entire answer. 
        """
        
        retry_messages = [
            {"role": "system", "content": dynamic_system_prompt},
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
            
            retry_eval = evaluate_response(improved_draft)
            retry_score = retry_eval.get("score", 0.0)
            
            print(f"[EVAL-RETRY] score={retry_score}")

            if retry_score > initial_score:
                final_output = improved_draft
                final_score = retry_score
                validate_structure(final_output, structure_list)
        except Exception as e:
            print(f"[MERGE] Logic fault: {e}")

    # 5. FINAL DECISION (Part 2 & 4)
    # Calculate confidence from final score (0-10 -> 0.0-1.0)
    confidence = min(max(final_score / 10.0, 0.0), 1.0) if final_score is not None else 0.5
    
    # Debug (Part 6)
    print(f"[EVAL SCORE] {final_score}")
    print(f"[CONFIDENCE] {confidence}")

    if final_score < 4.0:
        return {
            "text": generate_fallback_analysis(input_data),
            "confidence": confidence
        }

    if 4.0 <= final_score < 6.0:
        final_output += "\n\n(Note: This analysis provides a high-level view based on prioritized signals.)"

    return {
        "text": final_output,
        "confidence": confidence
    }

def stream_final_response(user_query: str, intents: List[str], portfolio_id: str, tool_outputs: dict, memory_context: dict):
    input_data = {
        "user_query": user_query,
        "tool_outputs": tool_outputs if isinstance(tool_outputs, dict) else {},
        "memory": memory_context
    }
    # Resolve structured return (Part 3)
    response_payload = generate_validated_response(input_data)
    final_text = response_payload.get("text", "")
    final_conf = response_payload.get("confidence", 0.5)
    
    # Store confidence in session state if possible (Streamlit side)
    # For streaming, we yield words, but we can yield the confidence metadata at the end or use a shared state.
    # We will yield a special metadata marker if needed, or simply let main.py call generate_validated_response directly.
    
    for word in final_text.split(" "):
        yield word + " "
        time.sleep(0.04)
    
    # Yield confidence as a terminal marker
    yield f"__CONFIDENCE__:{final_conf}"
