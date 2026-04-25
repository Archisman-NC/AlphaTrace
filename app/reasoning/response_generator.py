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

STRUCTURES = {
    "full_analysis": "Use a structured format with sections for ### 1. Key Insight, ### 2. Top Drivers, ### 3. Risks, and ### 4. Recommended Actions.",
    "explanation": "Use a natural, paragraph-based format with clear, data-backed reasoning. Avoid rigid section headers.",
    "comparison": "Identify clear differences using ### Comparison, ### Pros, ### Cons, and ### Final Verdict.",
    "advice": "Provide concise, actionable financial suggestions. Use bullet points for steps.",
    "general": "Provide a direct, context-aware response based on the provided data."
}

def get_structure_guideline(intents: List[str]) -> str:
    """Selects the structure guideline based on dominant intent."""
    if not intents: return STRUCTURES["general"]
    
    # Priority order for structure
    for key in ["full_analysis", "explanation", "comparison", "advice"]:
        if key in intents:
            return STRUCTURES[key]
    
    return STRUCTURES["general"]

def validate_structure(response: str, intent: str):
    """Informal check to ensure analysis queries maintain structure."""
    if intent == "full_analysis" and "###" not in response:
        print(f"[WARN] Analysis intent detected but structure is missing.")

# --- ADVISORY SYSTEM PROMPT ---
ADVISORY_SYSTEM_PROMPT = """
You are a sharp AlphaTrace Financial Analyst.

Your job is to answer the USER QUESTION directly using the provided STRICT DATA SIGNALS.

INSTRUCTIONS:
1. DIRECT ANSWER: First, directly answer the user's question in 1–2 decisive sentences. 
2. DATA SUPPORT: Then support your answer using specific data points (tickers, %, weights) from the provided signals.
3. RELEVANCE: Only use and explain information strictly relevant to the question.
4. ANALYST PERSONA: Be sharp and decisive. Speak like a lead analyst, not a report generator.
5. NO REPETITION: Do NOT generate a generic portfolio summary with fixed sections unless explicitly asked or the mode is 'structured'.

STYLE RULES:
- BANNED WORDS: "appears", "may", "suggests", "could", "likely", "portfolio shows diversified exposure", "analysis reveals".
- USE DECISIVE VERBS: "is driving", "has triggered", "is the primary reason", "is impacting".
- DATA MINIMUMS: Every response MUST include at least 1-2 tickers and numeric values.

RESPONSE MODE GUIDELINE:
{structure_guideline}
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

def generate_validated_response(input_data: dict, intents: List[str]) -> dict:
    """
    ADAPTIVE RESPONSE PIPELINE:
    intent -> structure -> generate -> evaluate -> repair -> validate
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except:
        return {"text": "Synthesis engine is currently offline.", "confidence": 0.0}

    # 1. STRUCTURE SELECTION (Part 4)
    user_query = input_data.get("user_query", "")
    q_low = user_query.lower()
    
    if "analyze" in q_low or "full" in q_low:
        mode = "structured"
        structure_guideline = STRUCTURES.get("full_analysis")
    else:
        mode = "natural"
        structure_guideline = get_structure_guideline(intents)

    print(f"[MODE] {mode}")
    print(f"[GUIDELINE] {structure_guideline}")

    # 2. INITIAL GENERATION (Part 1 & 2)
    dynamic_system_prompt = ADVISORY_SYSTEM_PROMPT.format(structure_guideline=structure_guideline)
    
    messages = [
        {"role": "system", "content": dynamic_system_prompt},
        {"role": "user", "content": f"USER QUESTION: {user_query}\n\nSTRICT DATA SIGNALS: {json.dumps(input_data['tool_outputs'])}"}
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
        return {"text": "Connection fault in reasoning engine.", "confidence": 0.0}

    # 3. INITIAL EVALUATION
    eval_result = evaluate_response(initial_draft)
    initial_score = eval_result.get("score", 0.0)
    initial_breakdown = eval_result.get("details", {})
    
    print(f"[EVAL] score={initial_score}")
    
    # Check for structural anomalies only in structured mode
    if mode == "structured":
        validate_structure(initial_draft, intents[0] if intents else "general")

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
        The previous response is missing mandatory data points: {', '.join(missing_elements)}.
        Sharp and decisively incorporate these metrics while answering the USER QUESTION. 
        Stick to the persona: decisive, data-backed, and brief.
        Do NOT rewrite the entire answer, just fix the missing metrics.
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
    response_payload = generate_validated_response(input_data, intents)
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
