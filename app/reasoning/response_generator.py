import os
import json
import logging
import time
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse
from app.evaluation.llm_evaluator import evaluate_explanation

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """
You are AlphaTrace, a financial AI copilot.

## CORE PRINCIPLES:
1. TRUTH OVER CONFIDENCE: If uncertain, say so.
2. SPECIFIC > GENERIC: Mention tickers (HDFCBANK, TCS, etc) and specific triggers.
3. DATA-DRIVEN: Use tool_outputs (%, changes).
4. MEMORY-PRIORITY: Link to past drivers discussed in memory_context.

## STYLE:
- Direct Answer -> Causal Explanation -> Actionable Insight.
- Length: 5-8 sentences.
"""

def _generate_base_response(client: Groq, prompt: str, system_msg: str) -> str:
    """Internal helper for raw generation."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

def generate_validated_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, memory_context: dict = None) -> str:
    """
    CENTRAL WRAPER: High-Fidelity Generation -> Evaluation -> Context-Aware Self-Correction.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Groq Init Failed: {e}")
        return "Synthesis engine is currently offline."

    # Prepare complete input context
    synthesis_input = {
        "user_query": user_query,
        "tool_outputs": tool_outputs,
        "memory_context": memory_context or {}
    }
    user_msg = json.dumps(synthesis_input)

    # LANGFUSE TRACE INITIALIZATION
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(name="validated_reasoning", metadata={"portfolio_id": portfolio_id})

    # 1. INITIAL GENERATION
    start_time = time.time()
    draft = _generate_base_response(client, user_msg, RESPONSE_SYSTEM_PROMPT)
    initial_latency = time.time() - start_time

    # 2. INITIAL EVALUATION
    # Pass draft to strict Llama judge
    eval_result = evaluate_explanation(draft, synthesis_input, portfolio_id)
    initial_score = eval_result.get("score", 0.0)
    
    logger.info(f"[AUDITOR] Initial Score: {initial_score}")

    # OPTIMIZED EARLY EXIT: If score is high enough, don't regenerate
    if initial_score >= 6.5:
        if trace:
            trace.generation(
                name="reasoning_turn",
                input=user_msg,
                output=draft,
                metadata={"initial_score": initial_score, "retry": False, "final_score": initial_score}
            )
            langfuse.flush()
        return draft

    # 3. CONTEXT-AWARE SELF-CORRECTION
    logger.info(f"TRIGGERING SELF-CORRECTION: Quality {initial_score} below threshold.")
    
    # We pass the SAME DATA + instruction + previous failure to maintain context
    correction_instruction = {
        "instruction": "IMPROVE ADVISORY. Previous version was scored low by auditor.",
        "auditor_feedback": eval_result.get("reason", "Missing specific tickers or quantification."),
        "requirements": "Mention specific tickers (e.g. HDFCBANK), include %, and match causal trigger.",
        "previous_draft": draft,
        "original_data": synthesis_input # CRITICAL: Keep data in context
    }
    
    start_time_retry = time.time()
    final_text = _generate_base_response(client, json.dumps(correction_instruction), RESPONSE_SYSTEM_PROMPT)
    retry_latency = time.time() - start_time_retry

    # 4. FINAL EVALUATION (For metrics sanity)
    final_eval = evaluate_explanation(final_text, synthesis_input, portfolio_id)
    final_score = final_eval.get("score", 0.0)
    logger.info(f"[AUDITOR] Final Score: {final_score}")

    if trace:
        trace.generation(
            name="reasoning_turn_with_retry",
            input=json.dumps(correction_instruction),
            output=final_text,
            metadata={
                "initial_score": initial_score,
                "retry": True,
                "final_score": final_score,
                "improvement": final_score - initial_score,
                "total_latency": initial_latency + retry_latency
            }
        )
        langfuse.flush()

    return final_text

def stream_final_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, memory_context: dict = None) -> Generator[str, None, None]:
    """
    Simulated Streaming for Validated Narratives.
    """
    final_text = generate_validated_response(
        user_query=user_query,
        intents=intents,
        portfolio_id=portfolio_id,
        tool_outputs=tool_outputs,
        memory_context=memory_context
    )

    words = final_text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.01)
