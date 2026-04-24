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
    CENTRAL WRAPER: Generates -> Evaluates -> Self-Corrects in a single production gated loop.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Groq Init Failed: {e}")
        return "Synthesis engine is currently offline."

    synthesis_input = {
        "user_query": user_query,
        "tool_outputs": tool_outputs,
        "memory_context": memory_context or {}
    }
    user_msg = json.dumps(synthesis_input)

    # 1. INITIAL GENERATION (Draft)
    draft = _generate_base_response(client, user_msg, RESPONSE_SYSTEM_PROMPT)

    # 2. EVALUATION
    # Pass draft to strict Llama judge
    eval_result = evaluate_explanation(draft, synthesis_input, portfolio_id)
    score = eval_result.get("score", 0.0)
    
    logger.info(f"[AUDITOR] Draft Score: {score}")

    # 3. SELF-CORRECTION (One Retry)
    if score < 6.5: # Strict threshold for AlphaTrace
        logger.info("TRIGGERING SELF-CORRECTION: Quality threshold breach.")
        
        correction_prompt = f"""
        IMPROVE THIS ADVISORY. It was scored low ({score}/10) by our internal auditor.
        REASON: {eval_result.get('reason', 'Too generic or missing tickers.')}
        
        REQUIRED IMPROVEMENTS:
        - Mention specific tickers from tool_outputs (e.g. HDFCBANK, TCS).
        - Include percentage performance changes.
        - Link directly to the causal news trigger.
        
        PREVIOUS DRAFT: {draft}
        """
        
        final_text = _generate_base_response(client, correction_prompt, RESPONSE_SYSTEM_PROMPT)
        logger.info("[AUDITOR] Self-Correction Complete. Final response gated.")
        return final_text

    return draft

def stream_final_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, memory_context: dict = None) -> Generator[str, None, None]:
    """
    STREAMING COMPATIBILITY LAYER:
    Generates validated response FIRST, then pipes tokens to UI at a natural cadence.
    Ensures that "Typing UX" does NOT bypass the "Reasoning Evaluator".
    """
    # Generate full validated text behind the scenes
    final_text = generate_validated_response(
        user_query=user_query,
        intents=intents,
        portfolio_id=portfolio_id,
        tool_outputs=tool_outputs,
        memory_context=memory_context
    )

    # Break into tokens and yield for Streamlit st.write_stream
    # We use a tiny sleep to simulate professional typing flow
    words = final_text.split(" ")
    for i, word in enumerate(words):
        yield word + " "
        time.sleep(0.01) # Professional cadence (approx 100-150 wpm UI feel)
