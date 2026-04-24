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
Goal: Provide clear, honest, and causal insights.

## CORE PRINCIPLES:
1. TRUTH OVER CONFIDENCE: If uncertain, say so.
2. SPECIFIC > GENERIC: Mention tickers (HDFCBANK, TCS, etc) and specific triggers.
3. DATA-DRIVEN: Use tool_outputs (%, changes).
4. MEMORY-PRIORITY: Building on memory_context (past drivers).

## STYLE:
- Direct Answer -> Causal Explanation -> Actionable Insight.
- Length: 5-8 sentences.
"""

def generate_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, user_profile: dict = None, memory_context: dict = None) -> str:
    """
    Synthesizes narrative with automated self-correction if quality is low.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Groq Init Failed: {e}")
        return "Synthesis engine offline."

    # 1. INITIAL GENERATION
    synthesis_input = {
        "user_query": user_query,
        "portfolio_id": portfolio_id,
        "tool_outputs": tool_outputs,
        "memory_context": memory_context or {}
    }
    
    system_msg = RESPONSE_SYSTEM_PROMPT
    user_msg = json.dumps(synthesis_input)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.7,
            max_tokens=600
        )
        draft = response.choices[0].message.content.strip()

        # 2. EVALUATION & SELF-CORRECTION
        eval_result = evaluate_explanation(draft, synthesis_input, portfolio_id)
        score = eval_result.get("score", 0.0)
        
        logger.info(f"[EVALUATOR] Draft Score: {score}")

        if score < 6.0:
            logger.info("Triggering Self-Correction: Reasoning quality below threshold.")
            correction_instruction = f"""
            IMPROVE THIS RESPONSE. It was scored low ({score}) by an internal judge.
            FIX: {eval_result.get('reason', 'Generic reasoning detected.')}
            REQUIREMENT: Be more specific. Mention tickers from tool_outputs. Include percentage changes.
            Link specifically to the trigger.
            PREVIOUS DRAFT: {draft}
            """
            
            corrected_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": correction_instruction}
                ],
                temperature=0.4 # More deterministic for correction
            )
            final_response = corrected_response.choices[0].message.content.strip()
            logger.info(f"[SELF-CORRECTION] Complete. Final quality improved.")
            return final_response
            
        return draft
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "I encountered an error during temporal synthesis."

def stream_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, user_profile: dict = None, memory_context: dict = None) -> Generator[str, None, None]:
    """
    Streaming version (Note: Self-correction is disabled for streaming to preserve low latency).
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        synthesis_input = {"user_query": user_query, "tool_outputs": tool_outputs, "memory_context": memory_context or {}}
        
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(synthesis_input)}
            ],
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token: yield token
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield "Briefing aborted due to connection error."
