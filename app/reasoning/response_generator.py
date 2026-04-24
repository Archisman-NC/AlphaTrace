import os
import json
import logging
import time
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """
You are AlphaTrace, a financial AI copilot.

## CORE PRINCIPLES:
1. TRUTH OVER CONFIDENCE: If uncertain, say so.
2. SPECIFIC > GENERIC: Use tool_outputs results.
3. USER-AWARE: Adjust tone to risk/experience profile.
4. MEMORY-PRIORITY: Use memory_context (past drivers/risks) to explain current shifts.
   - Example: "Building on the [last driver] we discussed, we now see..."

## STYLE:
- Conversational, direct, and structured.
- Structure: Direct Answer -> Causal Explanation (linking memory) -> Actionable Insight.
- Length: 5-8 sentences.

PRIORITIZE memory_context (recent drivers/risks) when explaining the answer.
"""

def generate_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, user_profile: dict = None, memory_context: dict = None) -> str:
    """
    Synthesizes narrative with Active Memory injection.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Groq Init Failed: {e}")
        return "Synthesis engine offline."

    synthesis_input = {
        "user_query": user_query,
        "intents": intents,
        "portfolio_id": portfolio_id,
        "tool_outputs": tool_outputs,
        "user_profile": user_profile or {},
        "memory_context": memory_context or {} # Injected memory
    }

    start_time = time.time()
    system_msg = RESPONSE_SYSTEM_PROMPT
    user_msg = json.dumps(synthesis_input)
    
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(name="response_generation", metadata={"stage": "generation"})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.7,
            max_tokens=600
        )
        output_text = response.choices[0].message.content.strip()
        
        if trace:
            trace.generation(
                name="generation_call",
                input={"system": system_msg, "user": user_msg},
                output=output_text,
                model="llama-3.3-70b-versatile",
                metadata={"latency": time.time() - start_time}
            )
            langfuse.flush()
        return output_text
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "I encountered an error synthesizing the temporal context."

def stream_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, user_profile: dict = None, memory_context: dict = None) -> Generator[str, None, None]:
    """
    Streaming narrative synthesis with Active Memory.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Streaming Init Failed: {e}")
        yield "Synthesis engine offline."
        return

    synthesis_input = {
        "user_query": user_query,
        "intents": intents,
        "portfolio_id": portfolio_id,
        "tool_outputs": tool_outputs,
        "user_profile": user_profile or {},
        "memory_context": memory_context or {}
    }

    start_time = time.time()
    system_msg = RESPONSE_SYSTEM_PROMPT
    user_msg = json.dumps(synthesis_input)
    
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(name="response_streaming", metadata={"stage": "generation"})

    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.7,
            max_tokens=600,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                yield token
        
        if trace:
            trace.generation(
                name="streaming_call",
                input={"system": system_msg, "user": user_msg},
                output=full_response,
                model="llama-3.3-70b-versatile",
                metadata={"latency": time.time() - start_time}
            )
            langfuse.flush()
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield "Connection timeout during temporal briefing."
        return
