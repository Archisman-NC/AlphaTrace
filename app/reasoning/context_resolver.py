import os
import json
import logging
import time
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse
from app.reasoning.memory_engine import extract_relevant_memory

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESOLVER_SYSTEM_PROMPT = """
You are a context resolution engine for a financial AI system.
Your job is to resolve ambiguous user queries using structured memory episodes.

## TASK:
1. Determine if the query depends on previous context (memory).
2. Resolve pronouns like "it", "this", "that", or follow-up "why" questions.
3. Identify the target portfolio ID from continuity.

## INPUT:
- user_query: current query
- current_portfolio: active selection
- recent_episodes: list of past turns with summary, drivers, and intents.

## OUTPUT:
{
  "resolved_query": "expanded string",
  "portfolio_id": "resolved ID",
  "use_memory": boolean
}
"""

def resolve_context(user_query: str, session: dict) -> dict:
    """
    Resolves pronouns and ambiguous context using structured memory.
    Returns resolution metadata including target portfolio and expanded query.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for context resolution: {e}")
        return {
            "resolved_query": user_query,
            "portfolio_id": session.get("current_portfolio", "N/A"),
            "use_memory": False
        }

    # Use Active Memory engine for prioritization
    memory = session.get("memory", [])
    memory_context = extract_relevant_memory(user_query, memory)
    
    resolution_input = {
        "user_query": user_query,
        "current_portfolio": session.get("current_portfolio"),
        "active_memory": memory_context # Prioritized context
    }

    start_time = time.time()
    
    system_msg = RESOLVER_SYSTEM_PROMPT
    user_msg = json.dumps(resolution_input)
    
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(
            name="context_resolution",
            metadata={"portfolio_id": session.get("current_portfolio"), "stage": "context"}
        )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        latency = time.time() - start_time
        output_text = response.choices[0].message.content
        
        if trace:
            trace.generation(
                name="context_call",
                input={"system": system_msg, "user": user_msg},
                output=output_text,
                model="llama-3.1-8b-instant",
                usage={
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
                },
                metadata={"latency": latency}
            )
            langfuse.flush()

        result = json.loads(output_text)
        
        # Fallback validation
        if not result.get("portfolio_id"):
            result["portfolio_id"] = session.get("current_portfolio", "N/A")
            
        return result
    except Exception as e:
        logger.error(f"Context resolution failed: {e}")
        return {
            "resolved_query": user_query,
            "portfolio_id": session.get("current_portfolio", "N/A"),
            "use_memory": False
        }
