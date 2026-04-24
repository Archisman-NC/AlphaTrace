import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESOLVER_SYSTEM_PROMPT = """
You are a context resolution engine for a financial AI system.
Your job is to resolve ambiguous user queries using session memory.

## TASK:
1. Determine if the query depends on previous context.
2. Resolve references like "it", "this", "that", or follow-up "why" questions.
3. Attach the correct portfolio and context.

## RULES:
- If query is ambiguous -> use last_analysis to expand it.
- If query contains "it/this/that" -> assume current_portfolio.
- If query is standalone (e.g. "show me PORTFOLIO_003") -> do not modify.
- NEVER invent new data.

## OUTPUT:
Return ONLY valid JSON:
{
  "resolved_query": "string",
  "portfolio_id": "string",
  "use_memory": boolean
}
"""

def resolve_context(user_query: str, session: dict) -> dict:
    """
    Resolves pronouns and ambiguous context in user queries.
    Returns structured resolution metadata.
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

    resolution_input = {
        "user_query": user_query,
        "session": session
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(resolution_input)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        
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
