import os
import json
import logging
import time
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """
You are a high-precision financial intent classification engine.
Use structured memory to disambiguate user queries and resolve portfolio context.

## INPUT:
- user_query: current user input
- current_portfolio: active selection
- memory: list of structured past turns (intents, summary, drivers, risks)

## OUTPUT FORMAT (STRICT):
{
  "intent": ["full_analysis", "reason", "risk", "switch_portfolio"],
  "portfolio_id": "PORTFOLIO_XXX",
  "confidence": float 
}

## RULES:
1. If query is a follow-up (e.g. "why?"), infer the intent from the previous memory turns.
2. If query mentions "risk/safe/danger", include "risk".
3. Use memory to detect if the user's "this" refers to a specific stock/sector from last turn.
"""

def classify_intent(query: str, current_portfolio: str, memory: list = None) -> dict:
    """
    High-precision classification utilizing structured memory.
    - Disambiguates vague queries using past intents/drivers.
    - Resolves portfolio_id from continuity context.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for high-precision classification: {e}")
        return {"intent": ["full_analysis"], "portfolio_id": current_portfolio, "confidence": 0.0}

    # Prepare context for classification
    classification_input = {
        "user_query": query,
        "current_portfolio": current_portfolio,
        "memory": memory[-3:] if memory else [] # Only last few turns to avoid noise
    }

    start_time = time.time()
    
    system_msg = CLASSIFIER_SYSTEM_PROMPT
    user_msg = json.dumps(classification_input)
    
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(
            name="intent_classification",
            metadata={"portfolio_id": current_portfolio, "stage": "classification"}
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
                name="classification_call",
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
        
        # NORMALIZE SCHEMA
        if "intents" in result and "intent" not in result:
            result["intent"] = result.pop("intents")
        
        if isinstance(result.get("intent"), str):
            result["intent"] = [result["intent"]]
        elif "intent" not in result:
            result["intent"] = ["full_analysis"]

        if "portfolio_id" not in result:
            result["portfolio_id"] = current_portfolio
            
        if "confidence" not in result:
            result["confidence"] = 0.5
            
        return result
    except Exception as e:
        logger.error(f"High-precision classification failed: {e}")
        return {"intent": ["full_analysis"], "portfolio_id": current_portfolio, "confidence": 0.0}
