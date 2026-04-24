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

CLASSIFIER_SYSTEM_PROMPT = """
You are a high-precision financial intent classification engine.
Use the active_memory context to disambiguate user queries.

## OUTPUT FORMAT (STRICT JSON):
{
  "intent": ["full_analysis", "reason", "risk", "switch_portfolio"],
  "portfolio_id": "PORTFOLIO_XXX",
  "confidence": float 
}

## RULES:
1. Use active_memory to infer the subject of follow-ups (e.g. "it", "that", "why?").
2. Focus on "risk" if active_memory shows high severity risks recently.
3. If user query matches recent drivers in memory, boost confidence for relevant intents.
"""

def classify_intent(query: str, current_portfolio: str, memory: list = None) -> dict:
    """
    High-precision classification utilizing Active Memory.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for high-precision classification: {e}")
        return {"intent": ["full_analysis"], "portfolio_id": current_portfolio, "confidence": 0.0}

    # Boost context using Active Memory Engine
    memory_context = extract_relevant_memory(query, memory or [])

    # Prepare context for classification
    classification_input = {
        "user_query": query,
        "current_portfolio": current_portfolio,
        "active_memory": memory_context # Weighted past insights
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
