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

CLASSIFICATION_SYSTEM_PROMPT = """
You are a high-precision financial intent classification engine.

Return STRICT JSON. No explanation.

## INPUT:
- user_query
- current_portfolio
- chat_history

## OUTPUT FORMAT (STRICT):
{
  "intent": ["full_analysis", "reason", "risk", "switch_portfolio"],
  "portfolio_id": "PORTFOLIO_XXX",
  "confidence": 0.0 
}
*Note: The confidence score must be a dynamic float between 0.0 and 1.0 based on query clarity.*

## RULES:
1. Detect ALL relevant intents.
2. Use chat_history to resolve references.
3. Portfolio handling: resolve or use current_portfolio.
4. CONFIDENCE SCORING:
   - 0.9-1.0: extremely clear
   - 0.7-0.89: moderate/good
   - 0.5-0.69: ambiguous
   - <0.5: very weak

5. Edge handling:
   - vague query → ["full_analysis"]
   - "why" → include "reason"
   - "safe/risk/downside" → include "risk"
   - "switch/change" → include "switch_portfolio"

## STRICT:
- Must return valid JSON
- No extra keys
- No missing fields
"""

def classify_intent(query: str, current_portfolio: str, chat_history: list = None) -> dict:
    """
    High-precision classification:
    - Supports multi-intent detection
    - Includes confidence scoring
    - Resolves portfolio_id from context
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
        "chat_history": chat_history[-5:] if chat_history else []
    }

        start_time = time.time()
        
        system_msg = CLASSIFICATION_SYSTEM_PROMPT
        user_msg = json.dumps(classification_input)
        
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="intent_classification",
                metadata={"portfolio_id": current_portfolio, "stage": "classification"}
            )

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
        # 1. Handle plural 'intents' key if model provides it
        if "intents" in result and "intent" not in result:
            result["intent"] = result.pop("intents")
        
        # 2. Ensure intent is always a list
        if isinstance(result.get("intent"), str):
            result["intent"] = [result["intent"]]
        elif "intent" not in result:
            result["intent"] = ["full_analysis"]

        # 3. Ensure portfolio_id exists
        if "portfolio_id" not in result:
            result["portfolio_id"] = current_portfolio
            
        # 4. Ensure confidence exists
        if "confidence" not in result:
            result["confidence"] = 0.5
            
        return result
    except Exception as e:
        logger.error(f"High-precision classification failed: {e}")
        return {"intent": ["full_analysis"], "portfolio_id": current_portfolio, "confidence": 0.0}
