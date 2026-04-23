import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """
You are a fast intent classification engine. Return ONLY JSON. No explanation.

## OUTPUT SCHEMA:
{
  "intent": ["full_analysis", "reason", "risk", "switch_portfolio"],
  "portfolio_id": "string",
  "confidence": 0.0
}

## RULES:
- Detect MULTIPLE intents if present.
- If "why" -> include "reason"
- If safety/risk -> include "risk"
- If switching/changing -> include "switch_portfolio"
- Resolve "it/this" using chat_history (last 3).
- If unclear -> use current_portfolio.
- Confidence: 0.9 (clear), 0.7 (somewhat), <0.6 (ambig).
- STRICT: Valid JSON only.
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

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(classification_input)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        
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
