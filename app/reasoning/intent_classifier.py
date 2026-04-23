import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """
You are a high-precision intent classification engine for a financial AI system (AlphaTrace).

Your task is to classify user intent from a query, using context.

You MUST return STRICT JSON only. No explanation. No extra text.

---

## VALID INTENTS:

- "full_analysis"
- "reason"
- "risk"
- "switch_portfolio"

---

## RULES:

1. MULTI-INTENT:
- If query contains multiple intents → return multiple intents as array
- Example: "why did it fall and should I switch?" → ["reason", "switch_portfolio"]

2. CONTEXT RESOLUTION:
- Use chat_history to resolve "it", "this", "that"
- If unclear → assume current_portfolio

3. PORTFOLIO RESOLUTION:
- If user explicitly mentions a portfolio → use it
- Else → use current_portfolio
- NEVER invent new portfolio IDs

4. CONFIDENCE SCORING:
- 0.9–1.0 → very clear intent
- 0.7–0.89 → reasonably clear
- 0.5–0.69 → ambiguous
- <0.5 → very unclear

5. EDGE CASES:
- If query is vague → default to ["full_analysis"]
- If user asks “why” → include "reason"
- If user asks about safety, downside → include "risk"
- If user mentions switching → include "switch_portfolio"

6. STRICTNESS:
- Output MUST be valid JSON
- No markdown, no explanation, no trailing text
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
