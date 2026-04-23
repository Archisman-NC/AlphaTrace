import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """
You are an intent classification engine for a financial AI assistant.

Your ONLY job is to classify the user's intent based on their query and context.

You MUST return valid JSON only. No explanation. No extra text.

---

## POSSIBLE INTENTS:

1. "full_analysis"
→ User wants a complete breakdown of portfolio performance, metrics, or recommendations

2. "reason"
→ User is asking WHY something happened (e.g., "why did it fall?", "why is this down?")

3. "risk"
→ User is asking about risk, volatility, downside, safety

4. "switch_portfolio"
→ User wants to change portfolio or is referring to another portfolio

---

## RULES:

- Use chat_history to resolve ambiguity
- If the user says "this", "it", etc → infer from context
- If unclear → choose the closest intent (DO NOT return unknown)
- Always include portfolio_id:
    - If user mentions a portfolio → use it
    - Else → use current_portfolio

- DO NOT hallucinate new portfolio IDs
"""

def classify_intent(query: str, current_portfolio: str, chat_history: list = None) -> dict:
    """
    Classifies the user's intent and extracts/resolves the portfolio_id.
    Returns a dict with 'intent' and 'portfolio_id'.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for intent classification: {e}")
        return {"intent": "full_analysis", "portfolio_id": current_portfolio}

    # Prepare input for classification
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
        return result
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {"intent": "full_analysis", "portfolio_id": current_portfolio}
