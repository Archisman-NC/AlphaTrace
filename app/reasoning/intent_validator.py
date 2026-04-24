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

VALID_PORTFOLIOS = ["PORTFOLIO_001 (Rahul)", "PORTFOLIO_002 (Priya)", "PORTFOLIO_003 (Arun)"]

VALIDATOR_SYSTEM_PROMPT = f"""
You are the Advisory Guardian for AlphaTrace. 
Your mission is to ensure the AI only executes tasks it has data for.

## DATA UNIVERSE:
- Valid Portfolios: {", ".join(VALID_PORTFOLIOS)}
- Valid Intents: reason (causal analysis), risk (hazard detection), switch_portfolio (context change).

## ACTION RULES:
- EXECUTE: Confidence >= 0.5 and query maps to valid data.
- CLARIFY: Query is vague or asks for a user/portfolio not in the universe.
- FALLBACK: Query is non-financial or inappropriate.

## MANDATORY USER FEEDBACK:
If you action is 'clarify', the 'reason' must be a polite instruction.
- Example: "I don't have data for that user. I can assist with Rahul, Priya, or Arun's portfolios."
- Example: "Could you specify what kind of analysis you're looking for?"

## STRICT NO-LEAK:
- NEVER mention internal error names or data logic.
- Return STRICT JSON.
"""

def validate_and_route(user_query: str, classification: dict) -> dict:
    """
    Ensures intent is grounded and user-safe.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for intent validation: {e}")
        return {
            "action": "fallback", "validated_intent": ["full_analysis"],
            "portfolio_id": "N/A", "confidence": 0.0,
            "reason": "I'm having trouble connecting to my reasoning engine. Please try again."
        }

    validation_input = {
        "user_query": user_query,
        "classification": classification
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(validation_input)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # NORMALIZE SCHEMA
        if "action" not in result: result["action"] = "execute"
        if "validated_intent" not in result: result["validated_intent"] = classification.get("intent", ["full_analysis"])
        
        # Sanitize 'reason' to ensure no internal leakage
        if result.get("action") == "clarify" and not result.get("reason"):
            result["reason"] = "Could you please provide more details so I can assist you better?"
            
        return result
    except Exception as e:
        logger.error(f"Intent validation failed: {e}")
        return {
            "action": "execute", "validated_intent": classification.get("intent", ["full_analysis"]),
            "portfolio_id": classification.get("portfolio_id", "N/A"),
            "confidence": 0.5, "reason": "Self-corrected execution"
        }
