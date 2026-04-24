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

VALIDATOR_SYSTEM_PROMPT = """
You are a validation and routing engine for a financial AI system (AlphaTrace).

Your job is to:
1. Validate the correctness of an intent classification
2. Decide what action the system should take (execute | clarify | fallback)

Return STRICT JSON only.

## ACTION RULES:
- EXECUTE: confidence >= 0.5 (PERMISSIVE), or classification is logical.
- CLARIFY: Only if query is truly unintelligible or missing target data.
- FALLBACK: Only if query is clearly unrelated to finance.

## CONVERSATIONAL RULES:
- Short queries (e.g. "Why?", "What happened?") are VALID if a portfolio context exists.
- Proactive follow-ups (e.g. "Analyze rebalancing") MUST be marked as 'execute'.
- If classification is mostly right, fix it and 'execute'. DO NOT 'clarify' unless stuck.

## STRICT:
- Output MUST be valid JSON
- Keep "reason" under 15 words
"""

def validate_and_route(user_query: str, classification: dict) -> dict:
    """
    Validates intent classification with a bias towards execution for a smoother UX.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for intent validation: {e}")
        return {
            "action": "fallback",
            "validated_intent": classification.get("intent", ["full_analysis"]),
            "portfolio_id": classification.get("portfolio_id", "N/A"),
            "confidence": 0.0,
            "reason": "Internal validation system error"
        }

    validation_input = {
        "user_query": user_query,
        "classification": classification
    }

    start_time = time.time()
    
    system_msg = VALIDATOR_SYSTEM_PROMPT
    user_msg = json.dumps(validation_input)
    
    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(
            name="intent_validation",
            metadata={"portfolio_id": classification.get("portfolio_id"), "stage": "validation"}
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
        
        output_text = response.choices[0].message.content
        result = json.loads(output_text)
        
        # NORMALIZE SCHEMA
        if "validated_intent" not in result:
            result["validated_intent"] = classification.get("intent", ["full_analysis"])
        if isinstance(result.get("validated_intent"), str):
            result["validated_intent"] = [result["validated_intent"]]
            
        if "action" not in result:
            result["action"] = "execute" if classification.get("confidence", 0) > 0.5 else "clarify"
        result["action"] = result["action"].lower()

        if "portfolio_id" not in result:
            result["portfolio_id"] = classification.get("portfolio_id", "N/A")
        if "confidence" not in result:
            result["confidence"] = classification.get("confidence", 0.0)
            
        return result
    except Exception as e:
        logger.error(f"Intent validation failed: {e}")
        return {
            "action": "execute", # Default to execute for resilience
            "validated_intent": classification.get("intent", ["full_analysis"]),
            "portfolio_id": classification.get("portfolio_id", "N/A"),
            "confidence": 0.5,
            "reason": "Graceful fallback to execution"
        }
