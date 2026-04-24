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
- EXECUTE: confidence >= 0.7, intent is logical, portfolio_id is valid.
- CLARIFY: confidence 0.4-0.69, or query is ambiguous.
- FALLBACK: confidence < 0.4, or classification is clearly wrong/invalid.

## VALIDATION RULES:
- If query contains "why" -> must include "reason"
- If query contains "risk/safe/downside" -> must include "risk"
- If query contains "switch/change" -> must include "switch_portfolio"
- If classification misses obvious intent -> downgrade action to "clarify"
- If classification includes irrelevant intent -> remove it

## STRICT:
- Output MUST be valid JSON
- Keep "reason" under 15 words
"""

def validate_and_route(user_query: str, classification: dict) -> dict:
    """
    Validates the intent classification result and determines the routing action.
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
                name="validation_call",
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
        # 1. Ensure validated_intent exists and is a list
        if "validated_intent" not in result:
            result["validated_intent"] = classification.get("intent", ["full_analysis"])
        if isinstance(result.get("validated_intent"), str):
            result["validated_intent"] = [result["validated_intent"]]
            
        # 2. Ensure action exists and is lowercase
        if "action" not in result:
            result["action"] = "fallback"
        result["action"] = result["action"].lower()

        # 3. Ensure portfolio_id and confidence are passed through if missing
        if "portfolio_id" not in result:
            result["portfolio_id"] = classification.get("portfolio_id", "N/A")
        if "confidence" not in result:
            result["confidence"] = classification.get("confidence", 0.0)
            
        return result
    except Exception as e:
        logger.error(f"Intent validation failed: {e}")
        return {
            "action": "fallback",
            "validated_intent": classification.get("intent", ["full_analysis"]),
            "portfolio_id": classification.get("portfolio_id", "N/A"),
            "confidence": 0.0,
            "reason": "AI validation exception"
        }
