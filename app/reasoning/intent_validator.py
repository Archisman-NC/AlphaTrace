import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

VALID_PORTFOLIOS = ["PORTFOLIO_001 (Rahul)", "PORTFOLIO_002 (Priya)", "PORTFOLIO_003 (Arun)"]

VALIDATOR_SYSTEM_PROMPT = f"""
You are the Advisory Guardian for AlphaTrace. 
Your mission is to ensure the AI actions queries whenever possible.

## DATA UNIVERSE:
- Valid Portfolios: {", ".join(VALID_PORTFOLIOS)}
- Valid Intents: reason (causal), risk (hazards), switch_portfolio, full_analysis.

## ACTION RULES (BE PERMISSIVE):
- ACTION: Confidence >= 0.4.
- DEFAULT INTENT: If unsure, use "full_analysis".
- CLARIFY: ONLY if BOTH intent AND portfolio_id are completely missing.

## PATTERN OVERRIDES:
- "best stock", "top performing", "ranking" -> ranking intent
- "analysis", "breakdown", "check", "portfolio" -> full_analysis intent
- "risk", "hazard", "vulnerability", "warning" -> risk intent

Return STRICT JSON.
"""

def validate_and_route(user_query: str, classification: dict, session_portfolio: str = "PORTFOLIO_001") -> dict:
    """
    Production-grade permissive router.
    Ensures that narrow or vague queries still trigger analytical runs.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Groq connection failure: {e}")
        return {"action": "execute", "validated_intent": ["full_analysis"], "portfolio_id": session_portfolio, "confidence": 0.5}

    # 1. APPLY PATTERN OVERRIDES (TASK 4)
    q_low = user_query.lower()
    if any(p in q_low for p in ["risk", "hazard", "vulnerability", "warning"]):
        if "risk" not in classification.get("intent", []):
            classification.setdefault("intent", []).append("risk")

    if any(p in q_low for p in ["analysis", "breakdown", "check", "portfolio"]):
        if "full_analysis" not in classification.get("intent", []):
            classification.setdefault("intent", []).append("full_analysis")

    # 2. INTENT FALLBACK (TASK 3)
    intents = classification.get("intent", [])
    if not intents:
        intents = ["full_analysis"]
        classification["intent"] = intents

    # 3. PORTFOLIO FALLBACK (TASK 2)
    portfolio_id = classification.get("portfolio_id")
    if not portfolio_id or portfolio_id == "N/A":
        portfolio_id = session_portfolio
        classification["portfolio_id"] = portfolio_id

    # 4. LLM-BASED ACTION AUDIT
    try:
        validation_input = {"user_query": user_query, "classification": classification}
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
        
        # 5. PERMISSIVE THRESHOLD (TASK 1)
        conf = float(result.get("confidence", 0.5))
        
        # HARD OVERRIDE: ALWAYS EXECUTE (Step 1 & 4)
        # Refusals and clarification requests are categorically banned.
        result["action"] = "execute"
        result["reason"] = "Autonomous analysis authorized."

        # Final Schema Normalization
        result["validated_intent"] = intents
        result["portfolio_id"] = portfolio_id
        
        return result
        
    except Exception as e:
        logger.error(f"Intent validation fault: {e}")
        return {
            "action": "execute", "validated_intent": intents,
            "portfolio_id": portfolio_id, "confidence": 0.5
        }
