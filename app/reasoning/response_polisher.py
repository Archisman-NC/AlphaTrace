import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

POLISHER_SYSTEM_PROMPT = """
You are a financial AI copilot.
Your job is to improve the clarity and usefulness of a response, and optionally add one helpful follow-up.

## TASK:
1. Rewrite the response to:
- improve clarity and flow
- remove repetition
- keep it concise (5–8 sentences)
- make it slightly conversational

2. Do NOT:
- add new facts or reasoning
- change meaning

3. Adapt tone:
- beginner -> simpler
- advanced -> slightly deeper
- low risk -> emphasize caution
- high risk -> neutral/strategic tone

## FOLLOW-UP:
- Optionally add ONE short follow-up sentence (e.g., "Want me to break down which stocks contributed most?")
- Keep it natural, not pushy.

## OUTPUT:
Return plain text only. Final answer + optional follow-up on a new line.
"""

def polish_response(raw_response: str, intents: list, user_profile: dict, confidence: float = 1.0) -> str:
    """
    Conditionally triggers OpenAI to polish the response for premium quality.
    """
    # Trigger logic: Use OpenAI for expression complexity or ambiguity
    use_openai = (
        len(raw_response) > 200 
        or len(intents) > 1 
        or confidence < 0.7
    )

    if not use_openai:
        logger.info("Skipping OpenAI polish: Response is concise and high-confidence.")
        return raw_response

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Use gpt-4o-mini for efficient premium polishing
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": POLISHER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "response": raw_response,
                    "intents": intents,
                    "user_profile": user_profile
                })}
            ],
            temperature=0.4
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Polisher failed: {e}")
        return raw_response
