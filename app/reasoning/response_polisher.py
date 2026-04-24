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

Rewrite the response to improve clarity, flow, and tone.

## RULES:
- Preserve ALL original facts exactly. Do not change meaning in any way.
- Do NOT add new information or reasoning.
- Do NOT remove important details.
- Make it clear, concise, and conversational.
- Keep it within 5–8 sentences.
- Avoid repetition.

## PERSONALIZATION:
- Beginner → simpler explanation.
- Advanced → slightly deeper explanation.
- Low risk → emphasize caution.
- High risk → neutral/strategic tone.

## FOLLOW-UP:
- Optionally add ONE helpful follow-up sentence.
- Only if it improves usefulness.
- Keep it short and natural.

## OUTPUT:
Plain text only.
Final answer + optional follow-up on a new line.
"""

def should_use_openai(response: str, intents: list, confidence: float) -> bool:
    """
    Intelligent trigger logic to prevent over-triggering OpenAI.
    """
    return (
        len(intents) > 1 or
        confidence < 0.7 or
        len(response.split()) > 80
    )

def polish_response(raw_response: str, intents: list, user_profile: dict, confidence: float = 1.0) -> str:
    """
    Conditionally triggers OpenAI to polish the response for premium quality.
    Includes cost-control and failure-handling guardrails.
    """
    if not should_use_openai(raw_response, intents, confidence):
        logger.info("Skipping OpenAI: Response is concise and high-confidence.")
        return raw_response

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Use gpt-4o-mini for efficient premium polishing with cost cap
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
            temperature=0.4,
            max_tokens=200 # Financial cost-control
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Polisher failed: {e}. Falling back to raw response.")
        # Safe Fallback: Never break UX due to API failure
        return raw_response
