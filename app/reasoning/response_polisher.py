import os
import json
import logging
import time
from openai import OpenAI
from dotenv import load_dotenv
from app.utils.helpers import langfuse

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

        start_time = time.time()
        
        system_msg = POLISHER_SYSTEM_PROMPT
        user_msg = json.dumps({
            "response": raw_response,
            "intents": intents,
            "user_profile": user_profile
        })
        
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="premium_polish",
                metadata={"stage": "polish", "user_persona": user_profile.get("experience_level")}
            )

        # Use gpt-4o-mini for efficient premium polishing with cost cap
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4,
            max_tokens=200 # Financial cost-control
        )
        
        latency = time.time() - start_time
        output_text = response.choices[0].message.content.strip()
        
        if trace:
            trace.generation(
                name="polish_call",
                input={"system": system_msg, "user": user_msg},
                output=output_text,
                model="gpt-4o-mini",
                usage={
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
                },
                metadata={"latency": latency}
            )
            langfuse.flush()
            
        return output_text
    except Exception as e:
        logger.error(f"OpenAI Polisher failed: {e}. Falling back to raw response.")
        # Safe Fallback: Never break UX due to API failure
        return raw_response
