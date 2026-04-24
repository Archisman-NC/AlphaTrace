import os
import logging
from openai import OpenAI
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

# Initialize client properly
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None

def polish_response(raw_text: str, intents: list, context: dict, confidence: float) -> str:
    """
    Polishes the raw reasoning output into a professional advisor tone.
    Guaranteed to return at least the raw_text.
    """
    if not client:
        return raw_text

    try:
        system_prompt = """
        You are a senior financial advisor polisher for AlphaTrace. 
        Take the PROVIDED raw reasoning and ensure:
        1. Professional, institutional tone.
        2. High-precision numeric focus.
        3. Clear structure.
        
        STRICT: Do NOT invent new facts. ONLY refine the existing narrative.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            temperature=0.1
        )
        return str(response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"Polishing execution fault: {e}")
        return raw_text
