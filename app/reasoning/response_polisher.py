import os
import logging
from openai import OpenAI
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

def polish_response(raw_text: str, intents: list, context: dict, confidence: float) -> str:
    """
    Polishes the raw reasoning output into a professional, institutional-grade advisor tone.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing - skipping polishing phase.")
        return raw_text

    try:
        client = OpenAI(api_key=api_key)
        
        system_prompt = """
        You are a senior financial advisor polisher. 
        Take the provided raw reasoning and ensure:
        1. Professional and institutional tone (no chatty filler).
        2. High-precision numeric focus.
        3. Clear structure (Summary, Risks, Metrics).
        
        STRICT: Do NOT add new facts. ONLY refine the existing text.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Polishing phase failed: {e}")
        return raw_text
