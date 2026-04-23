import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """
You are a financial assistant.
Your job is to convert structured analysis into a natural, conversational response.

## RULES:
- Use ONLY the provided data.
- Do NOT add new reasoning, assumptions, or external information.
- Do NOT hallucinate.
- Keep it concise and clear.
- Slightly conversational tone.
- Avoid jargon unless already present.
- Do NOT repeat the same point.

## STRUCTURE:
- Start with a direct answer.
- Then briefly explain key drivers.
- Then mention risks (if provided).

## OUTPUT:
Return PLAIN TEXT ONLY. No markdown, no JSON, no bolding, no headers.
"""

def synthesize_text_response(analysis_data: dict) -> str:
    """
    Converts structured reasoning results into plain-text conversational advisory.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for text synthesis: {e}")
        return "Analysis complete. Data shows movement but narrative synthesis is currently unavailable."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(analysis_data)}
            ],
            temperature=0.0 # High precision, no creativity
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Text synthesis failed: {e}")
        return "Structural analysis completed successfully. Results are ready for review."
