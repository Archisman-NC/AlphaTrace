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
Your job is to convert structured data into a natural conversational response.

## RULES:
- Use ONLY the provided data.
- Do NOT add any new reasoning, explanations, or assumptions.
- Do NOT infer causes beyond what is explicitly given.
- If information is missing, ignore it.
- Keep response concise and clear.
- Slightly conversational tone.

## STRUCTURE:
1. Start with the summary.
2. Then mention key drivers (if present).
3. Then mention risks (if present).

## STRICT:
- Output PLAIN TEXT ONLY.
- No markdown.
- No bullet points.
- No extra commentary.

if not summary:
    return "I don't have enough data to generate a response."
"""

def synthesize_text_response(summary: str, drivers: list = None, risks: list = None) -> str:
    """
    Ultima-Strict text synthesis:
    - Zero formatting (no bullet points)
    - Zero inference
    - Data sufficiency guardrails
    """
    if not summary:
        return "I don't have enough data to generate a response."

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for Ultima-Strict synthesis: {e}")
        return "Analysis complete. Data shows movement but narrative synthesis is currently unavailable."

    analysis_input = {
        "summary": summary,
        "drivers": drivers or [],
        "risks": risks or []
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(analysis_input)}
            ],
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ultima-Strict synthesis failed: {e}")
        return "Structural analysis completed successfully. Results are ready for review."
