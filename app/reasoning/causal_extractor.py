import os
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

CAUSAL_EXTRACTOR_PROMPT = """
You are a financial information extractor.
Given a news headline and summary, extract the core causal trigger in ONE short phrase.

RULES:
- Max 6–10 words
- No explanation
- No punctuation except hyphen
- No speculation

EXAMPLES:
Input: "RBI signals potential rate hike amid inflation concerns"
Output: "hawkish RBI rate outlook"

Return plain text only.
"""

def extract_causal_trigger(headline: str, summary: str) -> str:
    """
    Distills financial news into a concise causal trigger phrase.
    """
    if not headline:
        return "unknown market driver"

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Use llama-3.1-8b-instant for speed
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": CAUSAL_EXTRACTOR_PROMPT},
                {"role": "user", "content": f"Headline: {headline}\nSummary: {summary}"}
            ],
            temperature=0.1 # Absolute deterministic extraction
        )
        
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.error(f"Causal Extraction failed: {e}")
        # Graceful fallback to raw headline snippet
        return headline[:50].split(".")[0].strip().lower()
