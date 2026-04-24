import os
import logging
import time
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

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

        start_time = time.time()
        
        system_msg = CAUSAL_EXTRACTOR_PROMPT
        user_msg = f"Headline: {headline}\nSummary: {summary}"
        
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="causal_extraction",
                metadata={"stage": "extraction"}
            )

        # Use llama-3.1-8b-instant for speed
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1 # Absolute deterministic extraction
        )
        
        latency = time.time() - start_time
        output_text = response.choices[0].message.content.strip().lower()
        
        if trace:
            trace.generation(
                name="extraction_call",
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
            
        return output_text
    except Exception as e:
        logger.error(f"Causal Extraction failed: {e}")
        # Graceful fallback to raw headline snippet
        return headline[:50].split(".")[0].strip().lower()
