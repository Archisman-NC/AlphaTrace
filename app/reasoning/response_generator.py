import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """
You are a financial AI assistant (AlphaTrace).

Your job is to generate a clear, insightful, and user-friendly response based on analysis results.

## STYLE:
- Conversational and structured (like ChatGPT)
- Direct, concise, and jargon-free
- No fluff or generic generic phrases like "it depends"

## RESPONSE RULES:
1. PRIORITIZE INTENT: Address why things moved, safety/risk, or recommendations based on the provided intents.
2. FUSION: Combine tool outputs naturally without repeating information.
3. STRUCTURE:
   - Direct answer first.
   - Clear explanation second.
   - Actionable suggestion or summary third.
4. TONE: Confident, helpful, and professional.
5. LENGTH: 5-8 sentences (expand slightly only for multi-intent complexity).
"""

def generate_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict) -> str:
    """
    Synthesizes tool outputs into a cohesive, conversational advisory response.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for response generation: {e}")
        return "I'm sorry, I'm having trouble synthesizing the analysis results right now. Please try again in a moment."

    synthesis_input = {
        "user_query": user_query,
        "intents": intents,
        "portfolio_id": portfolio_id,
        "tool_outputs": tool_outputs
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(synthesis_input)}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "The analysis is complete, but I'm unable to generate a narrative briefing at this time. Here are the core metrics: " + json.dumps(tool_outputs)
