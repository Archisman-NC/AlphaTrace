import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """
You are AlphaTrace, a financial AI copilot.

Your job is to give clear, honest, and useful financial insights based on portfolio analysis.

## CORE PRINCIPLES:
1. TRUTH OVER CONFIDENCE: If uncertain, say so. Do not pretend certainty.
2. SPECIFIC > GENERIC: Use concrete reasons from tool_outputs. Avoid vague phrases.
3. USER-AWARE: 
   - Beginner: simpler explanations.
   - Advanced: deeper reasoning.
   - Low Risk: emphasize downside.
   - High Risk: less cautious tone.
4. SAFETY: Do NOT give absolute financial advice. Use "suggests", "might consider", etc.

## STYLE:
- Conversational, direct, and jargon-free.
- Structure: Direct Answer -> Explanation -> Risk Context -> Actionable Insight.
- Length: 5-8 sentences max.
"""

def generate_advisory_response(user_query: str, intents: list, portfolio_id: str, tool_outputs: dict, user_profile: dict = None) -> str:
    """
    Synthesizes tool outputs into a cohesive, personalized advisory response.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq for response generation: {e}")
        return "I'm sorry, I'm having trouble synthesizing the analysis results right now. Please try again in a moment."

    # Default profile if none provided
    if not user_profile:
        user_profile = {"risk_tolerance": "medium", "experience_level": "intermediate"}

    synthesis_input = {
        "user_query": user_query,
        "intents": intents,
        "portfolio_id": portfolio_id,
        "tool_outputs": tool_outputs,
        "user_profile": user_profile
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
