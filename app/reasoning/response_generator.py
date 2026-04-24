import os
import json
import logging
import time
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

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

        start_time = time.time()
        
        system_msg = RESPONSE_SYSTEM_PROMPT
        user_msg = json.dumps(synthesis_input)
        
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="response_generation",
                metadata={"portfolio_id": portfolio_id, "stage": "generation", "user_persona": user_profile.get("experience_level")}
            )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        latency = time.time() - start_time
        output_text = response.choices[0].message.content.strip()
        
        if trace:
            trace.generation(
                name="generation_call",
                input={"system": system_msg, "user": user_msg},
                output=output_text,
                model="llama-3.3-70b-versatile",
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
        logger.error(f"Response generation failed: {e}")
        return "The analysis is complete, but I'm unable to generate a narrative briefing at this time. Here are the core metrics: " + json.dumps(tool_outputs)
