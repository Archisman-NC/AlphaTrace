import os
import json
import logging
from typing import Dict, List, Any
from groq import Groq

logger = logging.getLogger(__name__)

def generate_llm_explanation(
    portfolio_metrics: Dict[str, Any],
    top_drivers: List[dict],
    conflicts: List[dict],
    risks: List[dict]
) -> dict:
    """
    Sends structured portfolio data to Groq LLM to generate a concise, human-readable
    causal explanation in strict JSON format.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq client (check GROQ_API_KEY): {e}")
        return {
            "summary": "Error: AI generation unavailable. Please check API configuration.",
            "drivers": top_drivers,
            "risks": risks
        }

    # Step 3: Build Structured Input
    payload = {
        "portfolio_change": portfolio_metrics.get("daily_change_percent", 0.0),
        "top_drivers": top_drivers,
        "conflicts": conflicts,
        "risks": risks
    }
    
    # Ensure no UNCLASSIFIED leaks into the prompt
    clean_payload_str = json.dumps(payload, indent=2).replace("UNCLASSIFIED", "diversified holdings")

    # Step 4: System Prompt (Strict JSON)
    system_prompt = """
You are a financial analyst.

Explain portfolio movement clearly and concisely based purely on the provided quantitative and qualitative data.

Rules:
* Max 3 sentences
* Sentence 1: what happened
* Sentence 2: why (sector + exposure + stocks)
* Sentence 3: uncertainty/conflict (if any)
* Use simple, sharp language
* Avoid long sentences
* No repetition
* No filler words

You must return STRICT JSON describing your findings with this exact schema:
{
  "summary": "The main narrative explanation",
  "drivers": ["list of strings detailing each driver"],
  "risks": ["list of strings detailing critical risks or conflicts"]
}
"""

    # Step 5: User Input
    user_prompt = f"DATA:\n{clean_payload_str}"

    # Step 6: Groq API Call
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        # Step 7: Parse Output
        output_text = response.choices[0].message.content
        
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError:
            logger.error("LLM returned malformed JSON.")
            result = {
                "summary": output_text,
                "drivers": top_drivers,
                "risks": risks
            }
            
        return result
        
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return {
            "summary": "Error communicating with AI reasoning engine.",
            "drivers": [],
            "risks": []
        }
