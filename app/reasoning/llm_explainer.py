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
    change_val = portfolio_metrics.get("daily_change_percent", 0.0)
    payload = {
        "portfolio_change": f"{change_val:.2f}%",
        "top_drivers": top_drivers,
        "conflicts": conflicts,
        "risks": risks
    }
    
    # Ensure diversified logic renders natively inside LLM outputs without programmatic structural markers
    clean_payload_str = json.dumps(payload, indent=2).replace("DIVERSIFIED HOLDINGS", "broadly diversified holdings").replace("Diversified Holdings", "broadly diversified holdings")

    # Step 4: System Prompt (Strict JSON)
    system_prompt = """
You are a hedge-fund financial analyst writing a daily note.

Explain portfolio movement clearly and concisely based purely on the provided quantitative and qualitative data.

Rules for "summary":
* EXACTLY 3 sentences. No more, no less.
* Sentence 1: "Portfolio declined by X% amid sector divergence" OR "Portfolio rose by X% amid sector divergence". Use this EXACT phrasing if opposing trends exist, keeping it extremely short and sharp.
* Sentence 2: Main cause (sector + exposure + stock). Format exactly as: "[Sector] holdings contributed X%, primarily driven by [STOCK TICKER]". Always use the word "contributed" (never "drove impact").
* Sentence 3: Uncertainty / conflict. Emphasize concentration or diverging stock paths.
* Use simple, sharp, professional language avoiding robotic phrasing.
* NEVER use abbreviations. Always expand "IT" to "Information Technology".

CAUSAL ATTRIBUTION RULES (CRITICAL):
* DO NOT attach the same news to all sectors.
* Banking impacts -> macro (RBI, rates)
* Information Technology impacts -> earnings/global demand
* Energy impacts -> commodity/sector trends
* If no strong causal link exists for a sector in the data, DO NOT force one. Let the quantitative trend speak for itself.
* Risks must be phrased sharply like "X poses risk to Y". NEVER use weak phrasing like "may impact".

You must return STRICT JSON describing your findings with this exact schema without any markdown formatting or trailing text:
{
  "summary": "String matching the 3-sentence rules above",
  "drivers": ["list of strings detailing each driver using the exact format 'Sector holdings contributed X%, primarily driven by [STOCK TICKERS (e.g. HDFCBANK)]'"],
  "risks": ["list of strings detailing critical risks or conflicts using the 'X poses risk to Y' format"]
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
