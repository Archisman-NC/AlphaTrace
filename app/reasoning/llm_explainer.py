import os
import json
import logging
import time
from typing import Dict, List, Any
from groq import Groq
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

def generate_llm_explanation(
    portfolio_metrics: Dict[str, Any],
    top_drivers: List[dict],
    conflicts: List[dict],
    risks: List[dict],
    portfolio_id: str = "UNKNOWN"
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

    # Build Structured Input
    change_val = float(portfolio_metrics.get("daily_change_percent", 0.0))
    direction = "declined" if change_val < 0 else "rose"
    abs_change = abs(change_val)
    payload = {
        "portfolio_change": f"{direction} by {abs_change:.2f}%",
        "top_drivers": top_drivers,
        "conflicts": conflicts,
        "risks": risks
    }
    
    # Ensure diversified logic renders natively inside LLM outputs without programmatic structural markers
    clean_payload_str = json.dumps(payload, indent=2).replace("DIVERSIFIED HOLDINGS", "broadly diversified holdings").replace("Diversified Holdings", "broadly diversified holdings")

    # System Prompt (Strict JSON)
    system_prompt = """
You are a hedge-fund financial analyst writing a daily note.

Explain portfolio movement clearly and concisely based purely on the provided quantitative and qualitative data.

Rules for "summary":
* EXACTLY 3 sentences. No more, no less.
* Sentence 1: Use portfolio_change from the data verbatim (e.g. "Portfolio declined by 0.44% amid sector divergence."). Keep it extremely short.
* Sentence 2: Main cause (sector + stock + ONE causal trigger). Format: "[Sector] holdings contributed X%, primarily driven by [TICKER], as [short causal clause]." Always use "contributed" (never "drove impact").
* Sentence 3: ONLY express uncertainty or conflict. Start with "Uncertainty remains..." or similar. Emphasize concentration risk or diverging stock paths. Do NOT repeat the cause from Sentence 2.
* Use simple, sharp, professional language.
* NEVER use abbreviations. Always write "Information Technology" (not "IT").
* NEVER use apostrophes (write "RBI hawkish stance" not "RBI's hawkish stance").
* Ensure subject-verb agreement ("paths pose risk" not "paths poses risk").

CAUSAL ATTRIBUTION RULES (CRITICAL):
* Each sector MUST have a UNIQUE causal phrase. NEVER repeat the same cause across two drivers.
* Banking → "as hawkish RBI stance pressured lending outlook"
* Financial Services → "as tight liquidity conditions weighed on NBFCs"
* Information Technology → "as positive earnings momentum supported the sector"
* Energy → "as weakness in energy prices pressured the sector"
* DO NOT reference specific event names (e.g. "US Tech Giants report Q1 earnings"). Use generic causal language only.
* If no strong causal link exists, let the quantitative trend speak for itself.
* Risks must use sharp phrasing: "X poses risk to Y". NEVER use "may impact".

You must return STRICT JSON with this exact schema. No markdown, no trailing text:
{
  "summary": "3-sentence string matching the rules above",
  "drivers": ["Sector holdings contributed X%, primarily driven by TICKER, as [unique causal clause]"],
  "risks": ["X poses risk to Y"]
}
"""

    user_prompt = f"DATA:\n{clean_payload_str}"

    try:
        trace = None
        generation = None
        try:
            if hasattr(langfuse, "trace"):
                trace = langfuse.trace(
                    name="llm_explanation",
                    metadata={"portfolio_id": portfolio_id, "stage": "explanation"}
                )
            elif hasattr(langfuse, "start_as_current_generation"):
                generation = langfuse.start_as_current_generation(
                    name="llm_explanation",
                    input={"system": system_prompt, "user": user_prompt},
                    model="llama-3.3-70b-versatile",
                    metadata={"portfolio_id": portfolio_id, "stage": "explanation"}
                )
        except Exception as e:
            print(f"Langfuse init error: {e}")

        start_time = time.time()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time
        output_text = response.choices[0].message.content

        try:
            if trace:
                trace.generation(
                    name="explanation_call",
                    input={"system": system_prompt, "user": user_prompt},
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
                print(f"[LANGFUSE] Explanation Trace URL: {trace.get_trace_url()}")
            elif generation:
                generation.update(
                    output=output_text,
                    usage_details={
                        "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                        "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                        "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
                    }
                )
                generation.end()
                langfuse.flush()
        except Exception as e:
            print("Langfuse update error:", e)

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
        error_str = str(e)
        if "json_validate_failed" in error_str:
            try:
                # Retrying should also be traced if we want to be thorough, 
                # but following "minimal" rule I might skip or just trace the first one.
                # Let's keep it simple.
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                return result
            except Exception:
                pass
        
        logger.error(f"Groq API call failed: {e}")
        return {
            "summary": "Error communicating with AI reasoning engine.",
            "drivers": [],
            "risks": []
        }
