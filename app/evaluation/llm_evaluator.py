import os
import json
import logging
import time
from typing import Dict, List, Any
from groq import Groq
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

def evaluate_explanation(
    explanation: Dict[str, Any],
    original_input: Dict[str, Any],
    portfolio_id: str = "UNKNOWN"
) -> dict:
    """
    Acts as an LLM-as-a-judge to grade the quality of the generated natural language reasoning.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq client for evaluation: {e}")
        return {
            "score": 7.0,
            "feedback": "Internal Error: Unable to communicate with AI Judge. Default score."
        }

    system_prompt = """
You are a strict quantitative evaluator of financial explanations.
Evaluate the explanation based on:
1. Causality clarity (Does it clearly connect news -> sector -> portfolio?)
2. Accuracy (Does it strictly match the provided data?)
3. Conciseness (Is it short and tightly focused?)

Rules:
* Be extremely strict.
* Do NOT hallucinate.
* Do NOT rewrite the explanation.

You must return STRICT JSON describing your evaluation with this exact schema:
{
  "score": 8.5,
  "feedback": "Your concise one-sentence feedback here"
}
"""

    user_prompt = f"EXPLANATION:\n{json.dumps(explanation, indent=2)}\n\nINPUT DATA:\n{json.dumps(original_input, indent=2)}"

    try:
        # Langfuse Tracing (Universal Compatibility v2/v4)
        trace = None
        generation = None
        try:
            if hasattr(langfuse, "trace"):
                trace = langfuse.trace(
                    name="llm_evaluation",
                    metadata={"portfolio_id": portfolio_id, "stage": "evaluation"}
                )
            elif hasattr(langfuse, "start_as_current_generation"):
                # For v4, we'll use a single generation object as the trace to keep it clean
                generation = langfuse.start_as_current_generation(
                    name="llm_evaluation",
                    input={"system": system_prompt, "user": user_prompt},
                    model="llama-3.3-70b-versatile",
                    metadata={"portfolio_id": portfolio_id, "stage": "evaluation"}
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
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time
        output_text = response.choices[0].message.content
        
        # Update trace/generation
        try:
            if trace:
                trace.generation(
                    name="evaluation_call",
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
                print(f"[LANGFUSE] Evaluation Trace URL: {trace.get_trace_url()}")
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
            logger.error("Evaluator LLM returned malformed JSON.")
            result = {
                "score": 7.0,
                "feedback": "Parsing failed, default score assigned"
            }
            
        return result
        
    except Exception as e:
        logger.error(f"Groq Evaluator call failed: {e}")
        return {
            "score": 7.0,
            "feedback": "API execution failed, default score assigned"
        }

def compute_confidence(
    conflicts: List[dict],
    sector_alignment_strength: float,
    portfolio_change: float,
    signal_strength: str = "moderate",
    has_mixed_signals: bool = False
) -> float:
    """
    Deterministically computes a confidence score for the generated reasoning 
    using entirely quantitative pipeline metrics.
    """
    base = 0.8
    
    if abs(portfolio_change) < 0.1:
        base -= 0.15
        
    if conflicts and len(conflicts) > 0:
        base -= 0.1
        
    if has_mixed_signals:
        base -= 0.05
        
    if sector_alignment_strength > 0.5:
        base += 0.1
        
    confidence = max(0.0, min(1.0, base))
    
    if signal_strength.lower() == "moderate":
        confidence = min(confidence, 0.85)
        
    return confidence

def build_final_output(
    explanation: Dict[str, Any],
    evaluation: Dict[str, Any],
    confidence: float,
    signal_strength: str = "moderate"
) -> dict:
    """
    Builds the ultimate final output object bridging human-readable text 
    and machine-deterministic confidence scoring.
    """
    return {
        "summary": explanation.get("summary", ""),
        "drivers": explanation.get("drivers", []),
        "risks": explanation.get("risks", []),
        "confidence": round(confidence, 2),
        "evaluation_score": evaluation.get("score", 0),
        "signal_strength": signal_strength
    }

# --- Deterministic Rule-Check Layer ---

CAUSAL_KEYWORDS = [
    "due to", "because", "as a result", "driven by",
    "following", "amid", "led by", "pressure", "sentiment", "impact"
]

def rule_check(summary: str, top_drivers: list = None) -> dict:
    """
    Deterministic validation of explanation completeness.
    Checks whether the summary mentions the relevant sector, stock, and a causal trigger.
    Relying ONLY on dynamic runtime data from top_drivers.
    """
    if not summary:
        return {"mentions_sector": False, "mentions_stock": False, "mentions_cause": False}

    print("[RULE CHECK INPUT]", top_drivers)
    text = summary.lower()
    
    mentions_sector = False
    mentions_stock = False

    if top_drivers:
        for driver in top_drivers:
            # Dynamic Sector detection
            sector = driver.get("sector", "").lower()
            if sector and sector in text:
                mentions_sector = True
            
            # Dynamic Stock detection
            for stock in driver.get("stocks", []):
                if stock.lower() in text:
                    mentions_stock = True

    # Causal keyword detection
    mentions_cause = any(k in text for k in CAUSAL_KEYWORDS)
    
    return {
        "mentions_sector": mentions_sector,
        "mentions_stock": mentions_stock,
        "mentions_cause": mentions_cause
    }

def compute_rule_score(checks: dict) -> float:
    """
    Converts boolean rule checks into a 0-1 score.
    Sector = 0.3, Stock = 0.3, Cause = 0.4.
    """
    score = 0.0
    if checks.get("mentions_sector"):
        score += 0.3
    if checks.get("mentions_stock"):
        score += 0.3
    if checks.get("mentions_cause"):
        score += 0.4
    return score
