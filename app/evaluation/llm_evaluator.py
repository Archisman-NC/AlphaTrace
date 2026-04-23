import os
import json
import logging
from typing import Dict, List, Any
from groq import Groq

logger = logging.getLogger(__name__)

def evaluate_explanation(
    explanation: Dict[str, Any],
    original_input: Dict[str, Any]
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        output_text = response.choices[0].message.content
        
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
    signal_strength: float
) -> float:
    """
    Deterministically computes a confidence score for the generated reasoning 
    using entirely quantitative pipeline metrics, averting LLM hallucination.
    """
    base = 0.8
    
    # Detract for conflicting signals
    if conflicts and len(conflicts) > 0:
        base -= 0.1
        
    # Boost for strong alignment
    if sector_alignment_strength > 0.5:
        base += 0.1
        
    # Detract for weak overall signal
    if signal_strength < 0.2:
        base -= 0.1
        
    # Clamp value
    confidence = max(0.0, min(1.0, base))
    
    return confidence

def build_final_output(
    explanation: Dict[str, Any],
    evaluation: Dict[str, Any],
    confidence: float
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
        "evaluation_score": evaluation.get("score", 0)
    }
