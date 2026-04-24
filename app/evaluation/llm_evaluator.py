import os
import json
import logging
import time
import re
from typing import Dict, List, Any
from groq import Groq
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

# --- High-Discrimination Rubric Prompt ---
STRICT_JUDGE_PROMPT = """
You are a senior financial analyst and a strict evaluator of AI-generated reasoning.
Your job is to score AlphaTrace summaries from 0–10 based on a strict rubric.

## EVALUATION RUBRIC (0–10):
1. TICKER SPECIFICITY (0–3 points)
   - 3: Mentions specific stock tickers (e.g., HDFCBANK, TCS, AAPL).
   - 1: Mentions sectors only (e.g., IT, Banking).
   - 0: Mentions neither.

2. QUANTIFICATION (0–3 points)
   - 3: Includes precise percentages or numbers (e.g., 2.3%, -0.8%).
   - 1: Vague numeric terms (e.g., "significant drop", "major move").
   - 0: No quantification.

3. CAUSAL TRIGGER (0–3 points)
   - 3: Includes a specific news-derived trigger (e.g., "hawkish RBI rate outlook").
   - 1: Generic cause (e.g., "market conditions", "volatility").
   - 0: No cause.

4. CLARITY BONUS (0–1 point)
   - 1: Logically structured and jargon-appropriate.

## SCORING RULES:
- BE HARSH. Avoid scores above 9.0 unless exceptional.
- CLUSTER Scores: 3-5 for generic summaries, 8-9 for precise, quantified reasoning.
- Do NOT rewrite. Just Grade.

## FEW-SHOT EXAMPLES:
### GOOD (9-10)
Ex: "Your portfolio fell 1.2% primarily due to HDFCBANK reacting to hawkish RBI rate outlook."
Score: 10 (Ticker: 3, Quant: 3, Trigger: 3, Clarity: 1)

### MODERATE (6-7)
Ex: "The IT sector declined 0.8% due to weak demand impacting your TCS holding."
Score: 7 (Ticker: 3, Quant: 3, Trigger: 0, Clarity: 1)

### POOR (2-4)
Ex: "Your portfolio moved due to market conditions amid uncertainty."
Score: 2 (Ticker: 0, Quant: 0, Trigger: 1, Clarity: 1)

## OUTPUT FORMAT (STRICT JSON):
{
  "score": float,
  "reason": "concise explanation",
  "breakdown": {
    "ticker": int,
    "quant": int,
    "trigger": int,
    "clarity": int
  }
}
"""

def evaluate_explanation(
    explanation: str,
    original_input: Dict[str, Any],
    portfolio_id: str = "UNKNOWN"
) -> dict:
    """
    LLM-as-a-judge specifically tuned to audit causal reasoning quality.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Failed to initialize Groq client for evaluation: {e}")
        return {"score": 0.0, "feedback": "Judge offline"}

    user_prompt = f"SUMMARY TO EVALUATE:\n{explanation}\n\nDATA CONTEXT:\n{json.dumps(original_input, indent=2)}"

    try:
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="llm_evaluation",
                metadata={"portfolio_id": portfolio_id, "stage": "evaluation"}
            )

        start_time = time.time()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": STRICT_JUDGE_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time
        output_text = response.choices[0].message.content
        
        if trace:
            trace.generation(
                name="evaluation_call",
                input={"system": STRICT_JUDGE_PROMPT, "user": user_prompt},
                output=output_text,
                model="llama-3.3-70b-versatile",
                metadata={"latency": latency}
            )
            langfuse.flush()

        return json.loads(output_text)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"score": 0.0, "reason": "Evaluation engine timeout"}

def rule_check(summary: str) -> dict:
    """
    Deterministic validation of reasoning quality based on patterns.
    """
    details = {"ticker": 0, "quant": 0, "trigger": 0, "clarity": 0}
    if not summary:
        return {"score": 0, "details": details}

    # 1. Ticker Check: Regex for uppercase symbols 2-10 chars
    tickers = re.findall(r'\b[A-Z]{2,10}\b', summary)
    if tickers:
        details["ticker"] = 3
    elif any(s in summary.lower() for s in ["sector", "it", "bank", "tech", "growth", "energy"]):
        details["ticker"] = 1

    # 2. Quantification Check: Presence of %
    if "%" in summary:
        details["quant"] = 3
    elif any(n.isdigit() for n in summary.split()):
        details["quant"] = 1

    # 3. Causal Trigger Check
    CAUSAL_TRIGGERS = ["due to", "because", "driven by", "reaction", "result", "owing to", "led by"]
    if any(k in summary.lower() for k in CAUSAL_TRIGGERS):
        details["trigger"] = 3 # Simple heuristic for presence of causal structure
    
    # 4. Clarity Bonus
    if len(summary.split()) > 15: # Length heuristic for completeness
        details["clarity"] = 1

    score = sum(details.values())
    return {"score": score, "details": details}

def compute_confidence(
    conflicts: List[dict],
    sector_alignment_strength: float,
    portfolio_change: float
) -> float:
    """
    Quantitative confidence engine.
    """
    base = 0.8
    if abs(portfolio_change) < 0.1: base -= 0.15
    if conflicts: base -= 0.1
    if sector_alignment_strength > 0.5: base += 0.1
    return max(0.0, min(1.0, base))
