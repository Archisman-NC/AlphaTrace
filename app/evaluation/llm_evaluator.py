import os
import json
import logging
import time
import re
from typing import Dict, List, Any
from groq import Groq
from app.utils.helpers import langfuse

logger = logging.getLogger(__name__)

# --- High-Discrimination Rubric ---
STRICT_JUDGE_PROMPT = """
You are a senior financial analyst. Score AlphaTrace summaries (0-10) using this rubric:
1. TICKER SPECIFICITY (0-3): +3 for HDFCBANK, TCS, RELIANCE etc. +1 for sectors.
2. QUANTIFICATION (0-3): +3 for %, +1 for vague numbers.
3. CAUSAL TRIGGER (0-3): +3 for matching news trigger, +1 for generic.
4. CLARITY (0-1): +1 for logical structure.

BE HARSH. Cluster 3-5 for weak, 8-9 for precise.
Return JSON: {"score": float, "reason": "...", "breakdown": {"ticker":int, "quant":int, "trigger":int, "clarity":int}}
"""

VALID_TICKERS = {
    "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "LT", "WIPRO", "HCLTECH", "AXISBANK", "ADANIENT",
    "BHARTIARTL", "KOTAKBANK", "ITC", "HINDUNILVR"
}

def detect_ticker(summary: str) -> int:
    """Eliminates noise by validating uppercase tokens against a whitelist."""
    tokens = re.findall(r"\b[A-Z]{2,10}\b", summary)
    matches = [t for t in tokens if t in VALID_TICKERS]
    if len(matches) >= 1:
        return 3
    elif "sector" in summary.lower():
        return 1
    return 0

def detect_trigger(summary: str, trigger: str) -> int:
    """Ensures news-derived causal trigger is explicitly mentioned."""
    if not trigger: return 0
    if trigger.lower() in summary.lower():
        return 3
    elif any(word in summary.lower() for word in trigger.split() if len(word) > 3):
        return 1
    return 0

def rule_check(summary: str, trigger: str = "") -> dict:
    """Deterministic validation logic."""
    details = {
        "ticker": detect_ticker(summary),
        "quant": 3 if "%" in summary else (1 if any(c.isdigit() for c in summary) else 0),
        "trigger": detect_trigger(summary, trigger),
        "clarity": 1 if len(summary.split()) > 15 else 0
    }
    return {"score": sum(details.values()), "details": details}

def evaluate_explanation(summary: str, original_input: dict, portfolio_id: str = "N/A") -> dict:
    """LLM Judge with strict rubric."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        user_prompt = f"SUMMARY: {summary}\nDATA: {json.dumps(original_input)}"
        
        trace = None
        if hasattr(langfuse, "trace"):
            trace = langfuse.trace(name="llm_evaluation", metadata={"portfolio_id": portfolio_id})

        start_time = time.time()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": STRICT_JUDGE_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time
        result = json.loads(response.choices[0].message.content)
        
        if trace:
            trace.generation(name="audit_call", output=json.dumps(result), metadata={"latency": latency})
            langfuse.flush()
        return result
    except Exception as e:
        logger.error(f"Eval failed: {e}")
        return {"score": 0.0, "reason": "Judge error"}
