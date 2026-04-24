import re

print("[DEBUG] llm_evaluator loaded successfully")

def evaluate_response(response: str, context: dict = None) -> dict:
    """
    Production-safe evaluator fallback: Heuristic scoring.
    Ensures high-fidelity advisory without circular LLM dependencies.
    """
    if not response or not isinstance(response, str):
        return {"score": 0.0, "details": {}, "feedback": "empty response"}

    score = 0

    # 1. Ticker presence (Institutional proof)
    if re.search(r'\b[A-Z]{3,}\b', response):
        score += 3

    # 2. Quantitative grounding (Data proof)
    if "%" in response:
        score += 3

    # 3. Causal reasoning check (Logic proof)
    if any(word in response.lower() for word in ["because", "due to", "driven by", "impacted by"]):
        score += 4

    # 4. Conciseness/Length bonus
    word_count = len(response.split())
    if word_count > 20:
        score += 1

    # Normalize to 0-10 scale
    final_score = min(float(score), 10.0)

    return {
        "score": final_score,
        "details": {
            "has_ticker": score >= 3,
            "is_quant": "%" in response,
            "is_causal": "because" in response.lower() or "due to" in response.lower()
        },
        "feedback": "Automated heuristic audit complete."
    }
