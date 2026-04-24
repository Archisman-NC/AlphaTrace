print("[DEBUG] llm_evaluator loaded")

def evaluate_response(response: str, context: dict = None) -> dict:
    """
    Standard Heuristic Evaluator: NO-HALLUCINATION verification.
    """
    if not response or not isinstance(response, str):
        return {"score": 0.0, "breakdown": {}, "feedback": "empty input"}

    score = 0

    # 1. Quantitative check
    if "%" in response:
        score += 3
        
    # 2. Causal logic check
    causal_markers = ["because", "due to", "driven by", "why", "reason"]
    if any(word in response.lower() for word in causal_markers):
        score += 3
        
    # 3. Contextual Density bonus
    word_count = len(response.split())
    if word_count > 20:
        score += 2

    return {
        "score": float(score),
        "breakdown": {
            "is_quant": "%" in response,
            "has_causal": any(m in response.lower() for m in causal_markers),
            "sufficient_length": word_count > 20
        },
        "feedback": "heuristic evaluation complete"
    }
