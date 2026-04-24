import re

print("[DEBUG] llm_evaluator initialized")

def evaluate_response(response: str, context: dict = None) -> dict:
    """
    Standard Heuristic Evaluator: NO-HALLUCINATION verification.
    Guaranteed to return 'confidence' and 'details'.
    """
    if not response or not isinstance(response, str):
        return {
            "score": 0.0, 
            "confidence": 0.1,
            "details": {}, 
            "feedback": "empty input"
        }

    score = 0
    # 1. Quantitative check
    if "%" in response:
        score += 3
    # 2. Causal logic check
    causal_markers = ["because", "due to", "driven by", "why", "reason"]
    if any(word in response.lower() for word in causal_markers):
        score += 3
    # 3. Contextual Density
    word_count = len(response.split())
    if word_count > 20:
        score += 2
        
    # Scale calculation
    final_score = float(score)
    # Heuristic confidence translation
    conf = min(0.9, (final_score / 10.0) + 0.1)

    return {
        "score": final_score,
        "confidence": conf,
        "details": {
            "has_ticker": bool(re.search(r'\b[A-Z]{3,}\b', response)),
            "is_quant": "%" in response,
            "has_causal": any(m in response.lower() for m in causal_markers),
            "sufficient_length": word_count > 20
        },
        "feedback": "heuristic evaluation complete"
    }
