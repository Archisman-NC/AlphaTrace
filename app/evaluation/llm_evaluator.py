try:
    print("[DEBUG] START: app/evaluation/llm_evaluator.py initialization")
    import re

    def evaluate_response(response: str, context: dict = None) -> dict:
        """
        Standard Heuristic Evaluator.
        """
        if not response or not isinstance(response, str):
            return {"score": 0.0, "details": {}, "feedback": "empty"}

        score = 0
        if "%" in response: score += 3
        if any(w in response.lower() for w in ["because", "due to", "driven by"]): score += 3
        if len(response.split()) > 20: score += 2
        
        final_score = float(score)
        conf = min(0.9, (final_score / 10.0) + 0.1)

        return {
            "score": final_score,
            "confidence": conf,
            "details": {
                "is_quant": "%" in response,
                "has_causal": "because" in response.lower()
            },
            "feedback": "heuristic eval"
        }

    print("[DEBUG] MID: Function evaluate_response defined")
    
    # Simulate a potential point of failure (e.g. malformed regex or logic)
    marker_test = re.compile(r'\b[A-Z]{3,}\b')
    
    print("[DEBUG] END: app/evaluation/llm_evaluator.py fully initialized")

except Exception as e:
    print(f"[CRASH] evaluator import failed: {e}")
    # Optionally: print(traceback.format_exc())
