from app.reasoning.causal_chain_builder import build_causal_chains
from app.reasoning.conflict_detector import detect_conflicts

def test_empty_news_causal_chain():
    # Empty enriched_news should return empty list
    result = build_causal_chains([], {}, {})
    assert result == []

def test_empty_inputs_conflict_detector():
    # Empty causal_chains should return empty list
    result = detect_conflicts([], {}, {})
    assert result == []

def test_none_inputs_safety():
    # Defensive programming check: system should handle None gracefully if possible
    # (Though type hints suggest dict/list, good to check robustness)
    try:
        build_causal_chains(None, None, None)
    except Exception as e:
        # If it fails, that's okay as long as we know. 
        # But our implementation has "if not enriched_news or not sector_impact: return []"
        # which handles None because "if not None" is True.
        assert False, f"build_causal_chains raised an exception on None: {e}"

    result = build_causal_chains(None, None, None)
    assert result == []
