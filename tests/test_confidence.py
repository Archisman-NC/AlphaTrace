from app.evaluation.llm_evaluator import compute_confidence

def test_confidence_weak_signal():
    # compute_confidence(conflicts, sector_alignment_strength, portfolio_change, signal_strength, has_mixed_signals)
    confidence = compute_confidence(
        conflicts=[],
        sector_alignment_strength=0.0,  # strong_alignment=False
        portfolio_change=0.05,        # Weak portfolio change
        signal_strength="weak",       # signal_strength="weak"
        has_mixed_signals=False
    )

    # Base 0.8 - 0.15 (weak change) = 0.65
    assert confidence < 0.7

def test_confidence_high_conflict():
    confidence = compute_confidence(
        conflicts=[{"reason": "Something went wrong"}],
        sector_alignment_strength=0.8,
        portfolio_change=1.5,
        signal_strength="strong"
    )
    
    # Base 0.8 + 0.1 (strong align) - 0.1 (conflict) = 0.8
    assert confidence == 0.8
