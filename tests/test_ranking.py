from app.analytics.holding_ranker import rank_holdings

def test_top_holdings_sorted():
    holdings = {
        "A": {"weight": 0.2, "sector": "ENERGY", "day_change": 1.0},
        "B": {"weight": 0.5, "sector": "BANKING", "day_change": -1.0},
        "C": {"weight": 0.3, "sector": "IT", "day_change": 0.5}
    }

    ranked = rank_holdings(holdings)

    # Weights: B(0.5) > C(0.3) > A(0.2)
    assert ranked[0]["symbol"] == "B"
    assert ranked[1]["symbol"] == "C"
    assert ranked[2]["symbol"] == "A"

def test_rank_holdings_limit():
    holdings = {
        "A": {"weight": 0.2},
        "B": {"weight": 0.5},
        "C": {"weight": 0.3}
    }
    
    # top_n=2
    ranked = rank_holdings(holdings, top_n=2)
    
    assert len(ranked) == 2
    assert ranked[0]["symbol"] == "B"
