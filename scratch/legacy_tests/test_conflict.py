from app.reasoning.conflict_detector import detect_conflicts

def test_positive_news_negative_sector():
    # Setup inputs for detect_conflicts
    causal_chains = [
        {
            "sector": "BANKING",
            "news_sentiment": "positive",
            "stocks": [] # Sector-level only
        }
    ]
    normalized_holdings = {}
    sector_trends = {
        "BANKING": {"change": -1.5} # Sector went DOWN despite positive news
    }

    result = detect_conflicts(causal_chains, normalized_holdings, sector_trends)

    assert len(result) == 1
    assert result[0]["conflict"] is True
    assert "Positive sector news but BANKING declined" in result[0]["reason"]

def test_stock_specific_divergence_positive_sector():
    causal_chains = [
        {
            "sector": "IT",
            "news_sentiment": "positive",
            "stocks": ["TCS"]
        }
    ]
    normalized_holdings = {
        "TCS": {"day_change": -2.0, "sector": "IT"} # Stock DOWN
    }
    sector_trends = {
        "IT": {"change": 3.0} # Sector UP
    }

    result = detect_conflicts(causal_chains, normalized_holdings, sector_trends)

    assert len(result) == 1
    assert result[0]["stock"] == "TCS"
    assert "positive news but TCS declined" in result[0]["reason"]
