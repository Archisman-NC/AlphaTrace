from app.reasoning.causal_chain_builder import build_causal_chains

def test_filters_unexposed_sectors():
    news = [
        {
            "news": "Tech giants report earnings",
            "sector": "INFORMATION TECHNOLOGY",
            "news_sentiment": "positive",
            "portfolio_exposed": False,
            "portfolio_weight": 0.0
        }
    ]
    sector_impact = {
        "INFORMATION TECHNOLOGY": {"impact": 0.5, "portfolio_weight": 0.0, "sector_change": 2.0}
    }
    stock_drilldown = {
        "INFORMATION TECHNOLOGY": [{"symbol": "TCS"}]
    }

    result = build_causal_chains(news, sector_impact, stock_drilldown)

    # Should be empty because portfolio_exposed is False and portfolio_weight is 0.0
    assert result == []

def test_includes_exposed_sectors():
    news = [
        {
            "news": "RBI interest rate decision",
            "sector": "BANKING",
            "news_sentiment": "negative",
            "portfolio_exposed": True,
            "portfolio_weight": 0.3
        }
    ]
    sector_impact = {
        "BANKING": {"impact": -0.6, "portfolio_weight": 0.3, "sector_change": -2.0}
    }
    stock_drilldown = {
        "BANKING": [{"symbol": "HDFCBANK"}]
    }

    result = build_causal_chains(news, sector_impact, stock_drilldown)

    assert len(result) == 1
    assert result[0]["sector"] == "BANKING"
    assert result[0]["stocks"] == ["HDFCBANK"]
