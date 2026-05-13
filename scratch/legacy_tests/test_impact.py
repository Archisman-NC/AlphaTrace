from app.analytics.sector_impact import compute_sector_impact

def test_impact_calculation():
    # Input format for compute_sector_impact is a dict of {sector: {portfolio_weight, sector_change}}
    sector_link = {
        "BANKING": {
            "portfolio_weight": 0.30,
            "sector_change": -2.0,
            "sentiment": "negative"
        }
    }

    result = compute_sector_impact(sector_link)

    # weight * change = 0.3 * -2.0 = -0.6
    assert result["BANKING"]["impact"] == -0.6
    assert result["BANKING"]["portfolio_weight"] == 0.30

def test_impact_rounding():
    sector_link = {
        "IT": {
            "portfolio_weight": 0.155,
            "sector_change": 1.111,
            "sentiment": "positive"
        }
    }
    
    result = compute_sector_impact(sector_link)
    
    # 0.155 * 1.111 = 0.172205 -> rounded to 0.17
    assert result["IT"]["impact"] == 0.17
