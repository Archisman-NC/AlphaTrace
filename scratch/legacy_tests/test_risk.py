from app.analytics.risk_detection import detect_concentration_risk

def test_sector_concentration_high_risk():
    exposure = {
        "BANKING": 0.55,  # > 40%
        "IT": 0.10
    }

    risks = detect_concentration_risk(exposure)

    # Should detect critical risk for BANKING
    assert any(r["severity"] == "critical" and r["sector"] == "BANKING" for r in risks)

def test_sector_concentration_medium_risk():
    exposure = {
        "BANKING": 0.38,
        "FINANCIAL SERVICES": 0.35,
        "ENERGY": 0.27
    }
    
    risks = detect_concentration_risk(exposure)
    
    # Combined: 0.38 + 0.35 = 0.73 (> 0.70)
    # No single > 40%, but top 2 > 70%
    assert not any(r["severity"] == "critical" for r in risks)
    assert any(r["severity"] == "medium" for r in risks)
    assert sorted(risks[0]["sectors"]) == sorted(["BANKING", "FINANCIAL SERVICES"])
