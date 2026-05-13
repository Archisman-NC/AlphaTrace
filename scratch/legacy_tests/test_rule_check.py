from app.evaluation.llm_evaluator import rule_check

def test_rule_check_dynamic_detection():
    summary = "Banking sector declined due to RBI policy impacting HDFCBANK"
    
    # Note: rule_check in our app uses "mentions_sector", "mentions_stock", "mentions_cause"
    drivers = [
        {"sector": "BANKING", "stocks": ["HDFCBANK"]}
    ]

    result = rule_check(summary, drivers)

    assert result["mentions_sector"] is True
    assert result["mentions_stock"] is True
    assert result["mentions_cause"] is True

def test_rule_check_mismatch():
    summary = "Energy sector grew following oil price surge"
    
    drivers = [
        {"sector": "BANKING", "stocks": ["HDFCBANK"]}
    ]

    result = rule_check(summary, drivers)

    # Sector Mismatch
    assert result["mentions_sector"] is False
    assert result["mentions_stock"] is False
    # But cause "following" is present
    assert result["mentions_cause"] is True
