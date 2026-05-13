from app.analytics.sector_impact import compute_sector_impact
from app.reasoning.causal_chain_builder import build_causal_chains
from app.reasoning.top_drivers import select_top_drivers

def test_end_to_end_pipeline_small_case():
    # Setup inputs
    news = [{
        "news": "RBI hawkish stance",
        "sector": "BANKING",
        "news_sentiment": "negative",
        "portfolio_exposed": True,
        "portfolio_weight": 0.5,
        "sector_change": -2.0
    }]

    # In our actual pipeline, compute_sector_impact happens before chains are built 
    # to provide impact numbers for the chains.
    sector_link = {
        "BANKING": {
            "portfolio_weight": 0.5,
            "sector_change": -2.0,
            "sentiment": "negative"
        }
    }

    # Step 1: Compute impact
    sector_impacts = compute_sector_impact(sector_link)
    
    # Step 2: Build causal chain
    stock_drilldown = {"BANKING": [{"symbol": "HDFCBANK"}]}
    chains = build_causal_chains(news, sector_impacts, stock_drilldown)

    # Note: build_causal_chains returns impact from sector_impacts
    # Step 3: Select top drivers
    # We need to ensure the chain has a 'impact' key for select_top_drivers
    # In our real main.py, we copy it.
    for c in chains:
        c["impact"] = c.get("sector_impact", 0.0)

    top = select_top_drivers(chains)

    # Validations
    assert len(top) > 0
    assert top[0]["sector"] == "Banking" # title case checked
    assert top[0]["impact"] == -1.0 # 0.5 * -2.0
    assert top[0]["stocks"] == ["HDFCBANK"]
