import sys
import os
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.reasoning.router import load_portfolio_context
    from app.data.portfolio_builder import PORTFOLIOS

    print("--- Testing AI Signal Reasoning Integration ---")

    # 1. Load Real Portfolio Context (will trigger signal scanning)
    portfolio_id = "PORTFOLIO_001"
    print(f"Loading context for {portfolio_id}...")
    
    # We use a short period for faster verification
    context = load_portfolio_context(portfolio_id, period="6mo")
    
    if context.get("status") == "success":
        print("\nSUCCESS: Portfolio context loaded successfully.")
        
        # 2. Verify Signal Intelligence Structure
        sig_intel = context.get("signal_intelligence", {})
        print("\nSignal Intelligence Context:")
        print(f"  Market Bias: {sig_intel.get('market_bias')}")
        print(f"  Summary:     {sig_intel.get('summary')}")
        
        if sig_intel.get("top_opportunity"):
            top = sig_intel["top_opportunity"]
            print(f"  Top Opportunity: {top['ticker']} ({top['direction']})")
            print(f"  Signal Strength: {top['signal_strength']}")
            print(f"  Causal Reason:   {top['causal_reason'][:60]}...")
        else:
            print("  Top Opportunity: None")

        # 3. Verify Sector Clusters
        clusters = sig_intel.get("sector_clusters", [])
        print(f"\nSector Clusters Detected: {len(clusters)}")
        for c in clusters:
            print(f"  - {c}")

        # 4. Verify Context Fusion (Market Summary)
        summary = context.get("market_summary", "")
        print("\nFused Market Summary (AI-Ready):")
        print(f"  {summary}")

        # 5. JSON Serializability
        try:
            json_str = json.dumps(context)
            print("\nSUCCESS: Full reasoning context is JSON serializable.")
        except TypeError as e:
            print(f"\nFAILURE: JSON serialization failed: {e}")

        # 6. Verify Deterministic Fields
        expected_fields = ["market_bias", "top_opportunity", "sector_clusters", "summary"]
        missing = [f for f in expected_fields if f not in sig_intel]
        if not missing:
            print("SUCCESS: Signal intelligence contains all deterministic fields.")
        else:
            print(f"FAILURE: Missing fields: {missing}")

    else:
        print(f"FAILURE: Could not load context. Status: {context.get('status')}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
