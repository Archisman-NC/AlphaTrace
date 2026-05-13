import sys
import os
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.reasoning.router import load_portfolio_context, get_portfolio_context_data
    
    portfolio_id = "PORTFOLIO_001"
    print(f"--- Testing Reasoning Context: {portfolio_id} ---")
    
    ctx = load_portfolio_context(portfolio_id)
    
    print(f"Status: {ctx.get('status')}")
    print(f"Portfolio: {ctx.get('name')} ({ctx.get('portfolio_id')})")
    
    # 1. Verify latest_return
    print(f"Latest Return: {ctx.get('latest_return'):.6f}")
    
    # 2. Verify weekly_return (compounded)
    print(f"Weekly Compounded Return: {ctx.get('weekly_return'):.6f}")
    
    # 3. Verify volatility (annualized)
    print(f"Annualized Volatility: {ctx.get('volatility'):.6f}")
    
    # 4. Verify context windowing
    history = ctx.get('returns_summary', {}).get('portfolio_return', {})
    print(f"History Window Size: {len(history)}")
    
    # 5. Verify serializability
    try:
        json_str = json.dumps(ctx)
        print("SUCCESS: Context is JSON serializable.")
    except TypeError as e:
        print(f"FAILURE: Context is NOT JSON serializable: {e}")
        
    # 6. Verify Legacy Shim
    print("\n--- Testing Legacy Shim ---")
    legacy_ctx = get_portfolio_context_data(portfolio_id)
    print(f"Legacy Keys: {list(legacy_ctx.keys())}")
    print(f"Exposure Type: {type(legacy_ctx.get('exposure'))}")
    print(f"Ranked Holdings Count: {len(legacy_ctx.get('ranked_holdings', []))}")
    
    if 'analytics' in legacy_ctx:
        print("SUCCESS: New analytics context present in legacy shim.")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
