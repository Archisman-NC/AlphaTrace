import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.portfolio_builder import get_portfolio_returns, get_portfolio_metadata, PORTFOLIOS
    
    portfolio_id = "PORTFOLIO_001"
    print(f"--- Testing Portfolio: {portfolio_id} ---")
    
    # Test Metadata
    metadata = get_portfolio_metadata(portfolio_id)
    print(f"Metadata: {metadata['name']}, Tickers: {metadata['holdings_count']}")
    
    # Test Returns
    print("\nComputing portfolio returns...")
    df = get_portfolio_returns(portfolio_id, period="1mo")
    
    if not df.empty:
        print(f"Success! Result shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nDtypes:\n", df.dtypes)
        print("\nIndex Type:", type(df.index))
        
        # Check for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print("FAILURE: MultiIndex columns detected!")
        else:
            print("SUCCESS: Single-level columns verified.")
            
        # Check for Object dtypes
        if (df.dtypes == 'object').any():
            print("FAILURE: Object dtypes detected!")
        else:
            print("SUCCESS: Numeric-only dtypes verified.")
            
        # Check for final column
        if df.columns[-1] == 'portfolio_return':
            print("SUCCESS: Final column is 'portfolio_return'.")
        else:
            print(f"FAILURE: Final column is {df.columns[-1]}")
            
        print("\nFirst 2 rows of weighted returns and portfolio_return:\n", df.head(2))
        
        # Verify weight aggregation (sum of components should equal portfolio_return)
        # Note: we use sum(axis=1) in the code including ticker columns
        # To verify: sum of ticker columns == portfolio_return
        ticker_cols = [c for c in df.columns if c != 'portfolio_return']
        sum_check = df[ticker_cols].sum(axis=1)
        diff = (sum_check - df['portfolio_return']).abs().max()
        print(f"\nAggregation check (max diff): {diff}")
        if diff < 1e-10:
            print("SUCCESS: Row-wise aggregation is accurate.")
        else:
            print("FAILURE: Aggregation mismatch!")
            
    else:
        print("Failed to compute portfolio returns (DataFrame empty).")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
