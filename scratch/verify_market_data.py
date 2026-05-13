import sys
import os
import pandas as pd

# Add the project root to sys.path to import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.data.market_data import fetch_ohlcv, get_latest_prices, NIFTY50_TICKERS
    
    print("Testing fetch_ohlcv for RELIANCE.NS...")
    df = fetch_ohlcv("RELIANCE.NS", period="1mo")
    if not df.empty:
        print(f"Success! Fetched {len(df)} rows.")
        print("Columns:", df.columns.tolist())
        print("First 2 rows:\n", df.head(2))
    else:
        print("Failed to fetch data for RELIANCE.NS")

    print("\nTesting fetch_ohlcv for ['RELIANCE.NS'] (list input)...")
    df_list = fetch_ohlcv(["RELIANCE.NS"], period="1mo")
    if not df_list.empty:
        print(f"Success! Fetched {len(df_list)} rows.")
        print("Columns Index Type:", type(df_list.columns))
        print("Columns:", df_list.columns.tolist())
    else:
        print("Failed to fetch data for ['RELIANCE.NS']")

    print("\nTesting get_latest_prices for a few tickers...")
    test_tickers = NIFTY50_TICKERS[:3]
    prices = get_latest_prices(test_tickers)
    print(f"Prices for {test_tickers}: {prices}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An error occurred during testing: {e}")
