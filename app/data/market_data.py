import yfinance as yf
import pandas as pd
import streamlit as st
import logging
import time
from typing import Dict, List, Optional

# Initialize logger
logger = logging.getLogger(__name__)

NIFTY50_TICKERS = [
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ITC.NS",
    "SBIN.NS"
]

# Schema Contract
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

SECTOR_MAP = {
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "RELIANCE.NS": "Energy",
    "ITC.NS": "FMCG",
    "LT.NS": "Infrastructure"
}

@st.cache_data(ttl=3600)
def fetch_ohlcv(ticker: str, period: str = "6mo", retries: int = 1) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker from Yahoo Finance.
    
    Args:
        ticker: The NSE ticker symbol (e.g., 'RELIANCE.NS').
        period: Data period to fetch (e.g., '6mo', '1y', 'max').
        retries: Number of retries on failure.
        
    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume] 
                     and a DatetimeIndex. Returns empty DataFrame on failure.
    """
    for attempt in range(retries + 1):
        try:
            # Download data
            df = yf.download(
                ticker, 
                period=period, 
                auto_adjust=True, 
                progress=False
            )
            
            if df.empty:
                if attempt < retries:
                    logger.warning(f"Empty data for {ticker}, retrying... (Attempt {attempt + 1})")
                    time.sleep(1)
                    continue
                logger.warning(f"No data found for ticker: {ticker}")
                return pd.DataFrame(columns=REQUIRED_COLUMNS)

            # Clean data: drop NaN and ensure expected columns exist
            df = df.dropna()
            
            # CRITICAL: Handle yfinance MultiIndex columns (e.g., ('Close', 'HDFCBANK.NS'))
            # This ensures we always have single-level columns
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"Flattening MultiIndex columns for {ticker}")
                df.columns = df.columns.get_level_values(0)
            
            # Ensure columns are treated as strings and stripped of whitespace
            df.columns = [str(col).strip() for col in df.columns]

            # Filter and reorder to only include the required OHLCV columns
            available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
            
            if len(available_cols) < len(REQUIRED_COLUMNS):
                missing = set(REQUIRED_COLUMNS) - set(available_cols)
                logger.warning(f"Schema mismatch for {ticker}. Missing columns: {missing}")
                if 'Close' not in available_cols:
                    # If we don't even have 'Close', the data is likely unusable for quant analysis
                    logger.error(f"Critical schema failure for {ticker}: 'Close' column missing.")
                    return pd.DataFrame(columns=REQUIRED_COLUMNS)

            return df[available_cols]

        except Exception as e:
            if attempt < retries:
                logger.warning(f"Error fetching {ticker}: {e}. Retrying...")
                time.sleep(1)
                continue
            logger.error(f"Failed to fetch data for {ticker} after {retries + 1} attempts: {e}")
            
    return pd.DataFrame(columns=REQUIRED_COLUMNS)

def fetch_portfolio(
    tickers: List[str] = NIFTY50_TICKERS, 
    period: str = "6mo"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols.
        period: Data period to fetch.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping tickers to their respective DataFrames.
                                 Failed tickers are skipped gracefully.
    """
    portfolio_data = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, period=period)
        if not df.empty:
            portfolio_data[ticker] = df
        else:
            logger.warning(f"Skipping {ticker} due to fetch failure.")
            
    return portfolio_data

def get_latest_prices(tickers: List[str] = NIFTY50_TICKERS) -> Dict[str, float]:
    """
    Get the latest closing prices for a list of tickers.
    
    Args:
        tickers: List of ticker symbols.
        
    Returns:
        Dict[str, float]: Dictionary mapping tickers to their latest close price.
    """
    latest_prices = {}
    try:
        # Download small window to get latest close
        data = yf.download(
            tickers, 
            period="5d", 
            auto_adjust=True, 
            progress=False
        )
        
        if data.empty:
            logger.warning("Failed to fetch latest prices for tickers.")
            return {}

        # Handle multi-ticker download format
        if len(tickers) > 1:
            # If download returns a DataFrame with MultiIndex columns (Price, Ticker)
            if 'Close' in data.columns:
                close_data = data['Close']
            else:
                # auto_adjust=True might put it under 'Close' directly or sometimes 'Adj Close' 
                # but auto_adjust should handle it.
                close_data = data
                
            for ticker in tickers:
                try:
                    if ticker in close_data.columns:
                        series = close_data[ticker].dropna()
                        if not series.empty:
                            latest_prices[ticker] = float(series.iloc[-1])
                except Exception as e:
                    logger.warning(f"Could not extract latest price for {ticker}: {e}")
        else:
            # Single ticker download returns simple columns
            ticker = tickers[0]
            if not data.empty:
                latest_prices[ticker] = float(data['Close'].dropna().iloc[-1])

    except Exception as e:
        logger.error(f"Error in get_latest_prices: {e}")
        
    return latest_prices
