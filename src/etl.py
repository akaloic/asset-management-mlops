"""
ETL Pipeline for Asset Management.
Data ingestion, cleaning, and feature engineering for financial data.
"""
import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import sys

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, ASSETS, 
    START_DATE, END_DATE, ETL_CONFIG, LOG_CONFIG, LOGS_DIR
)

# Logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format=LOG_CONFIG["format"],
    level=LOG_CONFIG["level"]
)
logger.add(
    LOGS_DIR / "etl_{time:YYYY-MM-DD}.log",
    rotation=LOG_CONFIG["rotation"],
    retention=LOG_CONFIG["retention"],
    level=LOG_CONFIG["level"]
)


def fetch_data(ticker: str, start: str = START_DATE, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download ticker history from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol
        start: Start date (format YYYY-MM-DD)
        end: End date (format YYYY-MM-DD), None = today
        
    Returns:
        DataFrame with historical data
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    try:
        logger.info(f"Downloading {ticker} from {start} to {end}")
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return pd.DataFrame()
        
        # Flatten multi-index columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        
        # Rename Date to Date if necessary
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        logger.success(f"Data retrieved for {ticker}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()


def compute_var(returns: pd.Series, confidence: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) for a return series.
    
    Args:
        returns: Return series
        confidence: Confidence level (0.05 = VaR at 95%)
        
    Returns:
        VaR (negative value)
    """
    if returns.empty or returns.isna().all():
        return 0.0
    return float(np.percentile(returns.dropna(), confidence * 100))


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        window: Calculation window (default: 14)
        
    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast period (default: 12)
        slow: Slow period (default: 26)
        signal: Signal period (default: 9)
        
    Returns:
        DataFrame with MACD, Signal, Histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })


def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add advanced business features: volatility, drawdown, VaR, financial ratios, etc.
    
    Args:
        df: DataFrame with raw data
        ticker: Ticker symbol for logging
        
    Returns:
        Enriched DataFrame with features
    """
    if df.empty or 'Close' not in df.columns:
        logger.warning(f"Empty DataFrame or missing Close column for {ticker}")
        return df
    
    logger.info(f"Feature engineering for {ticker}")
    df = df.copy()
    
    # Returns
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling volatility on different windows
    windows = [21, 63, 252]  # 1 month, 3 months, 1 year
    for window in windows:
        df[f'Volatility_{window}d'] = df['Return'].rolling(window=window).std() * np.sqrt(252)
        df[f'Sharpe_{window}d'] = (
            df['Return'].rolling(window=window).mean() /
            df['Return'].rolling(window=window).std()
        ) * np.sqrt(252)
        df[f'VaR_{window}d'] = df['Return'].rolling(window=window).apply(
            lambda x: compute_var(x, confidence=0.05), raw=False
        )
    
    # Drawdown
    df['Cum_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Cum_Max']) / df['Cum_Max']
    df['Max_Drawdown'] = df['Drawdown'].rolling(window=252).min()
    
    # RSI
    df['RSI_14'] = compute_rsi(df['Close'], window=14)
    
    # MACD
    macd_df = compute_macd(df['Close'])
    df = pd.concat([df, macd_df], axis=1)
    
    # Moving Averages
    for window in [20, 50, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Price_vs_MA_{window}'] = (df['Close'] - df[f'MA_{window}']) / df[f'MA_{window}']
    
    # Volume features
    if 'Volume' in df.columns:
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Momentum
    for window in [5, 10, 21]:
        df[f'Momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
    
    # Bollinger Bands
    window_bb = 20
    df['BB_Middle'] = df['Close'].rolling(window=window_bb).mean()
    bb_std = df['Close'].rolling(window=window_bb).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Cleanup
    df = df.dropna()
    
    logger.success(f"Features calculated for {ticker}: {len(df.columns)} columns, {len(df)} rows")
    return df


def compute_correlation_matrix(assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix between all assets.
    
    Args:
        assets_data: Dictionary {ticker: DataFrame} with asset data
        
    Returns:
        Correlation matrix
    """
    logger.info("Computing correlation matrix")
    
    # Extract returns from each asset
    returns_df = pd.DataFrame()
    for ticker, df in assets_data.items():
        if not df.empty and 'Return' in df.columns:
            returns_df[ticker] = df.set_index('Date')['Return']
    
    # Align dates
    returns_df = returns_df.dropna()
    
    # Calculate correlation
    correlation_matrix = returns_df.corr()
    
    logger.success(f"Correlation matrix computed: {correlation_matrix.shape}")
    return correlation_matrix


def save_df(df: pd.DataFrame, name: str, folder: Path) -> None:
    """
    Save DataFrame as CSV in the given folder.
    
    Args:
        df: DataFrame to save
        name: File name (without extension)
        folder: Destination folder
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for {name}, skipping save")
        return
    
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{name}.csv"
    df.to_csv(file_path, index=False)
    logger.success(f"Data saved: {file_path}")


def main():
    """Main ETL pipeline function."""
    logger.info("=" * 60)
    logger.info("STARTING ASSET MANAGEMENT ETL PIPELINE")
    logger.info("=" * 60)
    
    assets_data = {}
    
    # 1. Download raw data
    logger.info("STEP 1: Downloading raw data")
    for ticker in ASSETS:
        raw_df = fetch_data(ticker)
        if not raw_df.empty:
            save_df(raw_df, ticker, RAW_DATA_DIR)
            assets_data[ticker] = raw_df
    
    # 2. Feature engineering
    logger.info("STEP 2: Feature engineering")
    processed_data = {}
    for ticker, raw_df in assets_data.items():
        processed_df = compute_features(raw_df, ticker)
        if not processed_df.empty:
            save_df(processed_df, ticker, PROCESSED_DATA_DIR)
            processed_data[ticker] = processed_df
    
    # 3. Calculate correlation matrix
    logger.info("STEP 3: Computing correlation matrix")
    if len(processed_data) > 1:
        correlation_matrix = compute_correlation_matrix(processed_data)
        save_df(
            correlation_matrix.reset_index(),
            "correlation_matrix",
            PROCESSED_DATA_DIR
        )
    
    logger.info("=" * 60)
    logger.success("ETL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return processed_data


if __name__ == "__main__":
    main()
