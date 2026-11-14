"""Price data: download adjusted daily closes from Yahoo Finance, cached to disk as CSV."""

from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str, start: str | date, end: str | date, cache: bool = True) -> pd.Series:
    """Adjusted close prices for `ticker`, indexed by date.

    Adjusted prices include dividends and splits, so returns are total returns.
    Results are cached so repeated runs (and tests) don't hit the network.
    """
    ticker = ticker.upper().strip()
    path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if cache and path.exists():
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf  # imported lazily so tests and offline use don't need it

        frame = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
        if frame.empty:
            raise ValueError(f"No price data for '{ticker}' between {start} and {end}")
        close = frame["Close"]
        series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        if cache:
            CACHE_DIR.mkdir(exist_ok=True)
            series.rename("close").to_csv(path)
    series = series.dropna().astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = ticker
    return series
