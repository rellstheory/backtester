"""Compare every strategy on one ticker and print a results table.

Usage:  python -m backtester.report SPY 2005-01-01 2025-01-01
"""

import sys

import pandas as pd

from backtester.data import load_prices
from backtester.engine import BacktestResult, run_backtest
from backtester.strategies import BuyAndHold, MeanReversion, Momentum, SmaCrossover

PERCENT_COLUMNS = ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate", "Exposure"]


def compare(prices: pd.Series, cost_bps: float = 5.0) -> list[BacktestResult]:
    strategies = [BuyAndHold(), SmaCrossover(), Momentum(), MeanReversion()]
    return [run_backtest(prices, s, cost_bps=cost_bps) for s in strategies]


def results_table(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({r.strategy: r.metrics for r in results}).T


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy().astype(object)
    for column in PERCENT_COLUMNS:
        out[column] = table[column].map(lambda v: f"{v:.1%}")
    out["Sharpe"] = table["Sharpe"].map(lambda v: f"{v:.2f}")
    out["Trades"] = table["Trades"].astype(int)
    return out


def main(argv: list[str]) -> None:
    ticker, start, end = (argv + ["SPY", "2005-01-01", "2025-01-01"][len(argv):])[:3]
    prices = load_prices(ticker, start, end)
    print(f"{ticker}: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, {len(prices)} trading days, 5 bps costs\n")
    print(format_table(results_table(compare(prices))).to_markdown())


if __name__ == "__main__":
    main(sys.argv[1:])
