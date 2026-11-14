"""A small, honest stock-strategy backtester."""

from backtester.engine import BacktestResult, run_backtest
from backtester.metrics import compute_metrics
from backtester.strategies import STRATEGIES, BuyAndHold, MeanReversion, Momentum, SmaCrossover, Strategy

__all__ = [
    "BacktestResult",
    "BuyAndHold",
    "MeanReversion",
    "Momentum",
    "STRATEGIES",
    "SmaCrossover",
    "Strategy",
    "compute_metrics",
    "run_backtest",
]
