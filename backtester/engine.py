"""The backtest engine: turns target positions into a realistic equity curve."""

from dataclasses import dataclass, field

import pandas as pd

from backtester.metrics import compute_metrics
from backtester.strategies import Strategy


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series  # portfolio value, starting at initial_capital
    returns: pd.Series  # daily strategy returns, after costs
    positions: pd.Series  # position actually held during each day
    trades: int
    metrics: dict[str, float] = field(default_factory=dict)


def run_backtest(
    prices: pd.Series,
    strategy: Strategy,
    cost_bps: float = 5.0,
    initial_capital: float = 10_000.0,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Simulate trading `strategy` on daily `prices`.

    - No look-ahead: a signal computed at day t's close is traded at that close and
      earns day t+1's return. That's why positions are shifted by one day.
    - Costs: every change in position pays `cost_bps` basis points (1 bp = 0.01%)
      of the amount traded, covering commissions and the bid/ask spread.
    """
    if len(prices) < 2:
        raise ValueError("need at least two prices")
    if cost_bps < 0:
        raise ValueError("cost_bps can't be negative")

    prices = prices.sort_index()
    asset_returns = prices.pct_change().fillna(0.0)
    target = strategy.positions(prices).reindex(prices.index).fillna(0.0).clip(0.0, 1.0)
    held = target.shift(1).fillna(0.0)

    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * cost_bps / 10_000
    returns = held * asset_returns - costs

    equity = initial_capital * (1 + returns).cumprod()
    trades = int((turnover > 0).sum())
    result = BacktestResult(strategy.describe(), equity, returns, held, trades)
    result.metrics = compute_metrics(returns, held, risk_free_rate)
    result.metrics["Trades"] = trades
    return result
