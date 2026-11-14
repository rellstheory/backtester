"""Performance metrics. All annualized figures assume 252 trading days per year."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate: the steady yearly return that ends at the same place."""
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    growth = (1 + returns).prod()
    if growth <= 0:
        return -1.0  # lost everything
    return float(growth ** (1 / years) - 1)


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Average excess return per unit of volatility, annualized. Above 1 is good; above 2 is rare."""
    excess = returns - risk_free_rate / TRADING_DAYS
    std = excess.std(ddof=1)
    if not std or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drop of the equity curve, as a negative fraction (-0.34 = -34%)."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def win_rate(returns: pd.Series, positions: pd.Series) -> float:
    """Share of invested days that made money (days in cash don't count)."""
    invested = returns[positions > 0]
    return float((invested > 0).mean()) if len(invested) else 0.0


def exposure(positions: pd.Series) -> float:
    """Fraction of days with money in the market."""
    return float((positions > 0).mean())


def compute_metrics(returns: pd.Series, positions: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    return {
        "Total Return": total_return(returns),
        "CAGR": cagr(returns),
        "Volatility": annual_volatility(returns),
        "Sharpe": sharpe_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(returns),
        "Win Rate": win_rate(returns, positions),
        "Exposure": exposure(positions),
    }
