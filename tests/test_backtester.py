import numpy as np
import pandas as pd
import pytest

from backtester import BuyAndHold, MeanReversion, Momentum, SmaCrossover, Strategy, run_backtest
from backtester.metrics import cagr, max_drawdown, sharpe_ratio, total_return


def make_prices(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


class Fixed(Strategy):
    """Test helper: a strategy with hand-picked positions."""

    def __init__(self, positions):
        self._positions = positions
        self.name = "Fixed"

    def positions(self, prices):
        return pd.Series(self._positions, index=prices.index, dtype=float)


# ---- metrics ----

def test_total_return_and_cagr():
    returns = pd.Series([0.10, -0.05])
    assert total_return(returns) == pytest.approx(1.10 * 0.95 - 1)
    one_year = pd.Series([0.0] * 251 + [0.21])
    assert cagr(one_year) == pytest.approx(0.21)


def test_max_drawdown():
    # 100 -> 120 -> 60 -> 90: the worst fall is 120 -> 60 = -50%
    returns = make_prices([100, 120, 60, 90]).pct_change().fillna(0)
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_sharpe_zero_volatility_is_zero_not_nan():
    assert sharpe_ratio(pd.Series([0.0] * 10)) == 0.0


def test_sharpe_matches_formula():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


# ---- engine ----

def test_buy_and_hold_matches_price_without_costs():
    prices = make_prices([100, 110, 99, 120])
    result = run_backtest(prices, BuyAndHold(), cost_bps=0, initial_capital=100)
    # Invested from day 2 onward (signal on day 1's close), so equity tracks price from there.
    assert result.equity.iloc[-1] == pytest.approx(120)


def test_no_look_ahead():
    """A strategy that 'knows' tomorrow's move must not profit from it on the same day."""
    prices = make_prices([100, 50, 100, 50, 100, 50])
    # Be invested exactly on days whose *own* return is positive (cheating if not shifted).
    cheater = Fixed([0, 0, 1, 0, 1, 0])
    result = run_backtest(prices, cheater, cost_bps=0)
    # Because positions are shifted a day, the cheater holds on the down days instead.
    assert result.equity.iloc[-1] < 10_000


def test_costs_are_charged_per_trade():
    prices = make_prices([100] * 6)  # flat market: any loss is pure cost
    result = run_backtest(prices, Fixed([1, 0, 1, 0, 1, 0]), cost_bps=10)
    assert result.trades == 5
    assert result.equity.iloc[-1] == pytest.approx(10_000 * (1 - 0.001) ** 5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        run_backtest(make_prices([100]), BuyAndHold())
    with pytest.raises(ValueError):
        run_backtest(make_prices([100, 101]), BuyAndHold(), cost_bps=-1)


# ---- strategies ----

def test_sma_crossover_invested_in_uptrend_cash_in_downtrend():
    up = list(range(100, 160))
    down = list(range(160, 100, -1))
    positions = SmaCrossover(5, 20).positions(make_prices(up + down))
    assert positions.iloc[:19].eq(0).all()  # not enough history yet
    assert positions.iloc[50] == 1.0
    assert positions.iloc[-1] == 0.0


def test_momentum_uses_only_past_data():
    prices = make_prices(np.linspace(100, 200, 300))
    positions = Momentum(lookback=60, skip=5).positions(prices)
    assert positions.iloc[:60].eq(0).all()
    assert positions.iloc[-1] == 1.0


def test_mean_reversion_enters_on_dip_and_exits_on_recovery():
    values = [100.0, 101.0] * 15 + [90.0] + [100.0] * 5
    positions = MeanReversion(window=20, entry_z=1.0).positions(make_prices(values))
    dip = 30
    assert positions.iloc[dip - 1] == 0.0
    assert positions.iloc[dip] == 1.0  # entered on the dip
    assert positions.iloc[-1] == 0.0  # exited after recovering


@pytest.mark.parametrize("bad", [lambda: SmaCrossover(200, 50), lambda: Momentum(10, 10), lambda: MeanReversion(window=1)])
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        bad()
