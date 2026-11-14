"""Trading strategies.

Every strategy turns a price series into a *target position* per day:
1.0 = fully invested, 0.0 = all cash. The engine decides when that position
can actually be traded (the next day), so strategies never need to worry
about look-ahead bias themselves.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class Strategy(ABC):
    name: str = "Strategy"

    @abstractmethod
    def positions(self, prices: pd.Series) -> pd.Series:
        """Target position (0.0 to 1.0) at each day's close, using only data up to that day."""

    def describe(self) -> str:
        return self.name


@dataclass
class BuyAndHold(Strategy):
    name: str = "Buy & Hold"

    def positions(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


@dataclass
class SmaCrossover(Strategy):
    """Trend following: be invested while the fast moving average is above the slow one."""

    fast: int = 50
    slow: int = 200
    name: str = "SMA Crossover"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("need 0 < fast < slow")

    def positions(self, prices: pd.Series) -> pd.Series:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        # Before the slow average exists (NaN), comparisons are False, so we stay in cash.
        return (fast > slow).astype(float)

    def describe(self) -> str:
        return f"{self.name} ({self.fast}/{self.slow})"


@dataclass
class Momentum(Strategy):
    """Time-series momentum: invested when the trailing return over `lookback` days is positive.

    Skips the most recent `skip` days, a standard tweak because very short-term
    returns tend to reverse.
    """

    lookback: int = 252
    skip: int = 21
    name: str = "Momentum"

    def __post_init__(self) -> None:
        if not 0 <= self.skip < self.lookback:
            raise ValueError("need 0 <= skip < lookback")

    def positions(self, prices: pd.Series) -> pd.Series:
        past = prices.shift(self.skip)
        trailing = past / prices.shift(self.lookback) - 1
        return (trailing > 0).astype(float)

    def describe(self) -> str:
        return f"{self.name} ({self.lookback}d, skip {self.skip}d)"


@dataclass
class MeanReversion(Strategy):
    """Buy when price is unusually far below its recent average; sell once it reverts.

    z-score = (price - rolling mean) / rolling std. Enter below -entry_z,
    exit when z rises back above exit_z. Holding state carries day to day.
    """

    window: int = 20
    entry_z: float = 1.0
    exit_z: float = 0.0
    name: str = "Mean Reversion"

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.exit_z < -self.entry_z:
            raise ValueError("exit_z must be at or above -entry_z")

    def positions(self, prices: pd.Series) -> pd.Series:
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        z = (prices - mean) / std
        position = 0.0
        out = []
        for value in z.to_numpy():
            if position == 0.0 and value < -self.entry_z:
                position = 1.0
            elif position == 1.0 and value > self.exit_z:
                position = 0.0
            out.append(position)
        return pd.Series(out, index=prices.index)

    def describe(self) -> str:
        return f"{self.name} ({self.window}d, enter z<-{self.entry_z}, exit z>{self.exit_z})"


STRATEGIES: dict[str, type[Strategy]] = {
    "Buy & Hold": BuyAndHold,
    "SMA Crossover": SmaCrossover,
    "Momentum": Momentum,
    "Mean Reversion": MeanReversion,
}
