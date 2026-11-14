"""Streamlit UI.  Run with:  streamlit run app.py"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester.data import load_prices
from backtester.engine import run_backtest
from backtester.report import format_table, results_table
from backtester.strategies import BuyAndHold, MeanReversion, Momentum, SmaCrossover

st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("Stock Strategy Backtester")
st.caption("Compare trading strategies against buy-and-hold, with transaction costs and no look-ahead bias.")

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    start = st.date_input("Start", date(2005, 1, 1), min_value=date(1990, 1, 1))
    end = st.date_input("End", date(2025, 1, 1))
    cost_bps = st.number_input("Cost per trade (bps)", 0.0, 100.0, 5.0, 1.0, help="1 bp = 0.01% of the amount traded")
    rf = st.number_input("Risk-free rate (annual %)", 0.0, 10.0, 0.0, 0.25) / 100

    st.header("Strategies")
    use_sma = st.checkbox("SMA Crossover", True)
    fast = st.slider("Fast SMA (days)", 5, 150, 50, disabled=not use_sma)
    slow = st.slider("Slow SMA (days)", 20, 300, 200, disabled=not use_sma)
    use_mom = st.checkbox("Momentum", True)
    lookback = st.slider("Momentum lookback (days)", 21, 504, 252, disabled=not use_mom)
    use_mr = st.checkbox("Mean Reversion", True)
    window = st.slider("Mean-reversion window (days)", 5, 100, 20, disabled=not use_mr)
    entry_z = st.slider("Enter below z-score", 0.5, 3.0, 1.0, 0.25, disabled=not use_mr)

if start >= end:
    st.error("Start date must be before end date.")
    st.stop()
if use_sma and fast >= slow:
    st.error("Fast SMA must be shorter than slow SMA.")
    st.stop()


@st.cache_data(show_spinner="Downloading prices…")
def cached_prices(ticker: str, start: date, end: date) -> pd.Series:
    return load_prices(ticker, start, end)


try:
    prices = cached_prices(ticker, start, end)
except Exception as exc:  # network errors, bad tickers
    st.error(f"Couldn't load prices for {ticker}: {exc}")
    st.stop()

strategies = [BuyAndHold()]
if use_sma:
    strategies.append(SmaCrossover(fast, slow))
if use_mom:
    strategies.append(Momentum(lookback, min(21, lookback - 1)))
if use_mr:
    strategies.append(MeanReversion(window, entry_z))

results = [run_backtest(prices, s, cost_bps=cost_bps, risk_free_rate=rf) for s in strategies]

st.subheader(f"{ticker}: {prices.index[0]:%b %Y} to {prices.index[-1]:%b %Y}")

equity = go.Figure()
for r in results:
    equity.add_trace(go.Scatter(x=r.equity.index, y=r.equity, name=r.strategy, mode="lines"))
equity.update_layout(yaxis_title="Value of $10,000 (log scale)", yaxis_type="log", yaxis_tickformat="$,.0f", height=420, margin=dict(t=10), legend=dict(orientation="h", y=-0.15))
st.plotly_chart(equity, use_container_width=True)

st.dataframe(format_table(results_table(results)), use_container_width=True)

drawdown = go.Figure()
for r in results:
    curve = r.equity / r.equity.cummax() - 1
    drawdown.add_trace(go.Scatter(x=curve.index, y=curve, name=r.strategy, mode="lines"))
drawdown.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".0%", height=300, margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
st.subheader("Drawdowns")
st.plotly_chart(drawdown, use_container_width=True)

with st.expander("How to read this, and its limits"):
    st.markdown(
        """
- **CAGR** is the steady yearly growth rate. **Sharpe** is return per unit of risk (above 1 is good).
  **Max drawdown** is the worst fall from a peak, the number that makes real investors panic-sell.
- Signals use only past data and trade the next day, and every trade pays the cost you set.
- **Overfitting:** tuning sliders until a strategy wins finds noise, not an edge. Check settings on other tickers and periods.
- **Survivorship bias:** testing on today's well-known tickers ignores companies that failed.
- Past performance doesn't predict future results. This is a learning tool, not investment advice.
"""
    )
