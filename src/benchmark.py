"""
Benchmark module: fetch SPY data and compute benchmark metrics
for side-by-side comparison with the portfolio.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENCHMARK_TICKER, STARTING_CASH, TRADING_DAYS_PER_YEAR
from src import metrics


def fetch_benchmark(start_date: date, end_date: date = None) -> pd.DataFrame:
    """
    Fetch benchmark (SPY) daily data and build an equity curve
    as if $STARTING_CASH was invested on start_date.

    Returns DataFrame matching the portfolio equity curve format:
    date, total_equity, daily_return, cumulative_return
    """
    if end_date is None:
        end_date = date.today()

    end_dt = end_date + timedelta(days=1)

    data = yf.download(
        BENCHMARK_TICKER,
        start=start_date.isoformat(),
        end=end_dt.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame(
            columns=["date", "total_equity", "daily_return", "cumulative_return"]
        )

    # Handle yfinance MultiIndex columns (even for single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"][BENCHMARK_TICKER].dropna()
    else:
        prices = data["Close"].dropna()

    # Flatten to a 1D Series
    prices = prices.squeeze()

    # Build equity curve: invest STARTING_CASH in SPY on day 1
    initial_price = float(prices.iloc[0])
    shares = STARTING_CASH / initial_price

    dates = [d.date() if hasattr(d, "date") else d for d in prices.index]
    equity_values = (prices.values.flatten() * shares).round(2)

    curve = pd.DataFrame(
        {
            "date": dates,
            "total_equity": equity_values,
        }
    )

    curve["daily_return"] = curve["total_equity"].pct_change().fillna(0)
    curve["cumulative_return"] = (curve["total_equity"] / STARTING_CASH) - 1

    return curve


def benchmark_metrics(benchmark_curve: pd.DataFrame) -> dict:
    """
    Calculate all standard metrics for the benchmark.
    Returns dict of metric name -> value.
    """
    return {
        "total_return": metrics.total_return(benchmark_curve),
        "cagr": metrics.cagr(benchmark_curve),
        "max_drawdown": metrics.max_drawdown(benchmark_curve),
        "sharpe": metrics.sharpe_ratio(benchmark_curve),
        "sortino": metrics.sortino_ratio(benchmark_curve),
        "volatility": _annualized_volatility(benchmark_curve),
    }


def portfolio_metrics(
    portfolio_curve: pd.DataFrame, benchmark_curve: pd.DataFrame
) -> dict:
    """
    Calculate all standard metrics for the portfolio (including beta/alpha vs benchmark).
    Returns dict of metric name -> value.
    """
    b = metrics.beta(portfolio_curve, benchmark_curve)
    return {
        "total_return": metrics.total_return(portfolio_curve),
        "cagr": metrics.cagr(portfolio_curve),
        "max_drawdown": metrics.max_drawdown(portfolio_curve),
        "sharpe": metrics.sharpe_ratio(portfolio_curve),
        "sortino": metrics.sortino_ratio(portfolio_curve),
        "beta": b,
        "alpha": metrics.alpha(portfolio_curve, benchmark_curve, b),
        "volatility": _annualized_volatility(portfolio_curve),
    }


def comparison_table(
    portfolio_curve: pd.DataFrame, benchmark_curve: pd.DataFrame
) -> pd.DataFrame:
    """
    Side-by-side comparison table: Portfolio vs SPY.
    """
    p = portfolio_metrics(portfolio_curve, benchmark_curve)
    b = benchmark_metrics(benchmark_curve)

    def fmt_pct(val):
        return f"{val:.2%}" if val is not None else "—"

    def fmt_num(val):
        return f"{val:.2f}" if val is not None else "—"

    rows = [
        {"Metric": "Total Return", "Portfolio": fmt_pct(p['total_return']), "SPY": fmt_pct(b['total_return'])},
        {"Metric": "CAGR", "Portfolio": fmt_pct(p['cagr']), "SPY": fmt_pct(b['cagr'])},
        {"Metric": "Max Drawdown", "Portfolio": fmt_pct(p['max_drawdown']), "SPY": fmt_pct(b['max_drawdown'])},
        {"Metric": "Sharpe Ratio", "Portfolio": fmt_num(p['sharpe']), "SPY": fmt_num(b['sharpe'])},
        {"Metric": "Sortino Ratio", "Portfolio": fmt_num(p['sortino']), "SPY": fmt_num(b['sortino'])},
        {"Metric": "Volatility (Ann.)", "Portfolio": fmt_pct(p['volatility']), "SPY": fmt_pct(b['volatility'])},
        {"Metric": "Beta", "Portfolio": fmt_num(p['beta']), "SPY": "1.00"},
        {"Metric": "Alpha (Ann.)", "Portfolio": fmt_pct(p['alpha']), "SPY": "—"},
    ]

    return pd.DataFrame(rows)


def _annualized_volatility(equity_curve: pd.DataFrame) -> float:
    """Annualized volatility from daily returns."""
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve["daily_return"].iloc[1:]
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
