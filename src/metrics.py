"""
Performance metrics calculator.

All metrics are computed from a daily equity curve DataFrame.
"""

import pandas as pd
import numpy as np
from datetime import date

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, STARTING_CASH


def cagr(equity_curve: pd.DataFrame, min_days: int = 30) -> float | None:
    """
    Compound Annual Growth Rate.
    Returns None if fewer than min_days of data (annualizing
    short periods produces misleading numbers).
    """
    if len(equity_curve) < 2:
        return None
    start_val = equity_curve["total_equity"].iloc[0]
    end_val = equity_curve["total_equity"].iloc[-1]
    days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    if days < min_days or start_val <= 0:
        return None
    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def total_return(equity_curve: pd.DataFrame) -> float:
    """Total return since inception."""
    if len(equity_curve) < 1:
        return 0.0
    return (equity_curve["total_equity"].iloc[-1] / STARTING_CASH) - 1


def max_drawdown(equity_curve: pd.DataFrame) -> float:
    """
    Maximum drawdown (as a negative percentage).
    Returns the largest peak-to-trough decline.
    """
    if len(equity_curve) < 2:
        return 0.0
    equity = equity_curve["total_equity"]
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return float(drawdowns.min())


def drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Full drawdown time series for charting.
    Returns DataFrame with date and drawdown columns.
    """
    if len(equity_curve) < 2:
        return pd.DataFrame({"date": [], "drawdown": []})
    equity = equity_curve["total_equity"]
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return pd.DataFrame({"date": equity_curve["date"], "drawdown": dd.values})


def sharpe_ratio(equity_curve: pd.DataFrame) -> float:
    """
    Annualized Sharpe Ratio.
    (mean excess return / std of returns) * sqrt(252)
    """
    if len(equity_curve) < 30:
        return 0.0
    returns = equity_curve["daily_return"].iloc[1:]  # skip first row (0 return)
    if returns.std() == 0:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity_curve: pd.DataFrame) -> float:
    """
    Annualized Sortino Ratio.
    Like Sharpe but only penalizes downside volatility.
    """
    if len(equity_curve) < 30:
        return 0.0
    returns = equity_curve["daily_return"].iloc[1:]
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = np.sqrt((downside**2).mean())  # semi-deviation
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def beta(
    portfolio_curve: pd.DataFrame, benchmark_curve: pd.DataFrame
) -> float:
    """
    Portfolio beta vs benchmark.
    beta = cov(rp, rb) / var(rb)
    """
    if len(portfolio_curve) < 30 or len(benchmark_curve) < 30:
        return 0.0

    # Align dates
    p_returns = portfolio_curve.set_index("date")["daily_return"]
    b_returns = benchmark_curve.set_index("date")["daily_return"]
    aligned = pd.DataFrame({"portfolio": p_returns, "benchmark": b_returns}).dropna()

    if len(aligned) < 30:
        return 0.0

    cov = aligned["portfolio"].cov(aligned["benchmark"])
    var = aligned["benchmark"].var()
    if var == 0:
        return 0.0
    return float(cov / var)


def alpha(
    portfolio_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    portfolio_beta: float = None,
) -> float | None:
    """
    Jensen's Alpha (annualized).
    alpha = Rp - [Rf + beta * (Rb - Rf)]
    Returns None if not enough data for CAGR.
    """
    if portfolio_beta is None:
        portfolio_beta = beta(portfolio_curve, benchmark_curve)

    rp = cagr(portfolio_curve)
    rb = cagr(benchmark_curve)

    if rp is None or rb is None:
        return None

    return rp - (RISK_FREE_RATE + portfolio_beta * (rb - RISK_FREE_RATE))


def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table of monthly returns: rows = year, columns = month (1-12).
    Values are percentage returns for each month.
    """
    if len(equity_curve) < 2:
        return pd.DataFrame()

    df = equity_curve[["date", "total_equity"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Resample to month-end equity values
    monthly_equity = df["total_equity"].resample("ME").last()

    # Calculate monthly returns
    monthly_ret = monthly_equity.pct_change()

    # Handle first month: return vs starting cash
    if len(monthly_equity) > 0:
        first_month_return = (monthly_equity.iloc[0] / STARTING_CASH) - 1
        monthly_ret.iloc[0] = first_month_return

    # Build pivot
    ret_df = monthly_ret.reset_index()
    ret_df.columns = ["date", "return"]
    ret_df["year"] = ret_df["date"].dt.year
    ret_df["month"] = ret_df["date"].dt.month

    pivot = ret_df.pivot_table(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivot.columns)]

    # Add YTD column
    pivot["YTD"] = (1 + pivot.fillna(0)).prod(axis=1) - 1

    return pivot


def annual_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """Annual return for each calendar year."""
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)

    df = equity_curve[["date", "total_equity"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    yearly = df["total_equity"].resample("YE").last()
    yearly_ret = yearly.pct_change()

    # First year: return vs starting cash
    if len(yearly) > 0:
        yearly_ret.iloc[0] = (yearly.iloc[0] / STARTING_CASH) - 1

    yearly_ret.index = yearly_ret.index.year
    yearly_ret.index.name = "Year"
    return yearly_ret


def trade_statistics(closed_trades: pd.DataFrame) -> dict:
    """
    Compute trade-level statistics from closed P&L DataFrame.

    Returns dict with: total_trades, winners, losers, win_rate,
    avg_win, avg_loss, avg_win_pct, avg_loss_pct, profit_factor,
    largest_win, largest_loss, expectancy
    """
    if closed_trades.empty:
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "expectancy": 0.0,
        }

    wins = closed_trades[closed_trades["pnl"] > 0]
    losses = closed_trades[closed_trades["pnl"] <= 0]

    total = len(closed_trades)
    n_wins = len(wins)
    n_losses = len(losses)

    avg_win = float(wins["pnl"].mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses["pnl"].mean()) if n_losses > 0 else 0.0
    avg_win_pct = float(wins["pnl_pct"].mean()) if n_wins > 0 else 0.0
    avg_loss_pct = float(losses["pnl_pct"].mean()) if n_losses > 0 else 0.0

    total_wins = float(wins["pnl"].sum()) if n_wins > 0 else 0.0
    total_losses = abs(float(losses["pnl"].sum())) if n_losses > 0 else 0.0

    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    win_rate = n_wins / total if total > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if total > 0 else 0.0

    return {
        "total_trades": total,
        "winners": n_wins,
        "losers": n_losses,
        "win_rate": win_rate,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": round(profit_factor, 2),
        "largest_win": round(float(closed_trades["pnl"].max()), 2),
        "largest_loss": round(float(closed_trades["pnl"].min()), 2),
        "expectancy": round(expectancy, 2),
    }
