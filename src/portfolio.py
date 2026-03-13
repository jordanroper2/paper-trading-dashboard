"""
Portfolio engine: replay trades to build daily equity curve.

Takes a normalized trades DataFrame and produces a daily time series of:
- Cash balance
- Holdings value (mark-to-market)
- Total equity
- Daily returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STARTING_CASH, CACHE_TTL_SECONDS


def _fetch_prices(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch daily closing prices for a list of tickers from yfinance.
    Returns a DataFrame with date index and ticker columns.
    """
    if not tickers:
        return pd.DataFrame()

    end_dt = end_date + timedelta(days=1)  # yfinance end is exclusive
    data = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_dt.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    # yf.download returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single ticker — column is just "Close"
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index).date
    return prices


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Replay trades chronologically and build a daily equity curve.

    Args:
        trades: Normalized trades DataFrame from data_loader

    Returns:
        DataFrame with columns:
        date, cash, holdings_value, total_equity, daily_return, cumulative_return
    """
    if trades.empty:
        today = date.today()
        return pd.DataFrame(
            {
                "date": [today],
                "cash": [STARTING_CASH],
                "holdings_value": [0.0],
                "total_equity": [float(STARTING_CASH)],
                "daily_return": [0.0],
                "cumulative_return": [0.0],
            }
        )

    # Get unique tickers and date range
    tickers = trades["ticker"].unique().tolist()
    first_trade_date = trades["trade_date"].min()
    end_date = date.today()

    # Fetch historical prices for all tickers
    prices = _fetch_prices(tickers, first_trade_date, end_date)

    if prices.empty:
        raise ValueError("Could not fetch price data from Yahoo Finance")

    # Build list of all trading days from first trade to today
    all_dates = sorted(prices.index)

    # Group trades by date for efficient replay
    trades_by_date = {}
    for _, trade in trades.iterrows():
        td = trade["trade_date"]
        if td not in trades_by_date:
            trades_by_date[td] = []
        trades_by_date[td].append(trade)

    # Replay engine
    cash = float(STARTING_CASH)
    positions = {}  # ticker -> shares held
    cost_basis = {}  # ticker -> total cost (for P&L tracking)

    daily_records = []

    for d in all_dates:
        # Execute any trades on this date
        if d in trades_by_date:
            for trade in trades_by_date[d]:
                ticker = trade["ticker"]
                qty = int(trade["quantity"])
                price = float(trade["price"])
                cost = qty * price

                if trade["side"] == "BUY":
                    cash -= cost
                    positions[ticker] = positions.get(ticker, 0) + qty
                    cost_basis[ticker] = cost_basis.get(ticker, 0.0) + cost
                elif trade["side"] == "SELL":
                    cash += cost
                    positions[ticker] = positions.get(ticker, 0) - qty
                    # Reduce cost basis proportionally
                    if ticker in cost_basis and positions.get(ticker, 0) >= 0:
                        prev_shares = positions[ticker] + qty
                        if prev_shares > 0:
                            cost_per_share = cost_basis[ticker] / prev_shares
                            cost_basis[ticker] -= cost_per_share * qty
                        else:
                            cost_basis[ticker] = 0.0

                    # Clean up closed positions
                    if positions.get(ticker, 0) == 0:
                        positions.pop(ticker, None)
                        cost_basis.pop(ticker, None)

        # Mark-to-market: value all open positions at today's close
        holdings_value = 0.0
        for ticker, shares in positions.items():
            if ticker in prices.columns and d in prices.index:
                px = prices.loc[d, ticker]
                if pd.notna(px):
                    holdings_value += shares * float(px)

        total_equity = cash + holdings_value

        daily_records.append(
            {
                "date": d,
                "cash": round(cash, 2),
                "holdings_value": round(holdings_value, 2),
                "total_equity": round(total_equity, 2),
            }
        )

    # Build DataFrame
    curve = pd.DataFrame(daily_records)

    # Calculate daily and cumulative returns
    curve["daily_return"] = curve["total_equity"].pct_change().fillna(0)
    curve["cumulative_return"] = (curve["total_equity"] / STARTING_CASH) - 1

    return curve


def get_trade_pnl(trades: pd.DataFrame, prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate P&L for each closed round-trip trade using FIFO matching.

    Returns DataFrame with: ticker, buy_date, sell_date, quantity, buy_price,
    sell_price, pnl, pnl_pct
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "buy_date", "sell_date", "quantity",
                "buy_price", "sell_price", "pnl", "pnl_pct",
            ]
        )

    # FIFO matching per ticker
    open_lots = {}  # ticker -> list of (date, quantity, price)
    closed_trades = []

    for _, trade in trades.sort_values("date").iterrows():
        ticker = trade["ticker"]
        qty = int(trade["quantity"])
        price = float(trade["price"])
        trade_date = trade["trade_date"]

        if trade["side"] == "BUY":
            if ticker not in open_lots:
                open_lots[ticker] = []
            open_lots[ticker].append({"date": trade_date, "qty": qty, "price": price})

        elif trade["side"] == "SELL":
            remaining = qty
            if ticker not in open_lots:
                continue

            while remaining > 0 and open_lots[ticker]:
                lot = open_lots[ticker][0]
                matched_qty = min(remaining, lot["qty"])

                pnl = matched_qty * (price - lot["price"])
                pnl_pct = (price - lot["price"]) / lot["price"]

                closed_trades.append(
                    {
                        "ticker": ticker,
                        "buy_date": lot["date"],
                        "sell_date": trade_date,
                        "quantity": matched_qty,
                        "buy_price": lot["price"],
                        "sell_price": price,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 4),
                    }
                )

                lot["qty"] -= matched_qty
                remaining -= matched_qty

                if lot["qty"] == 0:
                    open_lots[ticker].pop(0)

    return pd.DataFrame(closed_trades)
