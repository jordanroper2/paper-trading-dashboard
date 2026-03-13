"""
Exposure analysis: sector allocation, position concentration, and holdings breakdown.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# Manual overrides for assets yfinance doesn't classify well
CRYPTO_TICKERS = {"BTC", "ETH", "DOGE", "SOL", "ADA", "XRP", "AVAX", "MATIC", "LTC"}

KNOWN_ETFS = {
    "SPY", "IVV", "VOO", "VTI", "QQQ", "IWM", "DIA", "VEA", "VWO",
    "AGG", "BND", "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _get_sector_cached(ticker: str) -> str:
    """Get sector for a single ticker (cached 24h)."""
    if ticker in CRYPTO_TICKERS:
        return "Cryptocurrency"
    if ticker in KNOWN_ETFS:
        return "ETF"
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", None)
        if sector:
            return sector
        # If no sector, check if it's a fund
        qt = info.get("quoteType", "")
        if qt == "ETF":
            return "ETF"
        return "Other"
    except Exception:
        return "Other"


def get_current_positions(trades: pd.DataFrame) -> dict[str, int]:
    """
    Replay trades to extract current open positions.
    Returns dict of ticker -> shares held.
    """
    positions: dict[str, int] = {}
    for _, trade in trades.sort_values("date").iterrows():
        ticker = trade["ticker"]
        qty = int(trade["quantity"])
        if trade["side"] == "BUY":
            positions[ticker] = positions.get(ticker, 0) + qty
        elif trade["side"] == "SELL":
            positions[ticker] = positions.get(ticker, 0) - qty
            if positions.get(ticker, 0) <= 0:
                positions.pop(ticker, None)
    return positions


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_latest_prices(tickers: tuple) -> dict[str, float]:
    """Fetch latest closing prices for a list of tickers."""
    if not tickers:
        return {}
    data = yf.download(
        list(tickers),
        period="5d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return {}

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})

    # Take last available close for each ticker
    prices = {}
    for t in tickers:
        if t in closes.columns:
            last_valid = closes[t].dropna()
            if len(last_valid) > 0:
                prices[t] = float(last_valid.iloc[-1])
    return prices


def build_exposure_table(
    trades: pd.DataFrame, total_equity: float
) -> pd.DataFrame:
    """
    Build full exposure table with position details, sector, and weight.
    Returns sorted DataFrame: Ticker, Shares, Price, Market Value, Weight, Sector.
    """
    positions = get_current_positions(trades)
    if not positions:
        return pd.DataFrame(
            columns=["Ticker", "Shares", "Price", "Market Value", "Weight", "Sector"]
        )

    prices = _fetch_latest_prices(tuple(sorted(positions.keys())))

    rows = []
    for ticker, shares in positions.items():
        price = prices.get(ticker, 0.0)
        mv = shares * price
        weight = mv / total_equity if total_equity > 0 else 0.0
        sector = _get_sector_cached(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Price": price,
                "Market Value": mv,
                "Weight": weight,
                "Sector": sector,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Market Value", ascending=False).reset_index(drop=True)
    return df


def sector_allocation(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure by sector. Returns Sector, Market Value, Weight."""
    if exposure_df.empty:
        return pd.DataFrame(columns=["Sector", "Market Value", "Weight"])
    grouped = (
        exposure_df.groupby("Sector")[["Market Value", "Weight"]]
        .sum()
        .sort_values("Weight", ascending=False)
        .reset_index()
    )
    return grouped


def concentration_metrics(exposure_df: pd.DataFrame) -> dict:
    """
    Position concentration metrics.
    Returns: top5_weight, hhi (Herfindahl index), num_positions, gross_exposure.
    """
    if exposure_df.empty:
        return {
            "top5_weight": 0.0,
            "hhi": 0.0,
            "num_positions": 0,
            "gross_exposure": 0.0,
        }
    weights = exposure_df["Weight"].values
    top5 = float(np.sum(sorted(weights, reverse=True)[:5]))
    hhi = float(np.sum(weights**2))
    return {
        "top5_weight": top5,
        "hhi": hhi,
        "num_positions": len(exposure_df),
        "gross_exposure": float(np.sum(np.abs(weights))),
    }
