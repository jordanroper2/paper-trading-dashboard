"""
Paper Trading Performance Dashboard
====================================
Streamlit dashboard that tracks WeBull paper trading performance
with full metrics benchmarked against SPY.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import os
import sys

# Ensure imports work regardless of CWD
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

from config import STARTING_CASH, BENCHMARK_TICKER, RISK_FREE_RATE
from src.data_loader import load_trades_from_file, load_trades_from_upload
from src.portfolio import build_equity_curve, get_trade_pnl
from src.metrics import (
    cagr, total_return, max_drawdown, sharpe_ratio, sortino_ratio,
    beta, monthly_returns, annual_returns, trade_statistics,
    drawdown_series,
)
from src.benchmark import fetch_benchmark, comparison_table

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Roper Advisory Group — Paper Trading",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Brand palette (derived from Roper Advisory Group logo)
# ---------------------------------------------------------------------------
BRAND_RED = "#D4213D"
BRAND_DARK = "#0C0E14"
BRAND_CARD = "#151820"
STEEL_GRAY = "#6B7A8D"
MUTED_RED = "#7A2030"

st.markdown(
    f"""
    <style>
    /* Metric cards */
    .metric-card {{
        background-color: {BRAND_CARD};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2D222A;
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: #8892A0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 4px 0;
    }}
    .metric-compare {{
        font-size: 0.75rem;
        color: #8892A0;
    }}
    .positive {{ color: {BRAND_RED}; }}
    .negative {{ color: {STEEL_GRAY}; }}
    .neutral  {{ color: #F0F0F0; }}

    /* Streamlit metric delta override — positive = red, negative = gray */
    [data-testid="stMetricDelta"] svg {{
        display: none;
    }}

    /* Divider styling */
    hr {{
        border-color: #1E2230 !important;
    }}

    /* Subtle red accent on sidebar */
    [data-testid="stSidebar"] {{
        border-right: 2px solid #2A1520 !important;
    }}

    /* Dataframe header accent */
    .stDataFrame thead th {{
        background-color: #1A1520 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_benchmark_data(start_date: date):
    """Fetch benchmark data (cached for 24h)."""
    return fetch_benchmark(start_date)


# ---------------------------------------------------------------------------
# Sidebar — file upload
# ---------------------------------------------------------------------------
st.sidebar.title("📁 Trade Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload WeBull Export",
    type=["csv", "xlsx", "xls"],
    help="Export your trade history from the WeBull mobile app. Supports .xlsx and .csv files.",
)

use_sample = st.sidebar.checkbox("Use sample data (demo)", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Starting Cash:** ${STARTING_CASH:,.0f}  \n"
    f"**Benchmark:** {BENCHMARK_TICKER}  \n"
    f"**Risk-Free Rate:** {RISK_FREE_RATE:.1%}"
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_sample_trades() -> pd.DataFrame:
    """Load sample/local trades from the data folder."""
    import glob
    # Look for any xlsx or csv file in data/ (prefer xlsx)
    for pattern in ["data/*.xlsx", "data/*.xls", "data/*.csv"]:
        files = sorted(glob.glob(pattern))
        if files:
            try:
                return load_trades_from_file(files[0])
            except Exception:
                continue
    return pd.DataFrame(
        columns=["date", "trade_date", "ticker", "side", "quantity", "price", "total"]
    )


if uploaded_file is not None:
    try:
        trades = load_trades_from_upload(uploaded_file)
        st.sidebar.success(f"Loaded {len(trades)} trades")
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {e}")
        trades = load_sample_trades()
elif use_sample:
    trades = load_sample_trades()
else:
    trades = pd.DataFrame(
        columns=["date", "trade_date", "ticker", "side", "quantity", "price", "total"]
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Paper Trading Dashboard")

if trades.empty:
    st.info(
        "No trades loaded. Upload a WeBull CSV export from the sidebar, "
        "or check 'Use sample data' to see a demo."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Build portfolio & benchmark curves
# ---------------------------------------------------------------------------
with st.spinner("Building portfolio and fetching market data..."):
    portfolio_curve = build_equity_curve(trades)
    first_date = portfolio_curve["date"].iloc[0]
    benchmark_curve = get_benchmark_data(first_date)
    closed_trades = get_trade_pnl(trades)

inception_date = first_date
current_value = portfolio_curve["total_equity"].iloc[-1]
total_ret = total_return(portfolio_curve)
portfolio_cagr = cagr(portfolio_curve)
portfolio_mdd = max_drawdown(portfolio_curve)
portfolio_sharpe = sharpe_ratio(portfolio_curve)
portfolio_sortino = sortino_ratio(portfolio_curve)
portfolio_beta = beta(portfolio_curve, benchmark_curve)

benchmark_total_ret = total_return(benchmark_curve) if not benchmark_curve.empty else 0.0
benchmark_cagr_val = cagr(benchmark_curve) if not benchmark_curve.empty else None


# ---------------------------------------------------------------------------
# Helper: format metric with color
# ---------------------------------------------------------------------------
def fmt_pct(val, invert=False):
    """Format percentage with color class."""
    if invert:
        cls = "positive" if val <= 0 else "negative"
    else:
        cls = "positive" if val >= 0 else "negative"
    return f'<span class="{cls}">{val:+.2%}</span>'


def fmt_num(val, decimals=2):
    """Format number with color class."""
    cls = "positive" if val >= 0 else "negative"
    return f'<span class="{cls}">{val:+.{decimals}f}</span>'


# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
st.markdown(f"**Inception:** {inception_date} &nbsp;|&nbsp; **Last Updated:** {date.today()}")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.metric("Portfolio Value", f"${current_value:,.0f}")
with col2:
    st.metric("Total Return", f"{total_ret:+.2%}", delta=f"SPY: {benchmark_total_ret:+.2%}")
with col3:
    cagr_display = f"{portfolio_cagr:+.2%}" if portfolio_cagr is not None else "—"
    cagr_delta = f"SPY: {benchmark_cagr_val:+.2%}" if benchmark_cagr_val is not None else "Need 30+ days"
    st.metric("CAGR", cagr_display, delta=cagr_delta)
with col4:
    st.metric("Max Drawdown", f"{portfolio_mdd:.2%}")
with col5:
    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
with col6:
    st.metric("Sortino Ratio", f"{portfolio_sortino:.2f}")
with col7:
    st.metric("Beta vs SPY", f"{portfolio_beta:.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Equity curve + drawdown chart
# ---------------------------------------------------------------------------
st.subheader("Equity Curve — Portfolio vs SPY")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("Growth of $1,000,000", "Drawdown"),
)

# Portfolio equity
fig.add_trace(
    go.Scatter(
        x=portfolio_curve["date"],
        y=portfolio_curve["total_equity"],
        name="Portfolio",
        line=dict(color=BRAND_RED, width=2.5),
        hovertemplate="$%{y:,.0f}<extra>Portfolio</extra>",
    ),
    row=1,
    col=1,
)

# Benchmark equity
if not benchmark_curve.empty:
    fig.add_trace(
        go.Scatter(
            x=benchmark_curve["date"],
            y=benchmark_curve["total_equity"],
            name=f"{BENCHMARK_TICKER}",
            line=dict(color=STEEL_GRAY, width=1.5, dash="dot"),
            hovertemplate="$%{y:,.0f}<extra>SPY</extra>",
        ),
        row=1,
        col=1,
    )

# Drawdown
dd = drawdown_series(portfolio_curve)
fig.add_trace(
    go.Scatter(
        x=dd["date"],
        y=dd["drawdown"],
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=MUTED_RED, width=1),
        fillcolor="rgba(122, 32, 48, 0.25)",
        hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=600,
    template="plotly_dark",
    paper_bgcolor=BRAND_DARK,
    plot_bgcolor=BRAND_DARK,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
    font=dict(color="#C0C4CC"),
)
fig.update_xaxes(gridcolor="#1E2230", zeroline=False)
fig.update_yaxes(gridcolor="#1E2230", zeroline=False)
fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickprefix="$", tickformat=",")
fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".1%")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Performance comparison table + Monthly returns
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Performance Comparison")
    comp = comparison_table(portfolio_curve, benchmark_curve)
    st.dataframe(comp, hide_index=True, use_container_width=True)

with col_right:
    st.subheader("Monthly Returns")
    monthly = monthly_returns(portfolio_curve)

    if not monthly.empty:
        # Format as percentages and color-code with brand palette
        styled = monthly.style.format("{:.2%}", na_rep="—").map(
            lambda v: (
                f"color: {BRAND_RED}; font-weight: 600"
                if isinstance(v, (int, float)) and v > 0
                else (
                    f"color: {STEEL_GRAY}; font-weight: 600"
                    if isinstance(v, (int, float)) and v < 0
                    else "color: #555B68"
                )
            )
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Not enough data for monthly returns")

st.markdown("---")

# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------
st.subheader("Trade Statistics")

stats = trade_statistics(closed_trades)

col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)

with col_a:
    st.metric("Total Trades", stats["total_trades"])
with col_b:
    st.metric("Win Rate", f"{stats['win_rate']:.1%}")
with col_c:
    st.metric("Avg Win", f"${stats['avg_win']:,.2f}")
with col_d:
    st.metric("Avg Loss", f"${stats['avg_loss']:,.2f}")
with col_e:
    st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
with col_f:
    st.metric("Expectancy", f"${stats['expectancy']:,.2f}")

# Additional trade stats
col_g, col_h, col_i, col_j = st.columns(4)
with col_g:
    st.metric("Winners", stats["winners"])
with col_h:
    st.metric("Losers", stats["losers"])
with col_i:
    st.metric("Largest Win", f"${stats['largest_win']:,.2f}")
with col_j:
    st.metric("Largest Loss", f"${stats['largest_loss']:,.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------
st.subheader("Trade Log")

display_trades = trades.copy()
display_trades["date"] = pd.to_datetime(display_trades["date"]).dt.strftime("%Y-%m-%d %H:%M")
display_trades["price"] = display_trades["price"].map("${:,.2f}".format)
display_trades["total"] = display_trades["total"].map("${:,.2f}".format)

st.dataframe(
    display_trades[["date", "ticker", "side", "quantity", "price", "total"]].rename(
        columns={
            "date": "Date",
            "ticker": "Ticker",
            "side": "Side",
            "quantity": "Qty",
            "price": "Price",
            "total": "Total",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Closed trades P&L detail
# ---------------------------------------------------------------------------
if not closed_trades.empty:
    st.subheader("Closed Trade P&L")

    ct_display = closed_trades.copy()
    ct_display["buy_price"] = ct_display["buy_price"].map("${:,.2f}".format)
    ct_display["sell_price"] = ct_display["sell_price"].map("${:,.2f}".format)
    ct_display["pnl"] = ct_display["pnl"].map("${:,.2f}".format)
    ct_display["pnl_pct"] = ct_display["pnl_pct"].map("{:+.2%}".format)

    st.dataframe(
        ct_display.rename(
            columns={
                "ticker": "Ticker",
                "buy_date": "Buy Date",
                "sell_date": "Sell Date",
                "quantity": "Qty",
                "buy_price": "Buy Price",
                "sell_price": "Sell Price",
                "pnl": "P&L",
                "pnl_pct": "P&L %",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Roper Advisory Group — Paper Trading Dashboard | Data from Yahoo Finance | "
    f"Benchmark: {BENCHMARK_TICKER} | Risk-Free Rate: {RISK_FREE_RATE:.1%}"
)
