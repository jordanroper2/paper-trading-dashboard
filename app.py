"""
Stoa Capital Management - Pilot Dashboard
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
    drawdown_series, rolling_sharpe, rolling_beta, rolling_volatility,
    top_drawdowns, up_capture, down_capture,
)
from src.benchmark import fetch_benchmark, comparison_table
from src.exposure import build_exposure_table, sector_allocation, concentration_metrics
from src.tearsheet import generate_tearsheet

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stoa Capital Management - Pilot",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
BRAND_RED = "#D4213D"
BRAND_CHARCOAL = "#1C1D2A"
BRAND_BG = "#FFFFFF"
BRAND_CARD_BG = "#F5F5F7"
STEEL_GRAY = "#6B7A8D"
MUTED_RED = "#D4213D"
GRID_COLOR = "#E8E8EC"

st.markdown(
    f"""
    <style>
    /* Center the dashboard title */
    h1 {{
        text-align: center !important;
        padding-bottom: 0.2rem !important;
    }}

    /* Center subheaders */
    h3 {{
        text-align: center !important;
        padding-top: 0.5rem !important;
    }}

    /* Center metrics within their columns */
    [data-testid="stMetric"] {{
        background-color: {BRAND_CARD_BG};
        border: 1px solid #E0E0E4;
        border-radius: 8px;
        padding: 12px 8px;
        text-align: center;
    }}
    [data-testid="stMetric"] > div {{
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    [data-testid="stMetric"] label {{
        font-size: 0.8rem !important;
        color: {STEEL_GRAY} !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 1.3rem !important;
    }}

    /* Legacy metric card classes */
    .positive {{ color: #00A63E; }}
    .negative {{ color: {BRAND_RED}; }}
    .neutral  {{ color: {BRAND_CHARCOAL}; }}

    /* Streamlit metric delta override */
    [data-testid="stMetricDelta"] svg {{
        display: none;
    }}

    /* Center inception/updated line */
    .inception-line {{
        text-align: center;
        color: {STEEL_GRAY};
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }}

    /* Divider styling */
    hr {{
        border-color: #E0E0E4 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }}

    /* Subtle red accent on sidebar */
    [data-testid="stSidebar"] {{
        border-right: 2px solid {BRAND_RED} !important;
    }}

    /* Center the footer caption */
    .stCaption {{
        text-align: center !important;
    }}

    /* Tabs centering */
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
        gap: 1rem;
    }}

    /* Center dataframes */
    [data-testid="stDataFrame"] {{
        margin-left: auto;
        margin-right: auto;
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Starting Cash:** ${STARTING_CASH:,.0f}  \n"
    f"**Benchmark:** {BENCHMARK_TICKER}  \n"
    f"**Risk-Free Rate:** {RISK_FREE_RATE:.1%}"
)

# PDF tear sheet download (populated after data loads)
tearsheet_placeholder = st.sidebar.empty()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        trades = load_trades_from_upload(uploaded_file)
        st.sidebar.success(f"Loaded {len(trades)} trades")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
        trades = pd.DataFrame(
            columns=["date", "trade_date", "ticker", "side", "quantity", "price", "total"]
        )
else:
    trades = pd.DataFrame(
        columns=["date", "trade_date", "ticker", "side", "quantity", "price", "total"]
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Stoa Capital Management - Pilot")

if trades.empty:
    st.info("No trades loaded. Upload a WeBull trade export from the sidebar to get started.")
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
portfolio_up_capture = up_capture(portfolio_curve, benchmark_curve) if not benchmark_curve.empty else None
portfolio_down_capture = down_capture(portfolio_curve, benchmark_curve) if not benchmark_curve.empty else None


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
st.markdown(
    f'<div class="inception-line"><strong>Inception:</strong> {inception_date} &nbsp;&bull;&nbsp; '
    f'<strong>Last Updated:</strong> {date.today()}</div>',
    unsafe_allow_html=True,
)

row1 = st.columns(4)
with row1[0]:
    st.metric("Portfolio Value", f"${current_value:,.0f}")
with row1[1]:
    st.metric("Total Return", f"{total_ret:+.2%}", delta=f"SPY: {benchmark_total_ret:+.2%}")
with row1[2]:
    cagr_display = f"{portfolio_cagr:+.2%}" if portfolio_cagr is not None else "—"
    cagr_delta = f"SPY: {benchmark_cagr_val:+.2%}" if benchmark_cagr_val is not None else "Need 30+ days"
    st.metric("CAGR", cagr_display, delta=cagr_delta)
with row1[3]:
    st.metric("Max Drawdown", f"{portfolio_mdd:.2%}")

row2 = st.columns(5)
with row2[0]:
    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
with row2[1]:
    st.metric("Sortino Ratio", f"{portfolio_sortino:.2f}")
with row2[2]:
    st.metric("Beta vs SPY", f"{portfolio_beta:.2f}")
with row2[3]:
    uc_display = f"{portfolio_up_capture:.0f}%" if portfolio_up_capture is not None else "—"
    st.metric("Up Capture", uc_display)
with row2[4]:
    dc_display = f"{portfolio_down_capture:.0f}%" if portfolio_down_capture is not None else "—"
    st.metric("Down Capture", dc_display)

st.markdown("---")

# ---------------------------------------------------------------------------
# Equity curve + drawdown chart
# ---------------------------------------------------------------------------
st.subheader("Equity Curve")

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
        fillcolor="rgba(212, 33, 61, 0.12)",
        hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=600,
    template="plotly_white",
    paper_bgcolor=BRAND_BG,
    plot_bgcolor=BRAND_BG,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
    font=dict(color=BRAND_CHARCOAL),
)
fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickprefix="$", tickformat=",")
fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".1%")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Rolling Metrics (Sharpe, Beta, Volatility)
# ---------------------------------------------------------------------------
st.subheader("Rolling Metrics")

roll_tab1, roll_tab2, roll_tab3 = st.tabs(["Rolling Sharpe", "Rolling Beta", "Rolling Volatility"])

ROLLING_COLORS = {
    "30d": BRAND_RED,
    "60d": STEEL_GRAY,
    "90d": BRAND_CHARCOAL,
}

with roll_tab1:
    r_sharpe = rolling_sharpe(portfolio_curve)
    data_cols = [c for c in r_sharpe.columns if c != "date"]
    if data_cols:
        fig_rs = go.Figure()
        for col in data_cols:
            fig_rs.add_trace(go.Scatter(
                x=r_sharpe["date"], y=r_sharpe[col],
                name=col, line=dict(color=ROLLING_COLORS.get(col, BRAND_RED), width=1.5),
                hovertemplate="%{y:.2f}<extra>" + col + "</extra>",
            ))
        fig_rs.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR)
        fig_rs.update_layout(
            height=300, template="plotly_white",
            paper_bgcolor=BRAND_BG, plot_bgcolor=BRAND_BG,
            font=dict(color=BRAND_CHARCOAL),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=30),
            yaxis_title="Sharpe Ratio",
        )
        fig_rs.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
        fig_rs.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
        st.plotly_chart(fig_rs, use_container_width=True)
    else:
        st.info("Need at least 30 days of data for rolling Sharpe")

with roll_tab2:
    r_beta = rolling_beta(portfolio_curve, benchmark_curve)
    data_cols = [c for c in r_beta.columns if c != "date"]
    if data_cols:
        fig_rb = go.Figure()
        for col in data_cols:
            fig_rb.add_trace(go.Scatter(
                x=r_beta["date"], y=r_beta[col],
                name=col, line=dict(color=ROLLING_COLORS.get(col, BRAND_RED), width=1.5),
                hovertemplate="%{y:.2f}<extra>" + col + "</extra>",
            ))
        fig_rb.add_hline(y=1, line_dash="dot", line_color=GRID_COLOR, annotation_text="Market (1.0)")
        fig_rb.update_layout(
            height=300, template="plotly_white",
            paper_bgcolor=BRAND_BG, plot_bgcolor=BRAND_BG,
            font=dict(color=BRAND_CHARCOAL),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=30),
            yaxis_title="Beta",
        )
        fig_rb.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
        fig_rb.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
        st.plotly_chart(fig_rb, use_container_width=True)
    else:
        st.info("Need at least 30 days of data for rolling Beta")

with roll_tab3:
    r_vol = rolling_volatility(portfolio_curve)
    data_cols = [c for c in r_vol.columns if c != "date"]
    if data_cols:
        fig_rv = go.Figure()
        for col in data_cols:
            fig_rv.add_trace(go.Scatter(
                x=r_vol["date"], y=r_vol[col],
                name=col, line=dict(color=ROLLING_COLORS.get(col, BRAND_RED), width=1.5),
                hovertemplate="%{y:.2%}<extra>" + col + "</extra>",
            ))
        fig_rv.update_layout(
            height=300, template="plotly_white",
            paper_bgcolor=BRAND_BG, plot_bgcolor=BRAND_BG,
            font=dict(color=BRAND_CHARCOAL),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=30),
            yaxis_title="Annualized Volatility",
            yaxis_tickformat=".1%",
        )
        fig_rv.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
        fig_rv.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
        st.plotly_chart(fig_rv, use_container_width=True)
    else:
        st.info("Need at least 30 days of data for rolling Volatility")

st.markdown("---")

# ---------------------------------------------------------------------------
# Drawdown Table (Top 5)
# ---------------------------------------------------------------------------
st.subheader("Top Drawdowns")

dd_table = top_drawdowns(portfolio_curve)
if not dd_table.empty:
    dd_display = dd_table.copy()
    dd_display["Depth"] = dd_display["Depth"].map("{:.2%}".format)
    st.dataframe(dd_display, use_container_width=True)
else:
    st.info("No drawdown periods detected yet")

st.markdown("---")

# ---------------------------------------------------------------------------
# Exposure Breakdown
# ---------------------------------------------------------------------------
st.subheader("Exposure Breakdown")

exposure_df = build_exposure_table(trades, current_value)

if not exposure_df.empty:
    conc = concentration_metrics(exposure_df)

    # Concentration KPIs
    exp_cols = st.columns(4)
    with exp_cols[0]:
        st.metric("Positions", conc["num_positions"])
    with exp_cols[1]:
        st.metric("Top 5 Weight", f"{conc['top5_weight']:.1%}")
    with exp_cols[2]:
        st.metric("HHI (Concentration)", f"{conc['hhi']:.4f}")
    with exp_cols[3]:
        st.metric("Gross Exposure", f"{conc['gross_exposure']:.1%}")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        # Sector allocation pie
        sectors = sector_allocation(exposure_df)
        fig_pie = go.Figure(go.Pie(
            labels=sectors["Sector"],
            values=sectors["Weight"],
            hole=0.45,
            marker=dict(colors=[
                BRAND_RED, STEEL_GRAY, BRAND_CHARCOAL, "#E8927C",
                "#A3B5C9", "#D4A0A0", "#8FA89E", "#C4B08B",
                "#7A8BA0", "#B0BEC5", "#D4A373", "#9E9E9E",
            ]),
            textinfo="label+percent",
            textfont=dict(size=10),
        ))
        fig_pie.update_layout(
            height=350, template="plotly_white",
            paper_bgcolor=BRAND_BG, font=dict(color=BRAND_CHARCOAL),
            title=dict(text="Sector Allocation", font=dict(size=14), x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_right:
        # Top holdings bar chart
        top_n = exposure_df.head(10)
        fig_bar = go.Figure(go.Bar(
            x=top_n["Weight"],
            y=top_n["Ticker"],
            orientation="h",
            marker=dict(color=BRAND_RED),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig_bar.update_layout(
            height=350, template="plotly_white",
            paper_bgcolor=BRAND_BG, plot_bgcolor=BRAND_BG,
            font=dict(color=BRAND_CHARCOAL),
            title=dict(text="Top Holdings by Weight", font=dict(size=14), x=0.5, xanchor="center"),
            margin=dict(l=60, r=20, t=50, b=30),
            xaxis=dict(tickformat=".0%", gridcolor=GRID_COLOR),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Holdings detail table
    with st.expander("Full Holdings Detail"):
        hold_display = exposure_df.copy()
        hold_display["Price"] = hold_display["Price"].map("${:,.2f}".format)
        hold_display["Market Value"] = hold_display["Market Value"].map("${:,.0f}".format)
        hold_display["Weight"] = hold_display["Weight"].map("{:.2%}".format)
        st.dataframe(hold_display, hide_index=True, use_container_width=True)
else:
    st.info("No open positions to analyze")

st.markdown("---")

# ---------------------------------------------------------------------------
# Performance comparison table + Monthly returns
# ---------------------------------------------------------------------------
st.subheader("Monthly Returns")
monthly = monthly_returns(portfolio_curve)

if not monthly.empty:
    styled = monthly.style.format("{:.2%}", na_rep="--").map(
        lambda v: (
            "color: #00A63E; font-weight: 600"
            if isinstance(v, (int, float)) and v > 0
            else (
                f"color: {BRAND_RED}; font-weight: 600"
                if isinstance(v, (int, float)) and v < 0
                else "color: #A0A4AE"
            )
        )
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("Not enough data for monthly returns")

st.subheader("Performance Comparison")
comp = comparison_table(portfolio_curve, benchmark_curve)
st.dataframe(comp, hide_index=True, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------
st.subheader("Trade Statistics")

stats = trade_statistics(closed_trades)

ts_row1 = st.columns(5)
with ts_row1[0]:
    st.metric("Total Trades", stats["total_trades"])
with ts_row1[1]:
    st.metric("Winners", stats["winners"])
with ts_row1[2]:
    st.metric("Losers", stats["losers"])
with ts_row1[3]:
    st.metric("Win Rate", f"{stats['win_rate']:.1%}")
with ts_row1[4]:
    st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

ts_row2 = st.columns(5)
with ts_row2[0]:
    st.metric("Avg Win", f"${stats['avg_win']:,.2f}")
with ts_row2[1]:
    st.metric("Avg Loss", f"${stats['avg_loss']:,.2f}")
with ts_row2[2]:
    st.metric("Largest Win", f"${stats['largest_win']:,.2f}")
with ts_row2[3]:
    st.metric("Largest Loss", f"${stats['largest_loss']:,.2f}")
with ts_row2[4]:
    st.metric("Expectancy", f"${stats['expectancy']:,.2f}")

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
# PDF Tear Sheet (sidebar download)
# ---------------------------------------------------------------------------
from src.benchmark import portfolio_metrics as _pm, _annualized_volatility

_ts_metrics = {
    "total_return": total_ret,
    "cagr": portfolio_cagr,
    "max_drawdown": portfolio_mdd,
    "sharpe": portfolio_sharpe,
    "sortino": portfolio_sortino,
    "beta": portfolio_beta,
    "alpha": None,
    "volatility": _annualized_volatility(portfolio_curve),
    "up_capture": portfolio_up_capture,
    "down_capture": portfolio_down_capture,
}

try:
    pdf_bytes = generate_tearsheet(
        portfolio_curve, benchmark_curve,
        _ts_metrics, dd_table, monthly,
    )
    tearsheet_placeholder.download_button(
        label="📄 Download Tear Sheet (PDF)",
        data=pdf_bytes,
        file_name=f"SCM_tearsheet_{date.today()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
except Exception:
    pass  # Silently skip if PDF generation fails

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f'<p style="text-align:center; color:{STEEL_GRAY}; font-size:0.8rem;">'
    f"Stoa Capital Management - Pilot &nbsp;&bull;&nbsp; Data from Yahoo Finance &nbsp;&bull;&nbsp; "
    f"Benchmark: {BENCHMARK_TICKER} &nbsp;&bull;&nbsp; Risk-Free Rate: {RISK_FREE_RATE:.1%}</p>",
    unsafe_allow_html=True,
)
