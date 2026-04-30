"""
One-page PDF tear sheet generator.

Produces an investor-ready performance summary using fpdf2 + matplotlib.
"""

import io
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fpdf import FPDF
from datetime import date

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STARTING_CASH, BENCHMARK_TICKER, RISK_FREE_RATE


# -----------------------------------------------------------------------
# Chart helpers (matplotlib → PNG bytes)
# -----------------------------------------------------------------------

BRAND_RED = "#D4213D"
STEEL_GRAY = "#6B7A8D"
CHARCOAL = "#1C1D2A"


def _equity_chart_png(
    portfolio_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    width_in: float = 7.5,
    height_in: float = 2.8,
) -> bytes:
    """Render equity curve as PNG bytes."""
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    dates_p = pd.to_datetime(portfolio_curve["date"])
    ax.plot(dates_p, portfolio_curve["total_equity"], color=BRAND_RED, linewidth=1.5, label="Portfolio")

    if not benchmark_curve.empty:
        dates_b = pd.to_datetime(benchmark_curve["date"])
        ax.plot(dates_b, benchmark_curve["total_equity"], color=STEEL_GRAY, linewidth=1, linestyle="--", label=BENCHMARK_TICKER)

    ax.set_title("Growth of $1,000,000", fontsize=9, color=CHARCOAL, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _drawdown_chart_png(
    equity_curve: pd.DataFrame,
    width_in: float = 7.5,
    height_in: float = 1.5,
) -> bytes:
    """Render drawdown chart as PNG bytes."""
    from src.metrics import drawdown_series

    dd = drawdown_series(equity_curve)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    dates = pd.to_datetime(dd["date"])
    ax.fill_between(dates, dd["drawdown"], 0, color=BRAND_RED, alpha=0.25)
    ax.plot(dates, dd["drawdown"], color=BRAND_RED, linewidth=0.8)
    ax.set_title("Drawdown", fontsize=9, color=CHARCOAL, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------
# PDF builder
# -----------------------------------------------------------------------


class TearSheet(FPDF):
    """Custom FPDF subclass with header/footer branding."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(28, 29, 42)  # CHARCOAL
        self.cell(0, 8, "Vector Growth Capital - Pilot", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(107, 122, 141)  # STEEL_GRAY
        self.cell(0, 5, "Performance Tear Sheet", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        # Red accent line
        self.set_draw_color(212, 33, 61)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(
            0,
            5,
            "This document is for informational purposes only and does not constitute investment advice. "
            "Past performance is not indicative of future results.",
            align="C",
        )


def generate_tearsheet(
    portfolio_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    metrics: dict,
    top_dd: pd.DataFrame,
    monthly_ret: pd.DataFrame,
) -> bytes:
    """
    Generate a one-page PDF tear sheet.

    Args:
        portfolio_curve: Daily equity curve DataFrame
        benchmark_curve: Benchmark equity curve DataFrame
        metrics: Dict with keys: total_return, cagr, max_drawdown, sharpe,
                 sortino, beta, alpha, volatility, up_capture, down_capture
        top_dd: Top drawdowns DataFrame from metrics.top_drawdowns()
        monthly_ret: Monthly returns pivot from metrics.monthly_returns()

    Returns:
        PDF file as bytes.
    """
    pdf = TearSheet(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ---- Period info ----
    inception = portfolio_curve["date"].iloc[0]
    today = date.today()
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(28, 29, 42)
    pdf.cell(
        0, 5,
        f"Period: {inception}  to  {today}    |    "
        f"Starting Capital: ${STARTING_CASH:,.0f}    |    "
        f"Benchmark: {BENCHMARK_TICKER}    |    "
        f"Risk-Free Rate: {RISK_FREE_RATE:.1%}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)

    # ---- KPI row ----
    _kpi_row(pdf, metrics)
    pdf.ln(3)

    # ---- Equity curve chart ----
    eq_png = _equity_chart_png(portfolio_curve, benchmark_curve)
    eq_path = io.BytesIO(eq_png)
    pdf.image(eq_path, x=10, w=190)
    pdf.ln(2)

    # ---- Drawdown chart ----
    dd_png = _drawdown_chart_png(portfolio_curve)
    dd_path = io.BytesIO(dd_png)
    pdf.image(dd_path, x=10, w=190)
    pdf.ln(4)

    # ---- Top Drawdowns table ----
    if not top_dd.empty:
        _section_title(pdf, "Top Drawdowns")
        _drawdown_table(pdf, top_dd)
        pdf.ln(3)

    # ---- Monthly returns table (if fits on page) ----
    if not monthly_ret.empty and pdf.get_y() < 220:
        _section_title(pdf, "Monthly Returns (%)")
        _monthly_table(pdf, monthly_ret)

    # Return bytes
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------
# PDF helper functions
# -----------------------------------------------------------------------


def _section_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(28, 29, 42)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _fmt(val, kind="pct"):
    if val is None:
        return "--"
    if kind == "pct":
        return f"{val:+.2%}" if isinstance(val, (int, float)) else str(val)
    if kind == "num":
        return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
    if kind == "capture":
        return f"{val:.0f}%" if isinstance(val, (int, float)) else str(val)
    return str(val)


def _kpi_row(pdf: FPDF, m: dict):
    """Render a row of KPI boxes."""
    kpis = [
        ("Total Return", _fmt(m.get("total_return"), "pct")),
        ("CAGR", _fmt(m.get("cagr"), "pct")),
        ("Max Drawdown", _fmt(m.get("max_drawdown"), "pct")),
        ("Sharpe", _fmt(m.get("sharpe"), "num")),
        ("Sortino", _fmt(m.get("sortino"), "num")),
        ("Beta", _fmt(m.get("beta"), "num")),
        ("Up Capture", _fmt(m.get("up_capture"), "capture")),
        ("Down Capture", _fmt(m.get("down_capture"), "capture")),
    ]
    box_w = 190 / len(kpis)
    start_y = pdf.get_y()

    for i, (label, value) in enumerate(kpis):
        x = 10 + i * box_w
        pdf.set_xy(x, start_y)
        # Box background
        pdf.set_fill_color(245, 245, 247)
        pdf.rect(x, start_y, box_w - 1, 14, style="F")
        # Label
        pdf.set_xy(x, start_y + 1)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(107, 122, 141)
        pdf.cell(box_w - 1, 4, label, align="C")
        # Value
        pdf.set_xy(x, start_y + 5.5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(28, 29, 42)
        pdf.cell(box_w - 1, 6, value, align="C")

    pdf.set_y(start_y + 16)


def _drawdown_table(pdf: FPDF, dd_df: pd.DataFrame):
    headers = ["#", "Start", "Trough", "Recovery", "Depth", "Days"]
    col_w = [10, 35, 35, 35, 30, 25]

    # Header
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(245, 245, 247)
    pdf.set_text_color(28, 29, 42)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 5, h, border=0, fill=True, align="C")
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 7.5)
    for idx, row in dd_df.iterrows():
        pdf.set_text_color(28, 29, 42)
        pdf.cell(col_w[0], 4.5, str(idx), align="C")
        pdf.cell(col_w[1], 4.5, str(row["Start"]), align="C")
        pdf.cell(col_w[2], 4.5, str(row["Trough"]), align="C")
        rec = str(row["Recovery"]) if row["Recovery"] != "Ongoing" else "Ongoing"
        pdf.cell(col_w[3], 4.5, rec, align="C")
        # Depth in red
        pdf.set_text_color(212, 33, 61)
        pdf.cell(col_w[4], 4.5, f"{row['Depth']:.2%}", align="C")
        pdf.set_text_color(28, 29, 42)
        pdf.cell(col_w[5], 4.5, str(row["Duration (days)"]), align="C")
        pdf.ln()


def _monthly_table(pdf: FPDF, monthly: pd.DataFrame):
    cols = list(monthly.columns)
    col_w = 14.5
    first_w = 16

    # Header
    pdf.set_font("Helvetica", "B", 6.5)
    pdf.set_fill_color(245, 245, 247)
    pdf.set_text_color(28, 29, 42)
    pdf.cell(first_w, 4.5, "Year", fill=True, align="C")
    for c in cols:
        pdf.cell(col_w, 4.5, str(c), fill=True, align="C")
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 6.5)
    for year, row in monthly.iterrows():
        pdf.cell(first_w, 4, str(year), align="C")
        for c in cols:
            val = row[c]
            if pd.isna(val):
                pdf.set_text_color(160, 160, 160)
                pdf.cell(col_w, 4, "--", align="C")
            elif val >= 0:
                pdf.set_text_color(0, 166, 62)  # green
                pdf.cell(col_w, 4, f"{val:.1%}", align="C")
            else:
                pdf.set_text_color(212, 33, 61)  # red
                pdf.cell(col_w, 4, f"{val:.1%}", align="C")
        pdf.ln()
