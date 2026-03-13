"""
Parse WeBull trade exports into a clean, normalized DataFrame.

Supports both:
- XLSX files (the actual format WeBull emails you)
- CSV files (in case WeBull changes formats or for manual entry)
"""

import pandas as pd
import numpy as np
from io import StringIO, BytesIO


# Map of all known WeBull column names → our normalized names.
# Covers both the real xlsx export format and the older CSV format.
COLUMN_MAP = {
    # Ticker
    "Symbol": "ticker",
    "symbol": "ticker",
    # Side
    "Side": "side",
    "side": "side",
    # Quantity
    "Filled Qty": "quantity",
    "filled_qty": "quantity",
    "Total Qty": "quantity_ordered",
    "Qty": "quantity_ordered",
    "qty": "quantity_ordered",
    # Price (real xlsx uses "Average Price", older CSV uses "Avg Price")
    "Average Price": "price",
    "Avg Price": "price",
    "avg_price": "price",
    "Filled Price": "filled_price",
    "Price": "order_price",
    "price": "order_price",
    # Date
    "Filled Time": "date",
    "filled_time": "date",
    "update_time": "date",
    # Status (real xlsx uses "Execute Status", older CSV uses "Status")
    "Execute Status": "status",
    "Status": "status",
    "status": "status",
    # Total
    "Filled Amount": "total",
    "Total": "total",
    "total": "total",
    # Order type
    "Order Type": "order_type",
    "order_type": "order_type",
}


def load_trades_from_file(file_path: str) -> pd.DataFrame:
    """Load and parse a WeBull export from a file path (csv or xlsx)."""
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return _normalize_trades(df)


def load_trades_from_upload(uploaded_file) -> pd.DataFrame:
    """Load and parse a WeBull export from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(BytesIO(raw))
    else:
        df = pd.read_csv(StringIO(raw.decode("utf-8")))

    return _normalize_trades(df)


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw WeBull export into a clean DataFrame with columns:
    date, trade_date, ticker, side, quantity, price, total
    """
    if df.empty:
        return _empty_trades_df()

    # Rename columns using the mapping
    renamed = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[col_stripped]
    df = df.rename(columns=renamed)

    # Filter to filled orders only (if status column exists)
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.upper()
        filled_mask = df["status"].isin(["FILLED", "FULLY FILLED"])
        df = df[filled_mask].copy()

    # Ensure required columns exist
    required = ["ticker", "side"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Resolve price: prefer "Average Price" → "Filled Price" → "Price"
    if "price" not in df.columns:
        if "filled_price" in df.columns:
            df["price"] = df["filled_price"]
        elif "order_price" in df.columns:
            df["price"] = df["order_price"]
        else:
            raise ValueError("No price column found (Average Price, Filled Price, or Price)")

    # Resolve quantity: prefer "Filled Qty" → "Total Qty" / "Qty"
    if "quantity" not in df.columns:
        if "quantity_ordered" in df.columns:
            df["quantity"] = df["quantity_ordered"]
        else:
            raise ValueError("No quantity column found (Filled Qty or Total Qty)")

    # Parse date
    if "date" not in df.columns:
        raise ValueError("No date column found (Filled Time)")

    df["date"] = df["date"].astype(str)
    # Strip timezone abbreviations like EST, EDT
    df["date"] = df["date"].str.replace(
        r"\s+(EST|EDT|CST|CDT|MST|MDT|PST|PDT)$", "", regex=True
    )
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False, utc=True)
    # Convert to naive datetime (drop timezone) for simpler handling
    df["date"] = df["date"].dt.tz_localize(None)

    # Drop rows with NaT dates (unfilled orders may have no fill time)
    df = df.dropna(subset=["date"])

    # Normalize side to uppercase BUY/SELL
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    df.loc[df["side"].isin(["BUY", "B"]), "side"] = "BUY"
    df.loc[df["side"].isin(["SELL", "S"]), "side"] = "SELL"

    # Clean numeric columns
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace("$", "").str.replace(",", ""),
        errors="coerce",
    ).fillna(0)

    # Calculate total if not present
    if "total" not in df.columns:
        df["total"] = df["quantity"] * df["price"]
    else:
        df["total"] = pd.to_numeric(
            df["total"].astype(str).str.replace("$", "").str.replace(",", ""),
            errors="coerce",
        ).fillna(df["quantity"] * df["price"])

    # Clean ticker symbols
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Extract just the date (no time) for daily grouping
    df["trade_date"] = df["date"].dt.date

    # Select and sort
    result = df[["date", "trade_date", "ticker", "side", "quantity", "price", "total"]].copy()
    result = result.sort_values("date").reset_index(drop=True)

    # Drop any rows with zero quantity or price
    result = result[(result["quantity"] > 0) & (result["price"] > 0)].reset_index(drop=True)

    return result


def _empty_trades_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(
        columns=["date", "trade_date", "ticker", "side", "quantity", "price", "total"]
    )
