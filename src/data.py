"""Load and clean daily price files exported from the Damascus Securities Exchange (DSE).

Each CSV is semicolon-separated with the columns:
date;openning;up;down;closing_price;changging;t_size
(date as dd/mm/YYYY). They are renamed to standard OHLCV names here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "openning": "open",
    "up": "high",
    "down": "low",
    "closing_price": "close",
    "changging": "change_pct",
    "t_size": "volume",
}
FEATURES = ["open", "high", "low", "close", "change_pct", "volume"]


def load_prices(path: str | Path) -> pd.DataFrame:
    """Return a date-indexed, chronologically sorted OHLCV frame."""
    df = pd.read_csv(path, sep=";").drop(columns=["Close"], errors="ignore")
    df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["change_pct"] = (
        df["change_pct"].astype(str).str.rstrip("%").replace("nan", "0")
    )
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", *FEATURES])
    # rows of zeros are days with no trading; drop them
    df = df[df["close"] > 0].sort_values("date")
    return df.set_index("date")


def make_supervised(df: pd.DataFrame, lags: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build scale-free features and a next-day return target.

    Syrian share prices rose many-fold between 2010 and 2024, mostly from
    inflation. A model trained on price levels cannot predict levels it has
    never seen, so the models learn the next-day *return* instead, from
    features that are ratios rather than prices:

    - intraday shape: high/close, low/close, open/close
    - the last `lags` daily returns
    - log change in volume

    Returns X, the target return (close[t+1] / close[t] - 1), and close[t],
    which turns a predicted return back into a price.
    """
    c = df["close"]
    ret = c.pct_change()
    feats = pd.DataFrame(index=df.index)
    feats["high_close"] = df["high"] / c - 1
    feats["low_close"] = df["low"] / c - 1
    feats["open_close"] = df["open"] / c - 1
    for k in range(lags):
        feats[f"ret_lag{k}"] = ret.shift(k)
    feats["log_vol_change"] = np.log1p(df["volume"]).diff()
    target = (c.shift(-1) / c - 1).rename("next_return")
    data = feats.join(target).join(c.rename("close_t")).replace([np.inf, -np.inf], np.nan).dropna()
    return data[feats.columns], data["next_return"], data["close_t"]


def chronological_split(X, y, train=0.7, val=0.15):
    """Split by time (no shuffling) so the test set is genuinely in the future."""
    n = len(X)
    a, b = int(n * train), int(n * (train + val))
    return (X.iloc[:a], y.iloc[:a]), (X.iloc[a:b], y.iloc[a:b]), (X.iloc[b:], y.iloc[b:])


def list_companies(data_dir: str | Path) -> dict[str, Path]:
    """Map a URL-friendly company slug to its CSV file."""
    out = {}
    for p in sorted(Path(data_dir).glob("*.csv")):
        name = p.stem.split(" 2010")[0].strip()
        slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        out[slug] = p
    return out
