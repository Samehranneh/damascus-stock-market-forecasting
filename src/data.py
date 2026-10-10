"""Load and clean daily price files exported from the Damascus Securities Exchange (DSE).

Each CSV is semicolon-separated with the columns:
date;openning;up;down;closing_price;changging;t_size
(date as dd/mm/YYYY). They are renamed to standard OHLCV names here.
"""
from __future__ import annotations

from pathlib import Path

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
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["change_pct"] = (
        df["change_pct"].astype(str).str.rstrip("%").replace("nan", "0")
    )
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", *FEATURES]).sort_values("date")
    return df.set_index("date")


def make_supervised(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Features on day t, target = closing price on day t + horizon.

    Shifting the target forward means the model never sees the value it
    has to predict, which is what makes the scores meaningful.
    """
    target = df["close"].shift(-horizon).rename("target_close")
    data = df[FEATURES].join(target).dropna()
    return data[FEATURES], data["target_close"]


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
