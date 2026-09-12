"""Parquet persistence for daily OHLCV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trendline.config import OHLCV_PATH, PARQUET_DIR

OHLCV_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str)
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["source"] = out["source"].fillna("unknown").astype(str)
    out = out.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    out = out[out["high"] >= out["low"]]
    out = out[out["close"] > 0]
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return out[OHLCV_COLS].reset_index(drop=True)


def save_ohlcv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or OHLCV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _normalize(df)
    clean.to_parquet(path, index=False)
    return path


def load_ohlcv(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or OHLCV_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No OHLCV parquet at {path}. Drop a file there or run: python scripts/fetch.py"
        )
    return _normalize(pd.read_parquet(path))


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"tickers": 0, "rows": 0}
    return {
        "tickers": int(df["ticker"].nunique()),
        "rows": int(len(df)),
        "min_date": str(df["date"].min().date()),
        "max_date": str(df["date"].max().date()),
        "days_median": int(df.groupby("ticker")["date"].nunique().median()),
    }
