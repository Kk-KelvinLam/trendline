"""Parquet persistence for daily OHLCV."""

from __future__ import annotations

from datetime import timedelta
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
    """Atomic write: parquet lands via ``.tmp`` then replace so readers never see a half file."""
    path = Path(path or OHLCV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _normalize(df)
    tmp = path.with_suffix(path.suffix + ".tmp")
    clean.to_parquet(tmp, index=False)
    tmp.replace(path)
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


DELTA_LOOKBACK_DAYS = 15


def delta_start(existing: pd.DataFrame, fallback: str, lookback_days: int = DELTA_LOOKBACK_DAYS) -> str:
    """Start date for an incremental fetch. Overlaps recent bars so Yahoo revisions land."""
    if existing is None or existing.empty or "date" not in existing.columns:
        return fallback
    last = pd.to_datetime(existing["date"]).max()
    if pd.isna(last):
        return fallback
    start = (last - timedelta(days=lookback_days)).normalize()
    fb = pd.Timestamp(fallback)
    if start < fb:
        start = fb
    return start.date().isoformat()


def merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """One file, newest bar wins on (ticker, date)."""
    frames = [df for df in (existing, incoming) if df is not None and not df.empty]
    if not frames:
        return _normalize(pd.DataFrame(columns=OHLCV_COLS))
    return _normalize(pd.concat(frames, ignore_index=True))


def equity_session_coverage(
    ohlcv: pd.DataFrame,
    session: pd.Timestamp | str | None = None,
    *,
    expected: pd.Timestamp | str | None = None,
) -> dict:
    """How many S&P names have a finite close on ``session`` (default: max date).

    Used to refuse cards/ledger when Yahoo only published macros / NaN-close stubs.

    ``complete`` requires both coverage (≥90% members) **and** freshness vs the
    expected completed equity session (stale max-date alone is not enough).
    """
    from trendline.universe import load_sp500
    from trendline.data.providers import expected_equity_session

    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    members = set(load_sp500()["ticker"])
    if session is None:
        # Prefer latest date that has any S&P row; else overall max
        sp = df[df["ticker"].isin(members)]
        session_ts = sp["date"].max() if not sp.empty else df["date"].max()
    else:
        session_ts = pd.Timestamp(session).tz_localize(None).normalize()
    day = df[(df["date"] == session_ts) & (df["ticker"].isin(members))]
    n_members = len(members)
    n_have = int(day["ticker"].nunique()) if not day.empty else 0
    frac = (n_have / n_members) if n_members else 0.0
    coverage_ok = frac >= 0.90
    if expected is None:
        expected_ts = expected_equity_session()
    else:
        expected_ts = pd.Timestamp(expected).tz_localize(None).normalize()
    fresh = bool(pd.notna(session_ts) and session_ts >= expected_ts)
    return {
        "session": str(session_ts.date()) if pd.notna(session_ts) else None,
        "expected": str(expected_ts.date()) if pd.notna(expected_ts) else None,
        "n_members": n_members,
        "n_have": n_have,
        "frac": frac,
        "coverage_ok": coverage_ok,
        "fresh": fresh,
        "complete": bool(coverage_ok and fresh),
    }


def full_replace_acceptable(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    *,
    min_ticker_frac: float = 0.85,
    min_depth_frac: float = 0.70,
    min_span_frac: float = 0.70,
) -> tuple[bool, str]:
    """Whether a ``--full`` download may replace the on-disk store.

    Rejects thin / short / membership-poor panels so a partial provider response
    cannot erase history. Empty existing always accepts.
    """
    if existing is None or existing.empty:
        return True, "no existing store"
    if incoming is None or incoming.empty:
        return False, "incoming empty"
    ex = _normalize(existing)
    inc = _normalize(incoming)
    ex_tickers = set(ex["ticker"])
    inc_tickers = set(inc["ticker"])
    if not ex_tickers:
        return True, "existing has no tickers"
    ticker_frac = len(inc_tickers & ex_tickers) / len(ex_tickers)
    if ticker_frac < min_ticker_frac:
        return False, f"ticker coverage {ticker_frac:.1%} < {min_ticker_frac:.0%}"
    ex_depth = float(ex.groupby("ticker")["date"].nunique().median())
    inc_depth = float(inc.groupby("ticker")["date"].nunique().median()) if not inc.empty else 0.0
    if ex_depth > 0 and (inc_depth / ex_depth) < min_depth_frac:
        return False, f"median depth {inc_depth:.0f}/{ex_depth:.0f} < {min_depth_frac:.0%}"
    ex_span = (ex["date"].max() - ex["date"].min()).days
    inc_span = (inc["date"].max() - inc["date"].min()).days if len(inc) else 0
    if ex_span > 0 and (inc_span / ex_span) < min_span_frac:
        return False, f"date span {inc_span}/{ex_span} days < {min_span_frac:.0%}"
    return True, "ok"

