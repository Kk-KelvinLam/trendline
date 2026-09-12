"""S&P 500 membership and sector map. Works fully offline from the CSV snapshot."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from trendline.config import (
    DEFAULT_TRAIN_TICKERS,
    MACRO_TICKERS,
    SECTOR_ETF,
    UNIVERSE_PATH,
)


@lru_cache(maxsize=1)
def load_sp500(path: str | None = None) -> pd.DataFrame:
    """Load the static S&P 500 list shipped in the repo.

    Snapshot date is the ``asof`` column (Wikipedia retrieval date).
    """
    p = path or UNIVERSE_PATH
    df = pd.read_csv(p)
    required = {"ticker", "name", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe file missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def sector_map() -> dict[str, str]:
    return dict(zip(load_sp500()["ticker"], load_sp500()["sector"], strict=False))


def sector_etf_for(ticker: str) -> str | None:
    sec = sector_map().get(ticker)
    if not sec:
        return None
    return SECTOR_ETF.get(sec)


def is_sp500(ticker: str) -> bool:
    return ticker in set(load_sp500()["ticker"])


def default_fetch_tickers(include_macros: bool = True) -> list[str]:
    """Liquid S&P subset used for the default download (plus macros)."""
    members = set(load_sp500()["ticker"])
    names = [t for t in DEFAULT_TRAIN_TICKERS if t in members]
    if include_macros:
        for m in MACRO_TICKERS:
            if m not in names:
                names.append(m)
    return names


def all_member_tickers() -> list[str]:
    return load_sp500()["ticker"].tolist()


def rank_by_dollar_volume(ohlcv: pd.DataFrame, asof: pd.Timestamp | None = None, top_n: int = 100) -> pd.DataFrame:
    """Rank S&P names by prior-session dollar volume (close × volume)."""
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    members = set(load_sp500()["ticker"])
    df = df[df["ticker"].isin(members)]
    if df.empty:
        return df
    if asof is None:
        asof = df["date"].max()
    day = df[df["date"] == pd.Timestamp(asof)].copy()
    day["dollar_volume"] = day["close"] * day["volume"]
    day = day.sort_values("dollar_volume", ascending=False)
    day["dvol_rank"] = range(1, len(day) + 1)
    return day.head(int(top_n)).reset_index(drop=True)
