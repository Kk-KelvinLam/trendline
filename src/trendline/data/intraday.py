"""RTH 5-minute bars for paper-ledger fills (Yahoo; soft-fail per ticker)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trendline.config import ARTIFACT_DIR

NY = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END_BAR = time(15, 55)  # bar that starts 15:55 and closes ~16:00
INTRADAY_CACHE_DIR = ARTIFACT_DIR / "intraday"


def yahoo_symbol(ticker: str) -> str:
    """Map share-class dots to Yahoo dashes (BRK.B → BRK-B)."""
    if ticker.startswith("^"):
        return ticker
    return ticker.replace(".", "-")


def _session_date(session_date: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(session_date, date) and not isinstance(session_date, datetime):
        return session_date
    return pd.Timestamp(session_date).date()


def _cache_path(ticker: str, session: date) -> Path:
    return INTRADAY_CACHE_DIR / session.isoformat() / f"{ticker}.parquet"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = str(c).lower().replace(" ", "")
        if cl in ("open", "high", "low", "close", "volume"):
            colmap[c] = cl
        elif cl == "adjclose":
            colmap[c] = "close"
    df = df.rename(columns=colmap)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(subset=["open", "high", "low", "close"], how="any")
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(NY)
    return df.sort_index()


def filter_rth(df: pd.DataFrame, session: date) -> pd.DataFrame:
    """Keep America/New_York RTH bars with start in [09:30, 15:55] inclusive."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    day = df[df.index.date == session]
    if day.empty:
        return day
    times = day.index.time
    mask = [(t >= RTH_START and t <= RTH_END_BAR) for t in times]
    return day.loc[mask]


def _download_one(ys: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        raw = yf.download(
            ys,
            start=start,
            end=end,
            interval="5m",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=False,
            timeout=30,
        )
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    try:
        return _normalize_ohlcv(raw)
    except Exception:
        return None


def fetch_rth_5m(
    tickers: list[str],
    session_date: str | date | datetime | pd.Timestamp,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch RTH 5m bars for ``session_date``; fail soft per ticker.

    Returns ``dict[ticker, DataFrame]`` with columns open/high/low/close
    (and volume when present), indexed in America/New_York. Tickers with no
    usable bars are omitted (caller falls back to daily OHLC fill).
    """
    session = _session_date(session_date)
    if not tickers:
        return {}
    cache_root = Path(cache_dir) if cache_dir is not None else INTRADAY_CACHE_DIR
    start = session.isoformat()
    end = (session + timedelta(days=1)).isoformat()

    out: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for t in sorted(set(tickers)):
        if use_cache:
            path = cache_root / session.isoformat() / f"{t}.parquet"
            if path.exists():
                try:
                    cached = pd.read_parquet(path)
                    if getattr(cached.index, "tz", None) is None and "datetime" in cached.columns:
                        cached = cached.set_index("datetime")
                    if getattr(cached.index, "tz", None) is None:
                        cached.index = pd.to_datetime(cached.index).tz_localize(NY)
                    else:
                        cached.index = cached.index.tz_convert(NY)
                    rth = filter_rth(cached, session)
                    if not rth.empty:
                        out[t] = rth
                        continue
                except Exception:
                    pass
        need.append(t)

    # Batch then per-ticker retry for misses
    ysym_map = {t: yahoo_symbol(t) for t in need}
    unique_ys = sorted(set(ysym_map.values()))
    batch_frames: dict[str, pd.DataFrame] = {}
    if unique_ys:
        try:
            import yfinance as yf

            raw = yf.download(
                tickers=unique_ys,
                start=start,
                end=end,
                interval="5m",
                group_by="ticker",
                auto_adjust=True,
                prepost=False,
                threads=True,
                progress=False,
                timeout=30,
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    level0 = set(raw.columns.get_level_values(0))
                    for ys in unique_ys:
                        if ys not in level0:
                            continue
                        sub = raw[ys]
                        if sub is None or sub.dropna(how="all").empty:
                            continue
                        batch_frames[ys] = _normalize_ohlcv(sub)
                elif len(unique_ys) == 1:
                    batch_frames[unique_ys[0]] = _normalize_ohlcv(raw)
        except Exception:
            batch_frames = {}

    for t in need:
        ys = ysym_map[t]
        norm = batch_frames.get(ys)
        if norm is None or norm.empty:
            norm = _download_one(ys, start, end)
        if norm is None or norm.empty:
            continue
        rth = filter_rth(norm, session)
        if rth.empty:
            continue
        out[t] = rth
        if use_cache:
            path = cache_root / session.isoformat() / f"{t}.parquet"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # store full-day normalized (pre-filter) for reuse; re-filter on load
                to_save = norm.copy()
                to_save.to_parquet(path)
            except Exception:
                pass
    return out
