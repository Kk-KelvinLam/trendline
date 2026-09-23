"""RTH 5-minute bars for paper-ledger fills (Alpaca first, Yahoo fallback)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trendline.config import ARTIFACT_DIR
from trendline.data.providers import _alpaca_credentials, _alpaca_symbol

NY = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END_BAR = time(15, 55)  # bar that starts 15:55 and closes ~16:00
INTRADAY_CACHE_DIR = ARTIFACT_DIR / "intraday"
_ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


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


def _rth_window_utc(session: date) -> tuple[str, str]:
    """Inclusive RTH window in RFC3339 UTC for Alpaca (handles EST/EDT)."""
    start_ny = datetime.combine(session, RTH_START, tzinfo=NY)
    # one bar past last RTH start so 15:55 bar is included
    end_ny = datetime.combine(session, time(16, 0), tzinfo=NY)
    start = start_ny.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_ny.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


def _bars_from_alpaca_payload(
    bars_map: dict,
    sym_to_orig: dict[str, str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, bars in (bars_map or {}).items():
        orig = sym_to_orig.get(sym, sym)
        if not bars:
            continue
        rows = []
        idx = []
        for b in bars:
            ts = pd.Timestamp(b.get("t"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            idx.append(ts)
            rows.append(
                {
                    "open": b.get("o"),
                    "high": b.get("h"),
                    "low": b.get("l"),
                    "close": b.get("c"),
                    "volume": b.get("v", 0),
                }
            )
        if not rows:
            continue
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="datetime"))
        try:
            out[orig] = _normalize_ohlcv(df)
        except Exception:
            continue
    return out


def fetch_alpaca_rth_5m(
    tickers: list[str],
    session: date,
    *,
    feed: str = "sip",
    chunk: int = 40,
) -> dict[str, pd.DataFrame]:
    """Fetch SIP 5Min bars for a completed US RTH session. Soft-fail.

    Free SIP returns HTTP 403 when ``end`` is too recent — callers should only
    request completed sessions and fall back to Yahoo on empty/403.
    Does not log credential values.
    """
    creds = _alpaca_credentials()
    if not creds or not tickers:
        return {}
    key_id, secret = creds

    want: list[tuple[str, str]] = []
    seen: set[str] = set()
    for t in tickers:
        sym = _alpaca_symbol(t)
        if sym is None or sym in seen:
            continue
        seen.add(sym)
        want.append((t, sym))
    if not want:
        return {}

    start, end = _rth_window_utc(session)
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(want), chunk):
        batch = want[i : i + chunk]
        sym_to_orig = {sym: orig for orig, sym in batch}
        try:
            part = _alpaca_5m_chunk(
                list(sym_to_orig.keys()),
                start=start,
                end=end,
                feed=feed,
                key_id=key_id,
                secret=secret,
                sym_to_orig=sym_to_orig,
            )
        except Exception as exc:
            # HTTP 403 (too-recent / plan) and empty batches → Yahoo fallback
            msg = str(exc)
            if "HTTP 403" in msg or isinstance(exc, urllib.error.HTTPError):
                print(
                    f"[intraday] alpaca 5m HTTP 403/forbidden "
                    f"session={session.isoformat()} batch={i // chunk}; falling back",
                    flush=True,
                )
            else:
                print(
                    f"[intraday] alpaca 5m failed session={session.isoformat()} "
                    f"batch={i // chunk}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            continue
        out.update(part)
    return out


def _alpaca_5m_chunk(
    symbols: list[str],
    *,
    start: str,
    end: str,
    feed: str,
    key_id: str,
    secret: str,
    sym_to_orig: dict[str, str],
) -> dict[str, pd.DataFrame]:
    merged: dict[str, list] = {s: [] for s in symbols}
    page_token: str | None = None
    while True:
        params: dict[str, str | int] = {
            "symbols": ",".join(symbols),
            "timeframe": "5Min",
            "start": start,
            "end": end,
            "adjustment": "split",
            "feed": feed,
            "limit": 10000,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{_ALPACA_BARS_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": key_id,
                "APCA-API-SECRET-KEY": secret,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

        bars_map = payload.get("bars") or {}
        for sym, bars in bars_map.items():
            if sym in merged and bars:
                merged[sym].extend(bars)
        page_token = payload.get("next_page_token") or None
        if not page_token:
            break

    return _bars_from_alpaca_payload(merged, sym_to_orig)


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


def fetch_yahoo_rth_5m(
    tickers: list[str],
    session: date,
) -> dict[str, pd.DataFrame]:
    """Yahoo 5m for ``session`` (calendar day window); soft-fail per ticker."""
    if not tickers:
        return {}
    start = session.isoformat()
    end = (session + timedelta(days=1)).isoformat()
    out: dict[str, pd.DataFrame] = {}
    ysym_map = {t: yahoo_symbol(t) for t in tickers}
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

    for t in tickers:
        ys = ysym_map[t]
        norm = batch_frames.get(ys)
        if norm is None or norm.empty:
            norm = _download_one(ys, start, end)
        if norm is None or norm.empty:
            continue
        out[t] = norm
    return out


def _load_cache(
    tickers: list[str],
    session: date,
    cache_root: Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for t in tickers:
        path = cache_root / session.isoformat() / f"{t}.parquet"
        if not path.exists():
            need.append(t)
            continue
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
    return out, need


def _save_cache(
    ticker: str,
    session: date,
    norm: pd.DataFrame,
    cache_root: Path,
) -> None:
    path = cache_root / session.isoformat() / f"{ticker}.parquet"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        norm.to_parquet(path)
    except Exception:
        pass


def fetch_rth_5m(
    tickers: list[str],
    session_date: str | date | datetime | pd.Timestamp,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    alpaca_fetcher=None,
    yahoo_fetcher=None,
) -> dict[str, pd.DataFrame]:
    """Fetch RTH 5m bars for ``session_date``; fail soft per ticker.

    Priority matches daily OHLCV: Alpaca SIP when ``APCA_*`` keys are set, then
    Yahoo. Tickers with no usable bars are omitted (caller treats as miss).

    Returns ``dict[ticker, DataFrame]`` with columns open/high/low/close
    (and volume when present), indexed in America/New_York.

    ``alpaca_fetcher`` / ``yahoo_fetcher`` are test hooks:
    ``(tickers, session) -> dict[str, DataFrame]`` (pre-RTH-filter OK).
    """
    session = _session_date(session_date)
    if not tickers:
        return {}
    cache_root = Path(cache_dir) if cache_dir is not None else INTRADAY_CACHE_DIR
    uniq = sorted(set(tickers))

    out: dict[str, pd.DataFrame] = {}
    if use_cache:
        cached, need = _load_cache(uniq, session, cache_root)
        out.update(cached)
    else:
        need = list(uniq)

    if not need:
        return out

    remaining = list(need)
    alpaca_fn = alpaca_fetcher if alpaca_fetcher is not None else fetch_alpaca_rth_5m
    yahoo_fn = yahoo_fetcher if yahoo_fetcher is not None else fetch_yahoo_rth_5m

    # --- 1. Alpaca when credentials present (or injected fetcher) ---
    use_alpaca = alpaca_fetcher is not None or _alpaca_credentials() is not None
    if use_alpaca and remaining:
        alpaca_tickers = [t for t in remaining if _alpaca_symbol(t) is not None]
        if alpaca_tickers:
            print(
                f"[intraday] alpaca 5m primary for {len(alpaca_tickers)} tickers "
                f"session={session.isoformat()}",
                flush=True,
            )
            try:
                got = alpaca_fn(alpaca_tickers, session) or {}
            except Exception as exc:
                print(
                    f"[intraday] alpaca 5m error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                got = {}
            n_ok = 0
            for t in alpaca_tickers:
                norm = got.get(t)
                if norm is None or getattr(norm, "empty", True):
                    continue
                rth = filter_rth(norm, session)
                if rth.empty:
                    continue
                out[t] = rth
                n_ok += 1
                if use_cache:
                    _save_cache(t, session, norm, cache_root)
            print(
                f"[intraday] alpaca 5m result: ok={n_ok}/{len(alpaca_tickers)}",
                flush=True,
            )
            remaining = [t for t in remaining if t not in out]
        else:
            print(
                f"[intraday] alpaca 5m skipped (no mappable equities) "
                f"session={session.isoformat()}",
                flush=True,
            )
    elif remaining:
        print(
            "[intraday] alpaca 5m skipped (missing APCA_API_KEY_ID/"
            "APCA_API_SECRET_KEY)",
            flush=True,
        )

    # --- 2. Yahoo for leftovers ---
    if remaining:
        print(
            f"[intraday] yahoo 5m fallback for {len(remaining)} tickers "
            f"session={session.isoformat()}",
            flush=True,
        )
        try:
            got = yahoo_fn(remaining, session) or {}
        except Exception as exc:
            print(
                f"[intraday] yahoo 5m error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            got = {}
        n_ok = 0
        for t in remaining:
            norm = got.get(t)
            if norm is None or getattr(norm, "empty", True):
                continue
            rth = filter_rth(norm, session)
            if rth.empty:
                continue
            out[t] = rth
            n_ok += 1
            if use_cache:
                _save_cache(t, session, norm, cache_root)
        print(
            f"[intraday] yahoo 5m result: ok={n_ok}/{len(remaining)}",
            flush=True,
        )

    return out
