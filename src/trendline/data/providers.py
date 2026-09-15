"""Price providers. No paid API key required.

Interface is provider-agnostic so a paid vendor can be swapped in later.
Yahoo Finance is tried first; Stooq is the fallback per ticker.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd
import requests

from trendline.data.store import OHLCV_COLS

_UA = {"User-Agent": "trendline/0.1 (research; +https://github.com/Kk-KelvinLam/trendline)"}


class DataProvider(Protocol):
    name: str

    def download(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return daily bars with columns date,ticker,open,high,low,close,adj_close,volume."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLS)


def _yf_symbol(ticker: str) -> str:
    # Yahoo uses '-' for share classes (BRK-B). Keep carets for indices.
    return ticker


def _stooq_symbol(ticker: str) -> str:
    if ticker.startswith("^"):
        return ticker.lower()
    return f"{ticker.replace('-', '.')}.us".lower()


class YahooFinanceProvider:
    name = "yfinance"

    def download(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed") from exc

        if not tickers:
            return _empty()

        frames: list[pd.DataFrame] = []
        # Small batches + pause: Actions IPs hit Yahoo 429 hard on wide pulls.
        batch_size = 8
        for i in range(0, len(tickers), batch_size):
            batch = [_yf_symbol(t) for t in tickers[i : i + batch_size]]
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=30,
            )
            frames.append(self._flatten(raw, batch))
            if i + batch_size < len(tickers):
                time.sleep(1.5)
        if not frames:
            return _empty()
        out = pd.concat(frames, ignore_index=True)
        out["source"] = "yfinance"
        return out

    @staticmethod
    def _flatten(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return _empty()
        rows: list[pd.DataFrame] = []
        # Single ticker: columns are OHLCV (no ticker level)
        if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
            part = raw.reset_index()
            part = _normalize_yf_frame(part, tickers[0])
            if not part.empty:
                rows.append(part)
            return pd.concat(rows, ignore_index=True) if rows else _empty()

        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            level0 = set(cols.get_level_values(0))
            level1 = set(cols.get_level_values(1))
            # yfinance 1.x sometimes uses ticker as level 0, field as level 1
            if level0 & set(tickers) or any(t in level0 for t in tickers):
                for t in tickers:
                    if t not in level0:
                        continue
                    part = raw[t].copy()
                    part = part.reset_index()
                    part = _normalize_yf_frame(part, t)
                    if not part.empty:
                        rows.append(part)
            elif level1 & set(tickers):
                for t in tickers:
                    if t not in level1:
                        continue
                    part = raw.xs(t, axis=1, level=1).copy()
                    part = part.reset_index()
                    part = _normalize_yf_frame(part, t)
                    if not part.empty:
                        rows.append(part)
        return pd.concat(rows, ignore_index=True) if rows else _empty()


def _normalize_yf_frame(part: pd.DataFrame, ticker: str) -> pd.DataFrame:
    part.columns = [str(c).strip().lower().replace(" ", "_") for c in part.columns]
    date_col = "date" if "date" in part.columns else ("datetime" if "datetime" in part.columns else None)
    if date_col is None:
        return _empty()
    rename = {}
    if "adj_close" not in part.columns and "adjclose" in part.columns:
        rename["adjclose"] = "adj_close"
    part = part.rename(columns=rename)
    if "adj_close" not in part.columns:
        part["adj_close"] = part.get("close")
    need = ["open", "high", "low", "close", "volume"]
    if any(c not in part.columns for c in need):
        return _empty()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(part[date_col], utc=True).dt.tz_localize(None),
            "ticker": ticker,
            "open": part["open"],
            "high": part["high"],
            "low": part["low"],
            "close": part["close"],
            "adj_close": part["adj_close"],
            "volume": part["volume"],
            "source": "yfinance",
        }
    )
    # Incomplete Yahoo rows (Close still NaN) are dropped — do not invent Close from
    # mid/open. Prefer Adj Close when Yahoo published that but left Close empty.
    # Nightly skips cards/ledger when the newest equity session is too thin; retry later.
    out["close"] = out["close"].fillna(out["adj_close"])
    out["adj_close"] = out["adj_close"].fillna(out["close"])
    return out.dropna(subset=["open", "high", "low", "close"])


class StooqProvider:
    """Daily bars from Stooq CSV (no key). US equities use TICKER.us; VIX is ^vix."""

    name = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def __init__(self, pause_s: float = 0.25) -> None:
        self.pause_s = pause_s

    def download(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        frames: list[pd.DataFrame] = []
        for t in tickers:
            try:
                part = self._one(_stooq_symbol(t), t)
            except Exception:
                part = _empty()
            if not part.empty:
                part = part[(part["date"] >= start_ts) & (part["date"] <= end_ts)]
                frames.append(part)
            time.sleep(self.pause_s)
        return pd.concat(frames, ignore_index=True) if frames else _empty()

    def _one(self, symbol: str, ticker: str) -> pd.DataFrame:
        url = f"{self.BASE}?s={symbol}&i=d"
        r = requests.get(url, headers=_UA, timeout=20)
        r.raise_for_status()
        text = r.text.strip()
        if not text or text.lower().startswith("<!") or "no data" in text.lower():
            return _empty()
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip().lower() for c in raw.columns]
        if "date" not in raw.columns or "close" not in raw.columns:
            return _empty()
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["date"]),
                "ticker": ticker,
                "open": raw.get("open"),
                "high": raw.get("high"),
                "low": raw.get("low"),
                "close": raw["close"],
                "adj_close": raw["close"],
                "volume": raw.get("volume", 0),
                "source": "stooq",
            }
        )
        return out.dropna(subset=["open", "high", "low", "close"])


@dataclass
class CombinedProvider:
    """Yahoo first, fill missing tickers from Stooq. Never invents prices."""

    name: str = "combined"
    yahoo: YahooFinanceProvider = field(default_factory=YahooFinanceProvider)
    stooq: StooqProvider = field(default_factory=StooqProvider)

    def download(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        """Yahoo first; Stooq for tickers missing entirely *or* missing the end session.

        Yahoo rate limits often return a partial panel (e.g. only ^VIX on the
        newest day). Treating any historical rows as success skipped Stooq and
        left equities stuck on an older asof.
        """
        frames: list[pd.DataFrame] = []
        y = _empty()
        try:
            y = self.yahoo.download(tickers, start=start, end=end)
            if not y.empty:
                frames.append(y)
        except Exception as exc:
            print(f"[combined] yfinance failed: {exc}")

        end_ts = pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize()
        # yfinance end is exclusive-ish; treat "as of end-1 calendar day" as OK when end is tomorrow
        # Callers pass end=None or next-day ISO; require max(date) >= start and cover latest requested day when end set.
        covered: set[str] = set()
        if not y.empty:
            y2 = y.copy()
            y2["date"] = pd.to_datetime(y2["date"]).dt.tz_localize(None).dt.normalize()
            # When end is given as exclusive next day (fetch.py), last session is end-1 trading day;
            # use max date across download and require each ticker to reach the panel max.
            panel_max = y2["date"].max()
            for t, g in y2.groupby("ticker"):
                if g["date"].max() >= panel_max:
                    covered.add(t)

        missing = [t for t in tickers if t not in covered]
        if missing:
            target_max = None
            if not y.empty:
                target_max = (
                    pd.to_datetime(y["date"]).dt.tz_localize(None).dt.normalize().max()
                )
            print(
                f"[combined] single-ticker Yahoo retry for {len(missing)} "
                f"(missing or behind panel max={target_max.date() if target_max is not None else 'n/a'})"
            )
            retry_frames: list[pd.DataFrame] = []
            still: list[str] = []
            for t in missing:
                try:
                    one = self.yahoo.download([t], start=start, end=end)
                except Exception:
                    one = _empty()
                if one is None or one.empty:
                    still.append(t)
                    time.sleep(0.35)
                    continue
                one = one.copy()
                one["date"] = pd.to_datetime(one["date"]).dt.tz_localize(None).dt.normalize()
                tmax = one["date"].max()
                if target_max is not None and tmax < target_max:
                    still.append(t)
                else:
                    retry_frames.append(one)
                time.sleep(0.35)
            if retry_frames:
                frames.append(pd.concat(retry_frames, ignore_index=True))
            if still:
                print(f"[combined] stooq fallback for {len(still)} tickers")
                try:
                    s = self.stooq.download(still, start=start, end=end)
                    if not s.empty:
                        frames.append(s)
                except Exception as exc:
                    print(f"[combined] stooq failed: {exc}")

        if not frames:
            return _empty()
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        return out.reset_index(drop=True)
