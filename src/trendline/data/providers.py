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

# Browser-like UA (was trendline/0.1 bot string). Used for Stooq requests and
# the yfinance session so Yahoo/Stooq see the same client label.
_UA_STRING = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) "
    "Gecko/20100101 Firefox/133.0"
)
_UA = {
    "User-Agent": _UA_STRING,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _http_session() -> requests.Session:
    """Shared requests session for Stooq; also passed into yfinance when possible."""
    s = requests.Session()
    s.headers.update(_UA)
    return s


def _yf_session():
    """Prefer yfinance's backend session (curl_cffi when available), with our UA."""
    try:
        from yfinance import _http as yf_http

        s = yf_http.new_session()
        s.headers.update(_UA)
        return s
    except Exception:
        return _http_session()


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
        session = _yf_session()
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
                session=session,
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


def _as_naive_day(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(None).normalize()


def expected_equity_session(end: str | None = None) -> pd.Timestamp:
    """Last weekday Yahoo should already have published for a post-close nightly.

    ``yfinance`` ``end`` is exclusive → last includable calendar day is end-1.
    With ``end=None`` (nightly fetch), use **yesterday** rolled to a weekday, not
    today — Actions runs Tue–Sat morning HKT after the prior US cash session.
    US holidays are not calendared; a holiday may briefly look "stale" and
    trigger Stooq (usually empty / harmless).
    """
    if end:
        d = _as_naive_day(end) - pd.Timedelta(days=1)
    else:
        d = _as_naive_day(pd.Timestamp.today()) - pd.Timedelta(days=1)
    while int(d.weekday()) >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _classify_yahoo_gap(
    tickers: list[str],
    y: pd.DataFrame,
    *,
    expected: pd.Timestamp,
) -> dict[str, list[str]]:
    """Bucket tickers after the Yahoo batch (before single-ticker retry)."""
    absent: list[str] = []
    behind_expected: list[str] = []
    behind_panel: list[str] = []
    ok: list[str] = []
    if y is None or y.empty:
        return {
            "ok": [],
            "absent": list(tickers),
            "behind_expected": [],
            "behind_panel": [],
            "panel_max": None,
            "panel_stale": True,
        }
    y2 = y.copy()
    y2["date"] = pd.to_datetime(y2["date"]).dt.tz_localize(None).dt.normalize()
    per_max = y2.groupby("ticker")["date"].max()
    panel_max = per_max.max()
    panel_stale = bool(panel_max < expected)
    for t in tickers:
        if t not in per_max.index:
            absent.append(t)
            continue
        tmax = per_max.loc[t]
        if tmax < expected:
            behind_expected.append(t)
        elif tmax < panel_max:
            behind_panel.append(t)
        else:
            ok.append(t)
    return {
        "ok": ok,
        "absent": absent,
        "behind_expected": behind_expected,
        "behind_panel": behind_panel,
        "panel_max": panel_max,
        "panel_stale": panel_stale,
    }


@dataclass
class CombinedProvider:
    """Yahoo first, fill missing tickers from Stooq. Never invents prices."""

    name: str = "combined"
    yahoo: YahooFinanceProvider = field(default_factory=YahooFinanceProvider)
    stooq: StooqProvider = field(default_factory=StooqProvider)

    def download(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        """Yahoo first; Stooq for tickers still short of the expected session.

        A panel where *every* name stops on the same old day used to look "caught
        up" to panel max and skipped Stooq — that is treated as stale vs
        ``expected_equity_session`` and retried / fallen back.
        """
        frames: list[pd.DataFrame] = []
        y = _empty()
        yahoo_batch_error: str | None = None
        try:
            y = self.yahoo.download(tickers, start=start, end=end)
            if not y.empty:
                frames.append(y)
        except Exception as exc:
            yahoo_batch_error = f"{type(exc).__name__}: {exc}"
            print(f"[combined] yfinance batch failed: {yahoo_batch_error}", flush=True)

        expected = expected_equity_session(end)
        buckets = _classify_yahoo_gap(tickers, y, expected=expected)
        panel_max = buckets["panel_max"]
        n = len(tickers)
        n_ok = len(buckets["ok"])
        n_absent = len(buckets["absent"])
        n_behind_exp = len(buckets["behind_expected"])
        n_behind_panel = len(buckets["behind_panel"])
        print(
            f"[combined] yahoo batch: ok={n_ok}/{n} "
            f"absent={n_absent} behind_expected={n_behind_exp} behind_panel={n_behind_panel} "
            f"expected_session={expected.date()} "
            f"panel_max={panel_max.date() if panel_max is not None else 'n/a'} "
            f"panel_stale={buckets['panel_stale']}",
            flush=True,
        )
        if yahoo_batch_error:
            print("[combined] diagnose: yahoo_batch_exception", flush=True)
        elif buckets["panel_stale"]:
            print(
                "[combined] diagnose: panel_stale_vs_expected "
                "(panel_max < expected_session) — will retry/Stooq all names",
                flush=True,
            )
        elif n_absent + n_behind_exp + n_behind_panel >= max(1, int(0.5 * n)):
            # Mass gaps without an exception often mean Actions IP rate limits.
            print(
                f"[combined] diagnose: suspect_rate_limit_or_partial_panel "
                f"(gap={n_absent + n_behind_exp + n_behind_panel}/{n})",
                flush=True,
            )
        elif n_behind_exp and not n_absent and not buckets["panel_stale"]:
            print(
                f"[combined] diagnose: some_tickers_behind_expected_session",
                flush=True,
            )
        elif n_absent and not n_behind_exp:
            print(
                f"[combined] diagnose: some_tickers_absent_from_yahoo",
                flush=True,
            )

        # Uniform stale: even "ok" vs panel_max are short of expected → retry all.
        if buckets["panel_stale"]:
            need = list(tickers)
        else:
            need = buckets["absent"] + buckets["behind_expected"] + buckets["behind_panel"]

        if need:
            target = expected if buckets["panel_stale"] or panel_max is None else max(expected, panel_max)
            print(
                f"[combined] single-ticker Yahoo retry for {len(need)} "
                f"(target_session>={target.date()})",
                flush=True,
            )
            retry_frames: list[pd.DataFrame] = []
            still: list[str] = []
            retry_ok = retry_empty = retry_still_stale = retry_error = 0
            for t in need:
                try:
                    one = self.yahoo.download([t], start=start, end=end)
                except Exception as exc:
                    retry_error += 1
                    still.append(t)
                    if retry_error <= 3:
                        print(f"[combined] yahoo retry error {t}: {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(0.35)
                    continue
                if one is None or one.empty:
                    retry_empty += 1
                    still.append(t)
                    time.sleep(0.35)
                    continue
                one = one.copy()
                one["date"] = pd.to_datetime(one["date"]).dt.tz_localize(None).dt.normalize()
                tmax = one["date"].max()
                if tmax < target:
                    retry_still_stale += 1
                    still.append(t)
                else:
                    retry_ok += 1
                    retry_frames.append(one)
                time.sleep(0.35)
            print(
                f"[combined] yahoo retry result: ok={retry_ok} empty={retry_empty} "
                f"still_stale={retry_still_stale} error={retry_error} → stooq_candidates={len(still)}",
                flush=True,
            )
            if retry_frames:
                frames.append(pd.concat(retry_frames, ignore_index=True))
            if still:
                print(f"[combined] stooq fallback for {len(still)} tickers", flush=True)
                try:
                    s = self.stooq.download(still, start=start, end=end)
                    if not s.empty:
                        s2 = s.copy()
                        s2["date"] = pd.to_datetime(s2["date"]).dt.tz_localize(None).dt.normalize()
                        got = set(s2.loc[s2["date"] >= target, "ticker"])
                        print(
                            f"[combined] stooq result: rows={len(s2)} "
                            f"tickers_reaching_target={len(got)}/{len(still)} "
                            f"target_session>={target.date()}",
                            flush=True,
                        )
                        frames.append(s)
                    else:
                        print("[combined] stooq result: empty", flush=True)
                except Exception as exc:
                    print(f"[combined] stooq failed: {type(exc).__name__}: {exc}", flush=True)

        if not frames:
            return _empty()
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        return out.reset_index(drop=True)

