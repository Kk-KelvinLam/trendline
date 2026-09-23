"""CombinedProvider retry + incomplete Close handling."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trendline.data.providers import CombinedProvider, _normalize_yf_frame
from trendline.data.store import equity_session_coverage


def _bars(ticker: str, dates: list[str], source: str = "yfinance") -> pd.DataFrame:
    rows = []
    for d in dates:
        rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": ticker,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 100,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class _FakeYahoo:
    singles: list = field(default_factory=list)

    def download(self, tickers, start, end=None):
        tickers = list(tickers)
        if tickers == ["AAPL"]:
            self.singles.append("AAPL")
            return _bars("AAPL", ["2026-09-11", "2026-09-14"])
        return pd.concat(
            [_bars("^VIX", ["2026-09-11", "2026-09-14"]), _bars("AAPL", ["2026-09-11"])],
            ignore_index=True,
        )


@dataclass
class _FakeStooq:
    called_with: list | None = None

    def download(self, tickers, start, end=None):
        self.called_with = list(tickers)
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        )


def test_single_yahoo_retry_fills_behind_panel_max():
    yahoo = _FakeYahoo()
    stooq = _FakeStooq()
    combined = CombinedProvider(yahoo=yahoo, stooq=stooq)  # type: ignore[arg-type]
    out = combined.download(["AAPL", "^VIX"], start="2026-09-10", end="2026-09-15")
    assert yahoo.singles == ["AAPL"]
    aapl = out[out["ticker"] == "AAPL"]
    assert pd.Timestamp("2026-09-14") in set(pd.to_datetime(aapl["date"]).dt.normalize())
    assert stooq.called_with is None


def test_normalize_drops_row_when_close_and_adj_nan():
    part = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-14")],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [float("nan")],
            "adj_close": [float("nan")],
            "volume": [1000],
        }
    )
    out = _normalize_yf_frame(part, "AAPL")
    assert out.empty


def test_normalize_uses_adj_close_when_close_nan():
    part = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-14")],
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [float("nan")],
            "adj_close": [11.0],
            "volume": [1000],
        }
    )
    out = _normalize_yf_frame(part, "AAPL")
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 11.0


def test_equity_session_coverage_incomplete():
    # Only one real S&P name on the day → well under 90% of the Wikipedia list.
    ohlcv = pd.concat(
        [
            _bars("AAPL", ["2026-09-14"]),
            _bars("^VIX", ["2026-09-14"]),
        ],
        ignore_index=True,
    )
    cov = equity_session_coverage(ohlcv, "2026-09-14")
    assert cov["n_have"] == 1
    assert cov["frac"] < 0.90
    assert cov["complete"] is False


def test_expected_equity_session_end_exclusive_skips_weekend():
    from trendline.data.providers import expected_equity_session

    # end Monday → last includable Sunday → roll to Friday
    assert expected_equity_session("2026-09-14").date().isoformat() == "2026-09-11"
    # end Tuesday → Monday
    assert expected_equity_session("2026-09-15").date().isoformat() == "2026-09-14"


def test_uniform_stale_panel_triggers_stooq():
    """Everyone stuck on the same old day must not count as caught up."""

    @dataclass
    class YahooAllOld:
        singles: list = field(default_factory=list)

        def download(self, tickers, start, end=None):
            tickers = list(tickers)
            if len(tickers) == 1:
                self.singles.append(tickers[0])
                # retry still cannot reach expected session
                return _bars(tickers[0], ["2026-09-10", "2026-09-11"])
            return pd.concat(
                [_bars(t, ["2026-09-10", "2026-09-11"]) for t in tickers],
                ignore_index=True,
            )

    @dataclass
    class StooqFill:
        called_with: list | None = None

        def download(self, tickers, start, end=None):
            self.called_with = list(tickers)
            return pd.concat(
                [_bars(t, ["2026-09-11", "2026-09-14"], source="stooq") for t in tickers],
                ignore_index=True,
            )

    yahoo = YahooAllOld()
    stooq = StooqFill()
    combined = CombinedProvider(yahoo=yahoo, stooq=stooq)  # type: ignore[arg-type]
    # end=2026-09-15 → expected 2026-09-14; panel max 2026-09-11 → stale
    out = combined.download(["AAPL", "MSFT"], start="2026-09-01", end="2026-09-15")
    assert stooq.called_with == ["AAPL", "MSFT"]
    assert set(yahoo.singles) == {"AAPL", "MSFT"}
    for t in ("AAPL", "MSFT"):
        days = set(pd.to_datetime(out.loc[out["ticker"] == t, "date"]).dt.normalize())
        assert pd.Timestamp("2026-09-14") in days


def test_classify_yahoo_gap_buckets():
    from trendline.data.providers import _classify_yahoo_gap

    y = pd.concat(
        [
            _bars("^VIX", ["2026-09-11", "2026-09-14"]),
            _bars("AAPL", ["2026-09-11"]),
        ],
        ignore_index=True,
    )
    b = _classify_yahoo_gap(
        ["AAPL", "^VIX", "MSFT"],
        y,
        expected=pd.Timestamp("2026-09-14"),
    )
    assert b["ok"] == ["^VIX"]
    assert b["behind_expected"] == ["AAPL"]
    assert b["absent"] == ["MSFT"]
    assert b["panel_stale"] is False
