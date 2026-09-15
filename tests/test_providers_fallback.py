"""CombinedProvider must retry tickers behind the panel's newest day."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trendline.data.providers import CombinedProvider


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
    """Batch returns VIX@9/14 + AAPL@9/11; single AAPL returns through 9/14."""

    singles: list = field(default_factory=list)

    def download(self, tickers, start, end=None):
        tickers = list(tickers)
        if tickers == ["AAPL"]:
            self.singles.append("AAPL")
            return _bars("AAPL", ["2026-09-11", "2026-09-14"])
        # batch
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
    out = combined.download(["AAPL", "^VIX"], start="2026-09-10", end="2026-09-16")
    assert yahoo.singles == ["AAPL"]
    aapl = out[out["ticker"] == "AAPL"]
    assert pd.Timestamp("2026-09-14") in set(pd.to_datetime(aapl["date"]).dt.normalize())
    assert stooq.called_with is None  # single retry succeeded


def test_normalize_keeps_row_when_close_nan(monkeypatch):
    import pandas as pd
    from trendline.data import providers as prov

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
    out = prov._normalize_yf_frame(part, "AAPL")
    assert len(out) == 1
    assert out.iloc[0]["close"] == 10.5  # mid high/low
