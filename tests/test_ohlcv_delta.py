import pandas as pd

from trendline.data.store import delta_start, merge_ohlcv


def _bar(date, ticker="AAA", close=10.0):
    return {
        "date": date,
        "ticker": ticker,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 1000,
        "source": "yfinance",
    }


def test_delta_start_overlaps():
    existing = pd.DataFrame([_bar("2026-09-11")])
    assert delta_start(existing, "2023-01-01") == "2026-08-27"


def test_delta_start_empty_uses_fallback():
    assert delta_start(pd.DataFrame(), "2023-01-01") == "2023-01-01"


def test_merge_keeps_one_file_newest_wins():
    old = pd.DataFrame([_bar("2026-09-10", close=10.0), _bar("2026-09-11", close=11.0)])
    new = pd.DataFrame([_bar("2026-09-11", close=11.5), _bar("2026-09-12", close=12.0)])
    out = merge_ohlcv(old, new)
    assert list(out["date"].dt.date.astype(str)) == ["2026-09-10", "2026-09-11", "2026-09-12"]
    assert float(out.loc[out["date"] == "2026-09-11", "close"].iloc[0]) == 11.5
