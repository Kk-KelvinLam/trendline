"""Alpaca-first RTH 5m fetch with Yahoo fallback (no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from trendline.data import intraday as intraday_mod
from trendline.data.intraday import fetch_rth_5m, filter_rth


def _rth_frame(session: str, ticker_px: float = 100.0) -> pd.DataFrame:
    """Three RTH 5m bars in America/New_York."""
    idx = pd.DatetimeIndex(
        [
            f"{session} 09:30:00",
            f"{session} 09:35:00",
            f"{session} 15:55:00",
        ],
        tz="America/New_York",
    )
    px = float(ticker_px)
    return pd.DataFrame(
        {
            "open": [px, px, px],
            "high": [px + 1, px + 1, px + 0.5],
            "low": [px - 1, px - 1, px - 0.5],
            "close": [px, px + 0.2, px - 0.1],
            "volume": [1000, 1100, 900],
        },
        index=idx,
    )


def test_alpaca_success_skips_yahoo(monkeypatch, tmp_path):
    session = date(2026, 9, 22)
    alpaca_calls: list[list[str]] = []
    yahoo_calls: list[list[str]] = []

    def alpaca_ok(tickers, sess):
        alpaca_calls.append(list(tickers))
        assert sess == session
        return {t: _rth_frame(session.isoformat(), 100 + i) for i, t in enumerate(tickers)}

    def yahoo_spy(tickers, sess):
        yahoo_calls.append(list(tickers))
        return {t: _rth_frame(session.isoformat(), 50) for t in tickers}

    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")

    out = fetch_rth_5m(
        ["AAPL", "BRK-B"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        alpaca_fetcher=alpaca_ok,
        yahoo_fetcher=yahoo_spy,
    )
    assert set(out) == {"AAPL", "BRK-B"}
    assert alpaca_calls == [["AAPL", "BRK-B"]]
    assert yahoo_calls == []
    assert not filter_rth(out["AAPL"], session).empty
    # BRK-B maps to Alpaca BRK.B via providers helper; still keyed by Yahoo ticker
    assert float(out["BRK-B"]["close"].iloc[0]) == pytest.approx(101.0)


def test_alpaca_empty_falls_back_to_yahoo(monkeypatch, tmp_path):
    session = date(2026, 9, 22)
    yahoo_calls: list[list[str]] = []

    def alpaca_empty(tickers, sess):
        return {}

    def yahoo_ok(tickers, sess):
        yahoo_calls.append(list(tickers))
        return {t: _rth_frame(session.isoformat()) for t in tickers}

    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")

    out = fetch_rth_5m(
        ["AAPL", "MSFT"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        alpaca_fetcher=alpaca_empty,
        yahoo_fetcher=yahoo_ok,
    )
    assert set(out) == {"AAPL", "MSFT"}
    assert yahoo_calls == [["AAPL", "MSFT"]]


def test_alpaca_partial_then_yahoo_for_rest(monkeypatch, tmp_path):
    session = date(2026, 9, 22)
    yahoo_calls: list[list[str]] = []

    def alpaca_partial(tickers, sess):
        return {"AAPL": _rth_frame(session.isoformat(), 150)}

    def yahoo_ok(tickers, sess):
        yahoo_calls.append(list(tickers))
        return {t: _rth_frame(session.isoformat(), 200) for t in tickers}

    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")

    out = fetch_rth_5m(
        ["AAPL", "MSFT"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        alpaca_fetcher=alpaca_partial,
        yahoo_fetcher=yahoo_ok,
    )
    assert set(out) == {"AAPL", "MSFT"}
    assert yahoo_calls == [["MSFT"]]
    assert float(out["AAPL"]["close"].iloc[0]) == pytest.approx(150.0)
    assert float(out["MSFT"]["close"].iloc[0]) == pytest.approx(200.0)


def test_no_keys_yahoo_only(monkeypatch, tmp_path):
    session = date(2026, 9, 22)
    yahoo_calls: list[list[str]] = []
    alpaca_calls: list[list[str]] = []

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    def alpaca_should_not_run(tickers, sess):
        alpaca_calls.append(list(tickers))
        raise AssertionError("alpaca should not run without keys")

    def yahoo_ok(tickers, sess):
        yahoo_calls.append(list(tickers))
        return {t: _rth_frame(session.isoformat()) for t in tickers}

    # Do not inject alpaca_fetcher — without keys, path must skip Alpaca entirely.
    out = fetch_rth_5m(
        ["AAPL"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        yahoo_fetcher=yahoo_ok,
    )
    assert set(out) == {"AAPL"}
    assert alpaca_calls == []
    assert yahoo_calls == [["AAPL"]]


def test_index_skipped_by_alpaca_goes_yahoo(monkeypatch, tmp_path):
    session = date(2026, 9, 22)
    yahoo_calls: list[list[str]] = []

    def alpaca_ok(tickers, sess):
        # Should only receive equities
        assert all(not t.startswith("^") for t in tickers)
        return {t: _rth_frame(session.isoformat()) for t in tickers}

    def yahoo_ok(tickers, sess):
        yahoo_calls.append(list(tickers))
        return {t: _rth_frame(session.isoformat()) for t in tickers}

    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")

    out = fetch_rth_5m(
        ["AAPL", "^VIX"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        alpaca_fetcher=alpaca_ok,
        yahoo_fetcher=yahoo_ok,
    )
    assert "AAPL" in out
    assert yahoo_calls == [["^VIX"]]
    assert "^VIX" in out


def test_rth_window_utc_edt():
    # 2026-09-22 is EDT (UTC-4): 09:30 → 13:30Z, 16:00 → 20:00Z
    start, end = intraday_mod._rth_window_utc(date(2026, 9, 22))
    assert start == "2026-09-22T13:30:00Z"
    assert end == "2026-09-22T20:00:00Z"


def test_bars_from_alpaca_payload_maps_brk():
    payload = {
        "BRK.B": [
            {
                "t": "2026-09-22T13:30:00Z",
                "o": 500.0,
                "h": 501.0,
                "l": 499.0,
                "c": 500.5,
                "v": 100,
            }
        ]
    }
    out = intraday_mod._bars_from_alpaca_payload(payload, {"BRK.B": "BRK-B"})
    assert "BRK-B" in out
    rth = filter_rth(out["BRK-B"], date(2026, 9, 22))
    assert len(rth) == 1
    assert float(rth["close"].iloc[0]) == pytest.approx(500.5)
