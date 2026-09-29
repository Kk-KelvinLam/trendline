"""Regression tests for external-review correctness fixes."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trendline.backtest import _expanding_beats_range, simulate_trades
from trendline.data.intraday import (
    is_complete_rth,
    rth_session_status,
    fetch_rth_5m,
)
from trendline.data.providers import AlpacaProvider
from trendline.data.store import (
    full_replace_acceptable,
    equity_session_coverage,
    save_ohlcv,
    merge_ohlcv,
)
from trendline.ledger import (
    _normalize_fx_close,
    _refresh_fx,
    _bar_on_session,
    _resolve_exchange_session,
)
from trendline.metrics import max_drawdown, trade_stats
from trendline.range_touch import fill_fade_bars


def _bar(date, ticker="AAA", close=10.0, **extra):
    row = {
        "date": date,
        "ticker": ticker,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "adj_close": close,
        "volume": 1000,
        "source": "yfinance",
    }
    row.update(extra)
    return row


# --- 1. Look-ahead bias ---


def _oos_row(d, ticker, y_high, y_low, pred_h=0.0, pred_l=0.0, base_h=0.01, base_l=-0.01):
    return {
        "date": pd.Timestamp(d),
        "ticker": ticker,
        "close": 100.0,
        "atr": 2.0,
        "y_high": y_high,
        "y_low": y_low,
        "y_close": 0.0,
        "pred_high_q50": pred_h,
        "pred_low_q50": pred_l,
        "pred_high_q10": pred_h - 0.01,
        "pred_high_q90": pred_h + 0.01,
        "pred_low_q10": pred_l - 0.01,
        "pred_low_q90": pred_l + 0.01,
        "pred_close_q50": 0.0,
        "pred_close_q10": -0.01,
        "pred_close_q90": 0.01,
        "base_high_q50": base_h,
        "base_low_q50": base_l,
        "base_close_q50": 0.0,
        "next_open": 100.0,
        "next_high": 102.0,
        "next_low": 97.0,
        "next_close": 100.0,
        "next_date": pd.Timestamp(d) + pd.Timedelta(days=1),
        "fold_id": 0,
    }


def test_expanding_beats_ignores_same_day_outcomes():
    """History for date D must not include D's own labels (look-ahead)."""
    rows = []
    # Early days: model worse than baseline
    for d in pd.bdate_range("2024-01-02", periods=50):
        rows.append(_oos_row(d, "AAA", y_high=0.05, y_low=-0.05, pred_h=0.0, pred_l=0.0, base_h=0.01, base_l=-0.01))
    # Later days: model suddenly perfect — must not leak into early eligibility
    for d in pd.bdate_range("2024-03-15", periods=50):
        rows.append(_oos_row(d, "AAA", y_high=0.01, y_low=-0.01, pred_h=0.01, pred_l=-0.01, base_h=0.05, base_l=-0.05))
    oos = pd.DataFrame(rows)
    # Before the perfect window, expanding gate should be False
    early = pd.Timestamp("2024-03-14")
    hist = oos[oos["date"] < early]
    beat, overall = _expanding_beats_range(hist, "shared", min_overall=20, min_ticker=5)
    assert overall is False
    # After perfect labels accumulate, gate can flip
    late = pd.Timestamp("2024-05-01")
    hist2 = oos[oos["date"] < late]
    _, overall2 = _expanding_beats_range(hist2, "shared", min_overall=20, min_ticker=5)
    assert overall2 is True


def test_simulate_trades_does_not_use_future_beats():
    """With only future-perfect labels, early dates must not trade on leaked gate."""
    rows = []
    for i, d in enumerate(pd.bdate_range("2024-01-02", periods=30)):
        # Predictions that would trade if allowed (large fade room)
        rows.append(
            _oos_row(
                d,
                "AAA",
                y_high=0.05,
                y_low=-0.05,
                pred_h=0.03,
                pred_l=-0.03,
                base_h=0.01,
                base_l=-0.01,
            )
        )
    oos = pd.DataFrame(rows)
    # Rename to shared family columns expected by pred_col(..., family)
    for c in list(oos.columns):
        if c.startswith("pred_"):
            oos[c.replace("pred_", "pred_shared_")] = oos[c]
    trades, daily = simulate_trades(oos, None, None, family="shared")
    # First ~40 rows needed for overall min — with only 30 bad labels, no trades
    assert trades.empty


# --- 2. Partial --full overwrite ---


def test_full_replace_rejects_thin_panel():
    existing = pd.DataFrame(
        [_bar(f"2023-01-{d:02d}", ticker=t, close=10.0) for t in ("AAA", "BBB", "CCC") for d in range(3, 28)]
    )
    thin = pd.DataFrame([_bar("2026-09-01", ticker="AAA", close=11.0)])
    ok, reason = full_replace_acceptable(existing, thin)
    assert ok is False
    assert "ticker" in reason or "depth" in reason or "span" in reason


def test_full_replace_accepts_comparable_panel():
    existing = pd.DataFrame(
        [_bar(f"2023-01-{d:02d}", ticker=t) for t in ("AAA", "BBB") for d in range(3, 20)]
    )
    incoming = pd.DataFrame(
        [_bar(f"2023-01-{d:02d}", ticker=t, close=11.0) for t in ("AAA", "BBB") for d in range(3, 20)]
    )
    ok, reason = full_replace_acceptable(existing, incoming)
    assert ok is True


def test_save_ohlcv_atomic(tmp_path):
    path = tmp_path / "ohlcv.parquet"
    df = pd.DataFrame([_bar("2026-09-01")])
    save_ohlcv(df, path)
    assert path.exists()
    assert not path.with_suffix(".parquet.tmp").exists()


# --- 5. Stop gap-through ---


def test_stop_gap_through_fills_at_open_not_stop():
    """Bar opens through stop → fill at open (worse), not theoretical stop."""
    bars = [
        {"open": 99.0, "high": 99.5, "low": 97.5, "close": 98.5},  # entry long @ 98
        {"open": 95.0, "high": 95.5, "low": 94.0, "close": 95.2},  # gaps through sl=96
    ]
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    assert filled.reason == "sl"
    assert filled.exit_px == 95.0  # open, not 96
    assert filled.ret == pytest.approx(95.0 / 98.0 - 1.0)


def test_short_stop_gap_through_fills_at_open():
    bars = [
        {"open": 101.0, "high": 102.5, "low": 100.5, "close": 101.5},  # short entry @ 102
        {"open": 105.0, "high": 105.5, "low": 104.0, "close": 104.5},  # gaps through sl=104
    ]
    filled = fill_fade_bars(-1, 102.0, 100.0, 104.0, bars)
    assert filled is not None
    assert filled.reason == "sl"
    assert filled.exit_px == 105.0


# --- 6. Incomplete 5m ---


def _full_rth(session: str, px: float = 100.0) -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 09:30:00", tz="America/New_York")
    idx = pd.date_range(start, periods=78, freq="5min")
    n = len(idx)
    return pd.DataFrame(
        {"open": [px] * n, "high": [px + 1] * n, "low": [px - 1] * n, "close": [px] * n, "volume": [1] * n},
        index=idx,
    )


def test_incomplete_5m_missing_close_is_incomplete():
    sess = date(2026, 9, 22)
    df = _full_rth("2026-09-22").iloc[:-10]  # truncated before 15:55
    assert rth_session_status(df, sess) == "incomplete"
    assert is_complete_rth(df, sess) is False


def test_incomplete_5m_missing_open_is_incomplete():
    sess = date(2026, 9, 22)
    df = _full_rth("2026-09-22").iloc[5:]  # starts after 09:30
    assert rth_session_status(df, sess) == "incomplete"


def test_complete_5m_accepted():
    sess = date(2026, 9, 22)
    assert is_complete_rth(_full_rth("2026-09-22"), sess) is True


def test_fetch_rth_rejects_incomplete(monkeypatch, tmp_path):
    session = date(2026, 9, 22)

    def alpaca_partial(tickers, sess):
        # Only morning bars — incomplete
        start = pd.Timestamp("2026-09-22 09:30:00", tz="America/New_York")
        idx = pd.date_range(start, periods=10, freq="5min")
        df = pd.DataFrame(
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            index=idx,
        )
        return {t: df for t in tickers}

    def yahoo_empty(tickers, sess):
        return {}

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    out = fetch_rth_5m(
        ["AAPL"],
        session,
        cache_dir=tmp_path,
        use_cache=False,
        alpaca_fetcher=alpaca_partial,
        yahoo_fetcher=yahoo_empty,
    )
    assert out == {}


# --- 7. Daily Alpaca pagination ---


def test_alpaca_daily_follows_next_page_token(monkeypatch):
    pages = [
        {
            "bars": {
                "AAPL": [
                    {"t": "2026-01-02T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 10},
                ]
            },
            "next_page_token": "page2",
        },
        {
            "bars": {
                "AAPL": [
                    {"t": "2026-01-03T05:00:00Z", "o": 1.5, "h": 2.5, "l": 1.4, "c": 2.0, "v": 11},
                ]
            },
            "next_page_token": None,
        },
    ]
    calls = {"n": 0}

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            import json

            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=90):
        i = calls["n"]
        calls["n"] += 1
        return _Resp(pages[i])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    prov = AlpacaProvider(key_id="k", secret="s")
    out = prov._fetch_chunk(["AAPL"], start="2026-01-01", end="2026-01-10T00:00:00Z", sym_to_orig={"AAPL": "AAPL"})
    assert calls["n"] == 2
    assert len(out) == 2
    assert list(out["date"].dt.date.astype(str)) == ["2026-01-02", "2026-01-03"]


# --- 8. Stale coverage ---


def test_coverage_stale_max_date_not_complete(monkeypatch):
    # Session present and fully covered for that day, but older than expected.
    from trendline.universe import load_sp500

    members = list(load_sp500()["ticker"])[:5]
    # Monkeypatch load_sp500 inside coverage to tiny universe so frac=1.0
    tiny = pd.DataFrame({"ticker": members[:2], "sector": ["X"] * 2})

    ohlcv = pd.DataFrame(
        [_bar("2026-09-10", ticker=t) for t in members[:2]]
    )
    with patch("trendline.universe.load_sp500", return_value=tiny):
        with patch("trendline.data.providers.expected_equity_session", return_value=pd.Timestamp("2026-09-14")):
            cov = equity_session_coverage(ohlcv, "2026-09-10", expected="2026-09-14")
    assert cov["coverage_ok"] is True
    assert cov["fresh"] is False
    assert cov["complete"] is False


# --- 9. Max DD start + Sharpe flat days ---


def test_max_dd_includes_starting_capital():
    # First day -50%: without prepend, peak=0.5 and dd=0; with prepend dd=-0.5
    trades = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ret": [-0.5]})
    daily = trades.copy()
    stats = trade_stats(trades, daily)
    assert stats["max_dd"] == pytest.approx(-0.5)


def test_sharpe_includes_no_trade_sessions():
    # One trade day then a gap — flat sessions dilute mean/vol
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "ret": [0.10, 0.10],
        }
    )
    daily = trades.copy()
    stats = trade_stats(trades, daily)
    # bdate_range 01-02..01-05 includes 01-03, 01-04 as zeros → 4 returns
    assert stats["n_trades"] == 2
    assert np.isfinite(stats["sharpe"])


# --- 10. FX Series normalize ---


def test_fx_normalize_multiindex_close():
    idx = pd.date_range("2026-09-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close"], ["HKD=X"]])
    df = pd.DataFrame([[7.8], [7.81], [7.82]], index=idx, columns=cols)
    # raw["Close"] shape when MultiIndex on columns from yfinance
    close = df["Close"]
    assert isinstance(close, pd.DataFrame)
    series = _normalize_fx_close(close)
    assert isinstance(series, pd.Series)
    assert float(series.iloc[-1]) == pytest.approx(7.82)


def test_fx_fallback_logs_warning(capsys):
    with patch("yfinance.download", side_effect=RuntimeError("boom")):
        out = _refresh_fx(7.75)
    assert out == 7.75
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "keeping 7.75" in captured.out


# --- 4. Exact session bar ---


def test_bar_on_session_exact_not_next_available():
    ohlcv = pd.DataFrame(
        [
            _bar("2026-09-14", "AAA", 100),
            _bar("2026-09-16", "AAA", 102),  # skipped 15th
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    assert _bar_on_session(ohlcv, "AAA", "2026-09-15") is None
    got = _bar_on_session(ohlcv, "AAA", "2026-09-16")
    assert got is not None
    assert float(got["close"]) == 102


def test_resolve_session_uses_modal_date():
    ohlcv = pd.DataFrame(
        [
            _bar("2026-09-15", "AAA", 100),
            _bar("2026-09-15", "BBB", 100),
            _bar("2026-09-16", "CCC", 100),  # odd one out
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    asof = pd.Timestamp("2026-09-12")
    sess = _resolve_exchange_session(ohlcv, asof, ["AAA", "BBB", "CCC"])
    assert sess == "2026-09-15"
