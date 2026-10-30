"""Per-ticker last_bar_date on trade cards."""

from __future__ import annotations

import pandas as pd

from trendline.cards import _last_bar_dates, build_cards


class _StubModel:
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        n = len(frame)
        z = pd.Series([0.0] * n)
        return pd.DataFrame(
            {
                "pred_high_q10": z + 0.005,
                "pred_high_q50": z + 0.01,
                "pred_high_q90": z + 0.02,
                "pred_low_q10": z - 0.02,
                "pred_low_q50": z - 0.01,
                "pred_low_q90": z - 0.005,
                "pred_close_q10": z - 0.01,
                "pred_close_q50": z,
                "pred_close_q90": z + 0.01,
            }
        )


def test_last_bar_dates_max_per_ticker():
    ohlcv = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2026-09-10"},
            {"ticker": "AAPL", "date": "2026-09-11"},
            {"ticker": "MSFT", "date": "2026-09-09"},
            {"ticker": "MSFT", "date": "2026-09-10"},
        ]
    )
    m = _last_bar_dates(ohlcv)
    assert m["AAPL"] == "2026-09-11"
    assert m["MSFT"] == "2026-09-10"


def test_last_bar_dates_normalizes_tz():
    ohlcv = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2026-09-11 16:00", tz="America/New_York"),
            },
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2026-09-10 16:00", tz="America/New_York"),
            },
        ]
    )
    assert _last_bar_dates(ohlcv)["AAPL"] == "2026-09-11"


def test_build_cards_last_bar_date_when_ohlcv_stops_early(monkeypatch):
    """Ticker whose ohlcv ends before card asof gets that earlier last_bar_date."""
    asof = pd.Timestamp("2026-09-12")
    featured = pd.DataFrame(
        [
            {
                "date": asof,
                "ticker": "AAPL",
                "sector": "Information Technology",
                "close": 100.0,
                "atr": 2.0,
                "ret_20d": 0.01,
                "dist_ma50": 0.02,
            },
            {
                "date": asof,
                "ticker": "MSFT",
                "sector": "Information Technology",
                "close": 200.0,
                "atr": 3.0,
                "ret_20d": 0.02,
                "dist_ma50": 0.01,
            },
        ]
    )
    # AAPL has bars through asof; MSFT stops two sessions earlier (partial Yahoo).
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-10", "ticker": "AAPL", "close": 99.0, "volume": 1e7, "source": "yfinance"},
            {"date": "2026-09-11", "ticker": "AAPL", "close": 99.5, "volume": 1e7, "source": "yfinance"},
            {"date": "2026-09-12", "ticker": "AAPL", "close": 100.0, "volume": 1e7, "source": "yfinance"},
            {"date": "2026-09-09", "ticker": "MSFT", "close": 198.0, "volume": 9e6, "source": "yfinance"},
            {"date": "2026-09-10", "ticker": "MSFT", "close": 199.0, "volume": 9e6, "source": "yfinance"},
        ]
    )
    ranked = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "dvol_rank": [1, 2],
            "dollar_volume": [1e9, 9e8],
        }
    )
    monkeypatch.setattr("trendline.cards.rank_by_dollar_volume", lambda *a, **k: ranked)

    per_ticker = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"], "beats_range": [False, False]}
    )
    payload = build_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=pd.DataFrame(),
        per_ticker=per_ticker,
        overall_beats=False,
        model=_StubModel(),
        asof=asof,
        family="shared",
    )
    by_ticker = {c["ticker"]: c for c in payload["cards"]}
    assert by_ticker["AAPL"]["last_bar_date"] == "2026-09-12"
    assert by_ticker["MSFT"]["last_bar_date"] == "2026-09-10"
    assert by_ticker["MSFT"]["asof"] == "2026-09-12"
    assert by_ticker["MSFT"]["last_bar_date"] < by_ticker["MSFT"]["asof"]
