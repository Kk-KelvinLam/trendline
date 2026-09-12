from __future__ import annotations

import numpy as np
import pandas as pd

from trendline.config import FEATURE_COLS
from trendline.features import assert_no_future_in_features, build_features


def test_features_do_not_use_next_day(toy_ohlcv):
    featured = build_features(toy_ohlcv)
    assert_no_future_in_features(toy_ohlcv, featured, "AAPL")


def test_targets_are_next_session_returns(toy_ohlcv):
    featured = build_features(toy_ohlcv)
    aapl = featured[featured["ticker"] == "AAPL"].dropna(subset=["y_close"]).sort_values("date")
    row = aapl.iloc[50]
    nxt = featured[(featured["ticker"] == "AAPL") & (featured["date"] == row["next_date"])].iloc[0]
    assert np.isclose(row["y_close"], nxt["close"] / row["close"] - 1.0)
    assert np.isclose(row["y_high"], nxt["high"] / row["close"] - 1.0)
    assert np.isclose(row["y_low"], nxt["low"] / row["close"] - 1.0)


def test_known_feature_values():
    # Two days so ret_1d on day 2 is determined only by day 1–2 closes.
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "ticker": ["AAA", "AAA", "SPY", "SPY"],
            "open": [10.0, 11.0, 100.0, 101.0],
            "high": [10.5, 11.5, 101.0, 102.0],
            "low": [9.5, 10.5, 99.0, 100.0],
            "close": [10.0, 12.0, 100.0, 103.0],
            "adj_close": [10.0, 12.0, 100.0, 103.0],
            "volume": [100, 100, 100, 100],
        }
    )
    feat = build_features(raw)
    day2 = feat[(feat["ticker"] == "AAA") & (feat["date"] == dates[1])].iloc[0]
    assert np.isclose(day2["ret_1d"], 0.2)
    assert np.isclose(day2["overnight_gap"], 11.0 / 10.0 - 1.0)
    assert "ret_1d" in FEATURE_COLS
