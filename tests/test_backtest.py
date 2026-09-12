from __future__ import annotations

import numpy as np
import pandas as pd

from trendline.backtest import _fill_long, _fill_short, walk_forward_folds
from trendline.metrics import coverage, mae, mape, rmse, trade_stats


def test_mae_rmse_known_values():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert mae(y, p) == 2.0 / 3.0
    assert np.isclose(rmse(y, p), np.sqrt((0 + 0 + 4) / 3))
    assert np.isclose(mape(np.array([10.0, 10.0]), np.array([11.0, 9.0])), 0.1)


def test_coverage_interval():
    y = np.array([0.0, 0.5, 1.0, 1.5])
    lo = np.array([-1.0, 0.0, 0.0, 2.0])
    hi = np.array([1.0, 1.0, 2.0, 3.0])
    # hits: yes, yes, yes, no
    assert coverage(y, lo, hi) == 0.75


def test_fill_long_stop_first_when_both_hit():
    ret, px, reason = _fill_long(entry=100, high=110, low=90, close=105, tp=108, sl=95)
    assert reason == "sl"
    assert px == 95
    assert np.isclose(ret, -0.05)


def test_fill_short_take_profit():
    ret, px, reason = _fill_short(entry=100, high=101, low=94, close=96, tp=95, sl=106)
    assert reason == "tp"
    assert np.isclose(ret, 0.05)


def test_walk_forward_purges_gap():
    dates = pd.bdate_range("2023-01-02", periods=400)
    folds = walk_forward_folds(pd.Series(dates))
    assert folds
    for f in folds:
        gap = (f.test_start - f.train_end).days
        assert gap > 0
        assert f.train_end < f.test_start


def test_trade_stats_from_known_pnl():
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ret": [0.02, -0.01, 0.01],
        }
    )
    daily = trades[["date", "ret"]]
    stats = trade_stats(trades, daily)
    assert stats["n_trades"] == 3
    assert np.isclose(stats["hit_rate"], 2 / 3)
    assert stats["payoff"] > 0
