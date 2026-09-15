"""Synthetic RTH 5m fill sequences — no network."""

import pandas as pd

from trendline.range_touch import fill_fade_bars


def _bars(*rows):
    return [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in rows]


def test_tp_before_entry_then_flatten_is_close_not_tp():
    bars = _bars(
        (99.0, 101.0, 98.5, 100.5),
        (99.0, 99.5, 97.5, 99.0),
        (99.0, 99.2, 98.8, 99.1),
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    assert filled.reason == "close"
    assert filled.exit_px == 99.1
    assert abs(filled.ret - (99.1 / 98.0 - 1.0)) < 1e-12


def test_same_bar_sl_and_tp_prefers_sl():
    bars = _bars((99.0, 101.0, 95.5, 100.0))
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    assert filled.reason == "sl"
    assert filled.exit_px == 96.0
    assert filled.ret < 0


def test_gap_through_returns_none():
    assert fill_fade_bars(1, 98.0, 100.0, 96.0, _bars((97.0, 99.0, 96.5, 98.0))) is None
    assert fill_fade_bars(-1, 102.0, 100.0, 104.0, _bars((103.0, 104.0, 101.0, 102.0))) is None


def test_last_bar_flatten_close():
    bars = _bars(
        (99.0, 99.5, 97.5, 98.5),
        (98.5, 99.0, 98.0, 98.8),
        (98.8, 99.0, 98.5, 98.7),
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    assert filled.reason == "close"
    assert filled.exit_px == 98.7


def test_short_fill_then_tp():
    bars = _bars(
        (101.0, 102.5, 100.5, 101.5),
        (101.0, 101.5, 99.5, 100.0),
    )
    filled = fill_fade_bars(-1, 102.0, 100.0, 104.0, bars)
    assert filled is not None
    assert filled.reason == "tp"
    assert filled.exit_px == 100.0
    assert abs(filled.ret - (102.0 - 100.0) / 102.0) < 1e-12


def test_never_touched_is_miss():
    bars = _bars(
        (99.0, 99.5, 98.5, 99.0),
        (99.0, 99.2, 98.6, 98.9),
    )
    assert fill_fade_bars(1, 98.0, 100.0, 96.0, bars) is None


def test_entry_exit_timestamps_from_index():
    idx = pd.DatetimeIndex(
        [
            "2026-09-14 09:30:00",
            "2026-09-14 09:35:00",
            "2026-09-14 09:40:00",
        ],
        tz="America/New_York",
    )
    df = pd.DataFrame(
        {
            "open": [99.0, 99.0, 99.0],
            "high": [99.5, 99.5, 99.2],
            "low": [98.5, 97.5, 98.8],
            "close": [99.0, 99.0, 99.1],
        },
        index=idx,
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, df)
    assert filled is not None
    assert filled.reason == "close"
    assert filled.entry_ts is not None and "09:35" in filled.entry_ts
    assert filled.exit_ts is not None and "09:40" in filled.exit_ts
