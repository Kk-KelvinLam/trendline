"""Synthetic RTH 5m fill sequences — no network."""

from trendline.range_touch import fill_fade_bars


def _bars(*rows):
    return [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in rows]


def test_tp_before_entry_then_flatten_is_close_not_tp():
    # Long: TP prints on bar 0 before touch; fill later; neither SL nor TP after → close
    bars = _bars(
        (99.0, 101.0, 98.5, 100.5),  # high tags TP=100 but low never <= entry=98
        (99.0, 99.5, 97.5, 99.0),  # fills; no SL/TP
        (99.0, 99.2, 98.8, 99.1),  # last bar flatten
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "close"
    assert exit_px == 99.1
    assert abs(ret - (99.1 / 98.0 - 1.0)) < 1e-12


def test_same_bar_sl_and_tp_prefers_sl():
    bars = _bars(
        (99.0, 101.0, 95.5, 100.0),  # fill long + both SL and TP → SL
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "sl"
    assert exit_px == 96.0
    assert ret < 0


def test_gap_through_returns_none():
    # Long: session open already through / at entry
    assert fill_fade_bars(1, 98.0, 100.0, 96.0, _bars((97.0, 99.0, 96.5, 98.0))) is None
    # Short: session open already through / at entry
    assert fill_fade_bars(-1, 102.0, 100.0, 104.0, _bars((103.0, 104.0, 101.0, 102.0))) is None


def test_last_bar_flatten_close():
    bars = _bars(
        (99.0, 99.5, 97.5, 98.5),  # fill long, no SL/TP
        (98.5, 99.0, 98.0, 98.8),  # still open
        (98.8, 99.0, 98.5, 98.7),  # last RTH → close
    )
    filled = fill_fade_bars(1, 98.0, 100.0, 96.0, bars)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "close"
    assert exit_px == 98.7


def test_short_fill_then_tp():
    bars = _bars(
        (101.0, 102.5, 100.5, 101.5),  # fill short at 102 (high>=entry)
        (101.0, 101.5, 99.5, 100.0),  # TP
    )
    filled = fill_fade_bars(-1, 102.0, 100.0, 104.0, bars)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "tp"
    assert exit_px == 100.0
    assert abs(ret - (102.0 - 100.0) / 102.0) < 1e-12


def test_never_touched_is_miss():
    bars = _bars(
        (99.0, 99.5, 98.5, 99.0),
        (99.0, 99.2, 98.6, 98.9),
    )
    assert fill_fade_bars(1, 98.0, 100.0, 96.0, bars) is None
