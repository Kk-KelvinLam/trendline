from trendline.range_touch import choose_setup, fill_fade, gapped_through


def test_picks_side_with_more_room():
    s = choose_setup(100.0, 2.0, q50h=0.01, q50l=-0.02, q10l=-0.035, q90h=0.03, allowed=True, q50c=0.001)
    assert s.side == 1
    assert s.entry == 98.0
    assert s.tp == 100.0


def test_rejects_close_direction_without_range_edge():
    s = choose_setup(100.0, 2.0, 0.02, -0.02, -0.03, 0.03, allowed=False)
    assert s.side == 0
    assert s.reason == "range_model_does_not_beat_baseline"


def test_skip_gap_through_no_chase():
    assert gapped_through(1, 97.0, 98.0) is True
    assert gapped_through(-1, 103.0, 102.0) is True
    assert fill_fade(1, 98.0, 100.0, 96.0, 97.0, 101.0, 96.5, 99.0) is None


def test_conservative_stop_if_both_print():
    filled = fill_fade(1, 98.0, 100.0, 96.0, 99.0, 101.0, 95.5, 100.5)
    assert filled is not None
    assert filled.reason == "sl"
    assert filled.exit_px == 96.0
    assert filled.ret < 0
    assert filled.entry_ts is None and filled.exit_ts is None


def test_flatten_at_session_close():
    filled = fill_fade(1, 98.0, 100.0, 96.0, 99.0, 99.5, 97.5, 99.2)
    assert filled is not None
    assert filled.reason == "close"
    assert filled.exit_px == 99.2
    assert abs(filled.ret - (99.2 / 98.0 - 1.0)) < 1e-12
    filled = fill_fade(-1, 102.0, 100.0, 104.0, 101.0, 103.0, 100.5, 101.4)
    assert filled is not None
    assert filled.reason == "close"
    assert filled.exit_px == 101.4


def test_close_gate_blocks_long_when_close_bearish():
    # Larger long room; short room below min; Close bearish → flat
    s = choose_setup(100.0, 2.0, q50h=0.005, q50l=-0.02, q10l=-0.035, q90h=0.03, allowed=True, q50c=-0.005)
    assert s.side == 0
    assert s.reason == "close_disagrees_with_fade"


def test_close_gate_allows_short_when_close_bearish():
    # Short has more room and Close agrees
    s = choose_setup(100.0, 2.0, q50h=0.025, q50l=-0.01, q10l=-0.02, q90h=0.04, allowed=True, q50c=-0.004)
    assert s.side == -1
