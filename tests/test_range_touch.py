from trendline.range_touch import choose_setup, fill_fade, gapped_through


def test_picks_side_with_more_room():
    # close 100, ATR 2; low -2% (room 2), high +1% (room 1) → long
    s = choose_setup(100.0, 2.0, q50h=0.01, q50l=-0.02, q10l=-0.035, q90h=0.03, allowed=True)
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
    # filled long at 98; low tags SL 96 and high tags TP 100 → SL
    ret, exit_px, reason = fill_fade(1, 98.0, 100.0, 96.0, 99.0, 101.0, 95.5, 100.5)
    assert reason == "sl"
    assert exit_px == 96.0
    assert ret < 0
