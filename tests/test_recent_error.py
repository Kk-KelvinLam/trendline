from trendline.cards import _recent_error


def test_walk_forward_fallback_when_oos_missing():
    out = _recent_error(
        None,
        "AAPL",
        per_ticker_row={"n": 441, "model_mae_close": 0.02},
        prior_close=100.0,
    )
    assert out["scope"] == "walk_forward"
    assert out["n"] == 441
    assert abs(out["mae_close_ret"] - 0.02) < 1e-12
    assert abs(out["mae_close_px"] - 2.0) < 1e-12
