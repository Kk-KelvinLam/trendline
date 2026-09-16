from trendline.cards import _recent_error


def test_walk_forward_fallback_when_oos_missing(monkeypatch):
    # Ignore on-disk recent_close_error_*.json so we exercise the WF fallback path.
    monkeypatch.setattr("trendline.cards._load_recent_error_artifact", lambda family="shared": {})
    out = _recent_error(
        None,
        "AAPL",
        per_ticker_row={"n": 441, "model_mae_close": 0.02},
        prior_close=100.0,
        recent_lookup={},
    )
    assert out["scope"] == "walk_forward"
    assert out["n"] == 441
    assert abs(out["mae_close_ret"] - 0.02) < 1e-12
    assert abs(out["mae_close_px"] - 2.0) < 1e-12

from trendline.range_touch import FadeSetup
from trendline.cards import _apply_recent_mae_decision, _recent_error


def test_apply_recent_mae_flattens_when_too_high():
    setup = FadeSetup(1, 100.0, 101.0, 99.0, 1.0, "ok")
    err = {"n": 20, "mae_close_ret": 0.03, "mae_close_px": 4.0, "scope": "recent"}
    out = _apply_recent_mae_decision(setup, err)
    assert out.side == 0
    assert out.reason == "recent_close_mae_too_high"




def test_apply_recent_mae_ignores_walk_forward_scope():
    setup = FadeSetup(1, 100.0, 101.0, 99.0, 1.0, "ok")
    err = {"n": 441, "mae_close_ret": 0.04, "mae_close_px": 4.0, "scope": "walk_forward"}
    out = _apply_recent_mae_decision(setup, err)
    assert out.side == 1




def test_recent_lookup_preferred():
    err = _recent_error(
        None,
        "AAPL",
        recent_lookup={"AAPL": {"n": 20, "mae_close_ret": 0.01, "mae_close_px": 1.0, "scope": "recent"}},
    )
    assert err["scope"] == "recent"
    assert err["n"] == 20
