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


from trendline.cards import _mae_score, compute_recent_close_errors, rank_by_recent_mae
import pandas as pd


def test_mae_score_weights():
    err = {"mae_high_ret": 0.10, "mae_low_ret": 0.20, "mae_close_ret": 0.30}
    assert abs(_mae_score(err) - (0.4 * 0.10 + 0.4 * 0.20 + 0.2 * 0.30)) < 1e-12
    assert _mae_score({"mae_high_ret": 0.1, "mae_low_ret": None, "mae_close_ret": 0.1}) is None


def test_rank_by_recent_mae_order_and_min_n():
    recent = {
        "AAA": {"n": 20, "mae_high_ret": 0.01, "mae_low_ret": 0.01, "mae_close_ret": 0.01},
        "BBB": {"n": 20, "mae_high_ret": 0.02, "mae_low_ret": 0.02, "mae_close_ret": 0.02},
        "CCC": {"n": 5, "mae_high_ret": 0.001, "mae_low_ret": 0.001, "mae_close_ret": 0.001},
        "DDD": {"n": 20, "mae_high_ret": 0.01, "mae_low_ret": 0.01, "mae_close_ret": 0.01},
    }
    dvol = {"AAA": 1.0, "DDD": 9.0, "BBB": 5.0}
    out = rank_by_recent_mae(recent, dvol, top_n=3)
    assert [r["ticker"] for r in out] == ["DDD", "AAA", "BBB"]
    assert out[0]["mae_rank"] == 1
    assert "CCC" not in {r["ticker"] for r in out}


class _HLStub:
    def predict(self, frame):
        # Constant forecasts so MAE equals |y - 0|
        n = len(frame)
        z = [0.0] * n
        return pd.DataFrame(
            {
                "pred_high_q50": z,
                "pred_low_q50": z,
                "pred_close_q50": z,
            }
        )


def test_compute_recent_includes_high_low():
    rows = []
    asof = pd.Timestamp("2026-09-20")
    for i in range(12):
        d = asof - pd.Timedelta(days=i + 1)
        rows.append(
            {
                "date": d,
                "ticker": "AAA",
                "close": 100.0,
                "y_high": 0.02,
                "y_low": -0.01,
                "y_close": 0.00,
            }
        )
        rows.append(
            {
                "date": d,
                "ticker": "BBB",
                "close": 50.0,
                "y_high": 0.10,
                "y_low": -0.10,
                "y_close": 0.05,
            }
        )
    featured = pd.DataFrame(rows)
    out = compute_recent_close_errors(featured, _HLStub(), asof=asof)
    assert out["AAA"]["n"] == 12
    assert "mae_high_ret" in out["AAA"]
    assert "mae_low_ret" in out["AAA"]
    assert out["AAA"]["mae_score"] < out["BBB"]["mae_score"]
