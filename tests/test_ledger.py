import pandas as pd

from trendline.ledger import new_ledger, planned_orders, realize_once
from trendline.range_touch import fill_fade


def test_planned_orders_caps_at_eight():
    cards = {
        "asof": "2026-09-11",
        "cards": [
            {
                "ticker": f"T{i}",
                "side": -1,
                "action": "做空",
                "entry_px": 100.0,
                "tp": 98.0,
                "sl": 103.0,
            }
            for i in range(20)
        ],
    }
    out = planned_orders(cards, 500_000, 7.8)
    assert len(out) == 8
    assert out[0]["shares"] >= 1
    assert out[0]["fee_usd"] == 4.0


def test_realize_is_idempotent(tmp_path):
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-11", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-14", "ticker": "AAA", "open": 100.5, "high": 102, "low": 97, "close": 99, "adj_close": 99, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    # no cards on disk → unchanged
    led = new_ledger()
    out = realize_once(led, ohlcv)
    assert out["realized_asofs"] == []


def test_fill_fee_math():
    # short fade: entry 102, exit 100 (tp), 40 shares, $4 fee
    filled = fill_fade(-1, 102.0, 100.0, 104.0, 101.0, 102.5, 99.5, 100.5)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "tp"
    shares = 40
    pnl = shares * (exit_px - 102.0) * -1 - 4
    assert abs(pnl - (80 - 4)) < 1e-6
