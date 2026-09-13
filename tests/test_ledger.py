import pandas as pd

from trendline.ledger import fade_score, new_ledger, planned_orders, realize_once
from trendline.range_touch import fill_fade


def _card(ticker, side, entry, prior, atr):
    return {
        "ticker": ticker,
        "side": side,
        "action": "做空" if side < 0 else "做多",
        "entry_px": entry,
        "prior_close": prior,
        "atr": atr,
        "tp": prior,
        "sl": entry + 3 if side < 0 else entry - 3,
    }


def test_higher_fade_score_gets_more_size():
    cards = {
        "asof": "2026-09-11",
        "cards": [
            _card("WEAK", -1, 101.0, 100.0, 2.0),   # room 1 / atr 2 = 0.5
            _card("STRONG", -1, 106.0, 100.0, 2.0), # room 6 / atr 2 = 3.0
        ],
    }
    out = {r["ticker"]: r for r in planned_orders(cards, 500_000, 7.8)}
    assert set(out) == {"WEAK", "STRONG"}
    assert out["STRONG"]["shares"] > out["WEAK"]["shares"]
    assert out["STRONG"]["weight"] <= 0.12 + 1e-9
    assert out["STRONG"]["fee_usd"] >= 2.0  # IBKR two-leg min around $1+$1


def test_skips_tiny_notional():
    # Short room must be entry > prior. Tiny book cannot clear US$2500 floor.
    cards = {"asof": "2026-09-11", "cards": [_card("TINY", -1, 102.0, 100.0, 2.0)]}
    assert planned_orders(cards, 8_000, 7.8) == []


def test_realize_is_idempotent():
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-11", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-14", "ticker": "AAA", "open": 100.5, "high": 102, "low": 97, "close": 99, "adj_close": 99, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    led = new_ledger()
    out = realize_once(led, ohlcv)
    assert out["realized_asofs"] == []


def test_fill_fee_math():
    from trendline.ibkr_fees import roundtrip_fees
    filled = fill_fade(-1, 102.0, 100.0, 104.0, 101.0, 102.5, 99.5, 100.5)
    assert filled is not None
    ret, exit_px, reason = filled
    assert reason == "tp"
    shares = 40
    fee = roundtrip_fees(-1, shares, 102.0, exit_px)["total"]
    pnl = shares * (exit_px - 102.0) * -1 - fee
    assert abs(pnl - (80 - fee)) < 1e-6
    assert fee >= 2.0


def test_three_books_same_start():
    from trendline.ledger import ledger_view, new_ledger
    v = ledger_view(new_ledger())
    eqs = [v["headlines"][f]["equity_hkd"] for f in ("shared", "sector", "stock")]
    assert eqs == [500_000.0, 500_000.0, 500_000.0]
    assert set(v["planned"]) == {"shared", "sector", "stock"}
