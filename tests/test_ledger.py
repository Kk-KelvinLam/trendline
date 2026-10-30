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
    # Many similar names so size still differs by fade score under 5%/N cap.
    cards = {
        "asof": "2026-09-11",
        "cards": [_card("STRONG", -1, 100.0, 94.0, 2.0)]
        + [_card(f"W{i}", -1, 100.0, 99.0, 2.0) for i in range(9)],
    }
    out = {r["ticker"]: r for r in planned_orders(cards, 500_000, 7.8)}
    assert "STRONG" in out
    weaks = [out[k]["shares"] for k in out if k != "STRONG"]
    assert weaks
    assert out["STRONG"]["shares"] > max(weaks)
    assert out["STRONG"]["weight"] <= 0.12 + 1e-9
    assert out["STRONG"]["fee_usd"] >= 2.0


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
    ret, exit_px, reason = filled.ret, filled.exit_px, filled.reason
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
    assert v["headlines"]["voo"]["equity_hkd"] == 500_000.0
    assert set(v["planned"]) == {"shared", "sector", "stock"}


def test_realize_marks_voo_buy_and_hold(monkeypatch):
    import trendline.ledger as ledger_mod
    from trendline.ledger import new_ledger, realize_once

    cards = {
        "asof": "2026-09-11",
        "cards": [
            {
                "ticker": "AAA",
                "side": 0,
                "action": "觀望",
                "prior_close": 100.0,
                "pred": {"high": {"q50": 101}, "low": {"q50": 99}, "close": {"q50": 100}},
            }
        ],
    }
    monkeypatch.setattr(ledger_mod, "_read_cards", lambda path: cards)
    monkeypatch.setattr(ledger_mod, "_refresh_fx", lambda default: 7.8)

    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-14", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-15", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-14", "ticker": "VOO", "open": 490, "high": 505, "low": 489, "close": 500, "adj_close": 500, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-15", "ticker": "VOO", "open": 508, "high": 512, "low": 507, "close": 510, "adj_close": 510, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    cards["asof"] = "2026-09-14"
    out = realize_once(new_ledger(), ohlcv, bars_by_ticker={})
    b = out["benchmark"]
    assert b["entry_session"] == "2026-09-14"
    assert b["entry_kind"] == "open"
    assert b["entry_px"] == 490.0
    assert b["last_px"] == 510.0
    assert b["shares"] == int((500_000 / 7.8) // 490)
    usd = 500_000 / 7.8
    cash = usd - b["shares"] * 490.0
    assert abs(b["equity_hkd"] - (b["shares"] * 510.0 + cash) * 7.8) < 1e-6
    assert out["days"][-1]["equity_hkd"]["voo"] == b["equity_hkd"]


def test_sync_benchmark_when_asof_already_realized():
    from trendline.ledger import new_ledger, realize_once, sync_benchmark

    led = new_ledger()
    led["realized_asofs"] = ["2026-09-14"]
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-14", "ticker": "VOO", "open": 490, "high": 505, "low": 489, "close": 500, "adj_close": 500, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-15", "ticker": "VOO", "open": 508, "high": 512, "low": 507, "close": 510, "adj_close": 510, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    skipped = realize_once(led, ohlcv, bars_by_ticker={})
    assert skipped["benchmark"]["shares"] == 0
    out = sync_benchmark(skipped, ohlcv, fx=7.8)
    b = out["benchmark"]
    assert b["entry_px"] == 490.0
    assert b["last_session"] == "2026-09-15"
    assert b["last_px"] == 510.0
    assert b["shares"] > 0


def test_spend_uses_current_equity_not_start():
    cards = {
        "asof": "2026-09-11",
        "cards": [_card(f"T{i}", -1, 100.0, 98.0, 2.0) for i in range(20)],
    }
    fat = planned_orders(cards, 500_000, 7.8)
    thin = planned_orders(cards, 250_000, 7.8)
    assert fat and thin

    def spend(rows):
        return sum(r["notional_usd"] + r["fee_usd"] for r in rows)

    assert spend(fat) <= 500_000 / 7.8 + 1e-6
    assert spend(thin) <= 250_000 / 7.8 + 1e-6
    assert spend(thin) < spend(fat)


def test_realize_uses_5m_when_bars_injected(monkeypatch):
    """Inject RTH 5m bars; do not hit Yahoo. Fill path must record fill_source=5m."""
    import trendline.ledger as ledger_mod
    from trendline.ledger import new_ledger, realize_once

    cards = {
        "asof": "2026-09-11",
        "cards": [
            {
                "ticker": "AAA",
                "side": 1,
                "action": "做多",
                "entry_px": 98.0,
                "prior_close": 100.0,
                "atr": 2.0,
                "tp": 100.0,
                "sl": 96.0,
                "pred": {"high": {"q50": 101}, "low": {"q50": 97}, "close": {"q50": 100}},
            }
        ],
    }

    def fake_read(path):
        return cards

    monkeypatch.setattr(ledger_mod, "_read_cards", fake_read)
    monkeypatch.setattr(ledger_mod, "_refresh_fx", lambda default: 7.8)

    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-11", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            # Daily path alone would miss (low never <= 98); 5m path fills then flattens.
            {"date": "2026-09-12", "ticker": "AAA", "open": 99.5, "high": 100.0, "low": 98.5, "close": 99.0, "adj_close": 99, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    bars = {
        "AAA": [
            # fill long (low<=98); high stays below TP=100 and low above SL=96
            {"open": 99.5, "high": 99.8, "low": 97.5, "close": 98.5},
            {"open": 98.5, "high": 99.0, "low": 98.0, "close": 98.8},
        ]
    }
    led = new_ledger()
    out = realize_once(led, ohlcv, bars_by_ticker=bars)
    assert "2026-09-11" in out["realized_asofs"]
    fills = [f for f in out["fills"] if f["ticker"] == "AAA"]
    assert fills
    assert all(f["fill_source"] == "5m" for f in fills)
    assert fills[0]["reason"] == "close"
    assert fills[0]["exit"] == 98.8


def test_realize_misses_when_5m_missing(monkeypatch):
    import trendline.ledger as ledger_mod
    from trendline.ledger import new_ledger, realize_once

    cards = {
        "asof": "2026-09-11",
        "cards": [
            {
                "ticker": "BBB",
                "side": -1,
                "action": "做空",
                "entry_px": 102.0,
                "prior_close": 100.0,
                "atr": 2.0,
                "tp": 100.0,
                "sl": 104.0,
                "pred": {"high": {"q50": 103}, "low": {"q50": 99}, "close": {"q50": 100}},
            }
        ],
    }
    monkeypatch.setattr(ledger_mod, "_read_cards", lambda path: cards)
    monkeypatch.setattr(ledger_mod, "_refresh_fx", lambda default: 7.8)

    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-11", "ticker": "BBB", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-12", "ticker": "BBB", "open": 101.0, "high": 102.5, "low": 99.5, "close": 100.5, "adj_close": 100.5, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    led = new_ledger()
    out = realize_once(led, ohlcv, bars_by_ticker={})  # empty 5m → miss, no daily fill
    fills = [f for f in out["fills"] if f["ticker"] == "BBB"]
    assert fills == []
    day = out["days"][-1]["families"]["shared"]
    assert day["n_signals"] == 1
    assert day["n_miss"] == 1
    assert day["paper_fills"] == 0


def test_realize_stores_planned_snapshot(monkeypatch):
    import trendline.ledger as ledger_mod
    from trendline.ledger import new_ledger, realize_once

    cards = {
        "asof": "2026-09-11",
        "cards": [
            {
                "ticker": "AAA",
                "side": 1,
                "action": "做多",
                "entry_px": 98.0,
                "prior_close": 100.0,
                "atr": 2.0,
                "tp": 100.0,
                "sl": 96.0,
                "pred": {"high": {"q50": 101}, "low": {"q50": 97}, "close": {"q50": 100}},
            }
        ],
    }
    monkeypatch.setattr(ledger_mod, "_read_cards", lambda path: cards)
    monkeypatch.setattr(ledger_mod, "_refresh_fx", lambda default: 7.8)
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-11", "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1, "source": "yfinance"},
            {"date": "2026-09-12", "ticker": "AAA", "open": 99.5, "high": 100.0, "low": 98.5, "close": 99.0, "adj_close": 99, "volume": 1, "source": "yfinance"},
        ]
    )
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    bars = {"AAA": [{"open": 99.5, "high": 99.8, "low": 97.5, "close": 98.5}]}
    out = realize_once(new_ledger(), ohlcv, bars_by_ticker=bars)
    planned = ((out["days"][0].get("families") or {}).get("shared") or {}).get("planned") or []
    assert planned and planned[0]["ticker"] == "AAA"
    assert planned[0]["shares"] >= 1
    assert out["days"][0]["actuals"]["AAA"]["close"] == 99.0


def test_skips_poor_mae_rank():
    liquid = _card("LIQ", -1, 102.0, 100.0, 2.0)
    liquid["mae_rank"] = 20
    thin = _card("THIN", -1, 102.0, 100.0, 2.0)
    thin["mae_rank"] = 401
    cards = {"asof": "2026-09-11", "cards": [liquid, thin]}
    out = {r["ticker"]: r for r in planned_orders(cards, 500_000, 7.8)}
    assert "LIQ" in out
    assert "THIN" not in out


def test_locked_open_plan_used_instead_of_recompute(monkeypatch):
    from trendline.ledger import _locked_plan_rows, new_ledger

    payload = {"asof": "2026-09-16", "cards": []}
    led = new_ledger()
    led["open_plan"] = {
        "asof": "2026-09-16",
        "asofs": {"shared": "2026-09-16"},
        "families": {"shared": [{"ticker": "LOCK", "shares": 7, "side": 1}]},
    }
    rows = _locked_plan_rows(led, "shared", payload, 500_000, 7.8)
    assert rows == [{"ticker": "LOCK", "shares": 7, "side": 1}]


def test_prune_plan_history_keeps_last_three():
    from trendline.ledger import _prune_plan_history

    def day(asof, planned=True):
        rec = {"n_fills": 0}
        if planned:
            rec["planned"] = [{"ticker": "X"}]
        return {"asof": asof, "families": {"shared": rec, "sector": dict(rec), "stock": dict(rec)}}

    led = {"days": [day("d1"), day("d2"), day("d3"), day("d4")]}
    _prune_plan_history(led)
    assert "planned" not in led["days"][0]["families"]["shared"]
    assert "planned" in led["days"][-1]["families"]["shared"]
    assert "planned" in led["days"][-3]["families"]["shared"]


def test_fewer_names_raise_name_risk():
    def wide(ticker):
        row = _card(ticker, -1, 20.0, 18.0, 2.0)
        row["sl"] = 28.0
        return row
    few = {"asof": "2026-09-11", "cards": [wide("A"), wide("B")]}
    many = {"asof": "2026-09-11", "cards": [wide(f"N{i}") for i in range(10)]}
    few_rows = planned_orders(few, 500_000, 7.8)
    many_rows = planned_orders(many, 500_000, 7.8)
    assert few_rows and many_rows
    assert max(r["risk_frac"] for r in few_rows) > max(r["risk_frac"] for r in many_rows)
    assert sum(r["risk_frac"] for r in few_rows) <= 0.05 + 1e-6
    assert sum(r["risk_frac"] for r in many_rows) <= 0.05 + 1e-6


def test_enrich_plan_vs_actual_miss_close_and_tp():
    from trendline.ledger import enrich_plan_vs_actual, index_fills_for_plan, index_session_actuals

    rows = [
        {"ticker": "MISS", "entry": 100.0, "tp": 98.0, "sl": 103.0},
        {"ticker": "CLS", "entry": 50.0, "tp": 52.0, "sl": 48.0},
        {"ticker": "HIT", "entry": 10.0, "tp": 11.0, "sl": 9.0},
        {"ticker": "NOBAR", "entry": 20.0, "tp": 21.0, "sl": 19.0},
    ]
    fills = [
        {
            "family": "shared",
            "session": "2026-09-18",
            "asof": "2026-09-17",
            "ticker": "CLS",
            "exit": 51.25,
            "reason": "close",
        },
        {
            "family": "shared",
            "session": "2026-09-18",
            "asof": "2026-09-17",
            "ticker": "HIT",
            "exit": 11.0,
            "reason": "tp",
        },
        {
            "family": "sector",
            "session": "2026-09-18",
            "ticker": "CLS",
            "exit": 99.0,
            "reason": "close",
        },
    ]
    fmap = index_fills_for_plan(fills, family="shared", session="2026-09-18")
    assert set(fmap) == {"CLS", "HIT"}
    actuals = index_session_actuals(
        {"MISS": {"high": 101.5, "low": 99.2, "close": 99.4}},
        tickers=["MISS", "NOBAR"],
        session="2026-09-18",
        ohlcv=pd.DataFrame(
            [
                {
                    "date": "2026-09-18",
                    "ticker": "NOBAR",
                    "high": 21.0,
                    "low": 19.5,
                    "close": 20.75,
                }
            ]
        ),
    )
    out = {r["ticker"]: r for r in enrich_plan_vs_actual(rows, fmap, actuals)}
    # Miss: session close − planned entry. No tp/sl distance.
    assert out["MISS"]["entry_vs_actual"] == -0.6
    assert out["MISS"]["tp_vs_actual"] is None
    assert out["MISS"]["sl_vs_actual"] is None
    assert out["NOBAR"]["entry_vs_actual"] == 0.75
    # Flattened at close: unfinished TP/SL, hide entry (they did enter).
    assert out["CLS"]["entry_vs_actual"] is None
    assert out["CLS"]["tp_vs_actual"] == -0.75
    assert out["CLS"]["sl_vs_actual"] == 3.25
    # TP/SL already done: no distance columns.
    assert out["HIT"]["entry_vs_actual"] is None
    assert out["HIT"]["tp_vs_actual"] is None
    assert out["HIT"]["sl_vs_actual"] is None


def test_index_fills_for_plan_asof_fallback():
    from trendline.ledger import index_fills_for_plan

    fills = [
        {"family": "stock", "asof": "2026-09-17", "session": "2026-09-18", "ticker": "A", "exit": 1.0},
    ]
    assert index_fills_for_plan(fills, family="stock", session="2026-09-18")["A"]["exit"] == 1.0
    assert index_fills_for_plan(fills, family="stock", asof="2026-09-17")["A"]["exit"] == 1.0
    assert index_fills_for_plan(fills, family="stock", session="2099-01-01") == {}

def test_format_plan_distance_absolute_and_pct():
    from trendline.ledger import format_plan_distance

    assert format_plan_distance(-15.01, 492.1311475409836) == "-15.01 (−3.05%)"
    assert format_plan_distance(2.90, 400.0) == "+2.90 (+0.72%)"
    assert format_plan_distance(0.0, 100.0) == "+0.00 (+0.00%)"
    assert format_plan_distance(None, 100.0) is None
    assert format_plan_distance(float("nan"), 100.0) is None
    # Missing / zero reference: signed absolute only, still blank on None distance.
    assert format_plan_distance(-1.5, None) == "-1.50"
    assert format_plan_distance(-1.5, 0) == "-1.50"
    # Enrich still returns floats; display helper is separate.
    assert format_plan_distance(-0.6, 100.0) == "-0.60 (−0.60%)"
    assert format_plan_distance(-0.75, 52.0) == "-0.75 (−1.44%)"

