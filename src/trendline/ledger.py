"""Paper ledger: realize last night's cards against the next session's bar.

Not a broker. One HKD book trades the shared fade; all three families are scored.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trendline.config import (
    ARTIFACT_DIR,
    CARDS_SECTOR_PATH,
    CARDS_SHARED_PATH,
    CARDS_STOCK_PATH,
    LEDGER_PATH,
    PAPER_FAMILY,
    PAPER_FEE_USD_PER_ORDER,
    PAPER_FX_HKD_PER_USD,
    PAPER_MAX_POSITIONS,
    PAPER_NOTIONAL_FRAC,
    PAPER_ORDERS_PER_ROUNDTRIP,
    PAPER_STARTING_HKD,
)
from trendline.data.store import load_ohlcv
from trendline.range_touch import fill_fade

FAMILY_PATHS = {
    "shared": CARDS_SHARED_PATH,
    "sector": CARDS_SECTOR_PATH,
    "stock": CARDS_STOCK_PATH,
}


def _empty_family() -> dict:
    return {
        "n_signals": 0,
        "n_fills": 0,
        "n_miss": 0,
        "n_wins": 0,
        "n_forecast": 0,
        "abs_high": 0.0,
        "abs_low": 0.0,
        "abs_close": 0.0,
    }


def new_ledger() -> dict:
    return {
        "account": {
            "name": "Kk paper",
            "started": "2026-09-13",
            "disclaimer": "模擬戶口，非真實下單。Paper book, not live broker orders.",
            "starting_equity_hkd": PAPER_STARTING_HKD,
            "equity_hkd": PAPER_STARTING_HKD,
            "cash_hkd": PAPER_STARTING_HKD,
            "fx_hkd_per_usd": PAPER_FX_HKD_PER_USD,
            "fee_usd_per_order": PAPER_FEE_USD_PER_ORDER,
            "orders_per_roundtrip": PAPER_ORDERS_PER_ROUNDTRIP,
            "traded_family": PAPER_FAMILY,
            "max_positions": PAPER_MAX_POSITIONS,
            "notional_frac": PAPER_NOTIONAL_FRAC,
        },
        "realized_asofs": [],
        "families": {fam: _empty_family() for fam in FAMILY_PATHS},
        "fills": [],
        "days": [],
        "updated_at_utc": None,
    }


def load_ledger(path: Path | None = None) -> dict:
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        return new_ledger()
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(ledger: dict, path: Path | None = None) -> Path:
    path = Path(path or LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_cards(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _next_bar(ohlcv: pd.DataFrame, ticker: str, asof: pd.Timestamp) -> pd.Series | None:
    g = ohlcv[(ohlcv["ticker"] == ticker) & (pd.to_datetime(ohlcv["date"]) > asof)]
    if g.empty:
        return None
    return g.sort_values("date").iloc[0]


def _refresh_fx(default: float) -> float:
    try:
        import yfinance as yf

        raw = yf.download("HKD=X", period="5d", progress=False, timeout=20)
        if raw is None or raw.empty:
            return default
        close = raw["Close"].dropna()
        if close.empty:
            return default
        px = float(close.iloc[-1])
        # Yahoo HKD=X is USD per HKD (~0.13) or HKD per USD depending on symbol.
        # HKD=X is typically HKD per 1 USD ≈ 7.8
        if 6.5 <= px <= 8.5:
            return px
        if 0.10 <= px <= 0.16:
            return 1.0 / px
    except Exception:
        return default
    return default


def _family_headline(fam: dict) -> dict:
    n = int(fam.get("n_forecast") or 0)
    fills = int(fam.get("n_fills") or 0)
    wins = int(fam.get("n_wins") or 0)
    return {
        "n_signals": fam.get("n_signals", 0),
        "n_fills": fills,
        "n_miss": fam.get("n_miss", 0),
        "hit_rate": (wins / fills) if fills else None,
        "mae_high": (fam["abs_high"] / n) if n else None,
        "mae_low": (fam["abs_low"] / n) if n else None,
        "mae_close": (fam["abs_close"] / n) if n else None,
        "n_forecast": n,
    }


def planned_orders(cards_payload: dict | None, equity_hkd: float, fx: float) -> list[dict]:
    if not cards_payload:
        return []
    actionable = [c for c in cards_payload.get("cards") or [] if c.get("side")]
    picks = actionable[:PAPER_MAX_POSITIONS]
    out = []
    for c in picks:
        entry = c.get("entry_px")
        if not entry or entry <= 0:
            continue
        notional_usd = (equity_hkd / fx) * PAPER_NOTIONAL_FRAC
        shares = int(math.floor(notional_usd / float(entry)))
        if shares < 1:
            continue
        out.append(
            {
                "ticker": c["ticker"],
                "side": int(c["side"]),
                "action": c.get("action"),
                "entry": float(entry),
                "tp": c.get("tp"),
                "sl": c.get("sl"),
                "shares": shares,
                "notional_usd": shares * float(entry),
                "fee_usd": PAPER_FEE_USD_PER_ORDER * PAPER_ORDERS_PER_ROUNDTRIP,
            }
        )
    return out


def _score_card(card: dict, bar: pd.Series) -> dict:
    prior = float(card["prior_close"])
    pred = card.get("pred") or {}
    rec = {
        "ticker": card["ticker"],
        "session": str(pd.Timestamp(bar["date"]).date()),
        "actual_high": float(bar["high"]),
        "actual_low": float(bar["low"]),
        "actual_close": float(bar["close"]),
        "err_high": abs(float(bar["high"]) - float((pred.get("high") or {}).get("q50") or prior)),
        "err_low": abs(float(bar["low"]) - float((pred.get("low") or {}).get("q50") or prior)),
        "err_close": abs(float(bar["close"]) - float((pred.get("close") or {}).get("q50") or prior)),
    }
    side = int(card.get("side") or 0)
    rec["side"] = side
    rec["fill"] = None
    if side and card.get("entry_px"):
        filled = fill_fade(
            side,
            float(card["entry_px"]),
            float(card["tp"]),
            float(card["sl"]),
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
        )
        if filled is None:
            rec["fill"] = "miss"
        else:
            ret, exit_px, reason = filled
            rec["fill"] = reason
            rec["ret"] = float(ret)
            rec["exit"] = float(exit_px)
            rec["entry"] = float(card["entry_px"])
    return rec


def realize_once(ledger: dict | None = None, ohlcv: pd.DataFrame | None = None) -> dict:
    """Score current on-disk cards against the next session in ohlcv. Idempotent per asof."""
    ledger = ledger or load_ledger()
    ohlcv = load_ohlcv() if ohlcv is None else ohlcv
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    shared_cards = _read_cards(FAMILY_PATHS["shared"])
    if not shared_cards:
        return ledger
    asof = pd.Timestamp(shared_cards["asof"])
    asof_s = str(asof.date())
    if asof_s in ledger.get("realized_asofs", []):
        return ledger

    # Need at least one ticker's next bar
    sample = _next_bar(ohlcv, (shared_cards.get("cards") or [{}])[0].get("ticker", ""), asof)
    if sample is None:
        return ledger
    session = str(pd.Timestamp(sample["date"]).date())

    fx = _refresh_fx(float(ledger["account"].get("fx_hkd_per_usd") or PAPER_FX_HKD_PER_USD))
    ledger["account"]["fx_hkd_per_usd"] = fx
    equity = float(ledger["account"]["equity_hkd"])
    fee_rt = PAPER_FEE_USD_PER_ORDER * PAPER_ORDERS_PER_ROUNDTRIP

    day = {"asof": asof_s, "session": session, "families": {}, "paper_pnl_hkd": 0.0, "paper_fills": 0}
    paper_pnl = 0.0
    paper_fills = 0

    for fam, path in FAMILY_PATHS.items():
        payload = _read_cards(path)
        if not payload:
            continue
        fam_state = ledger["families"].setdefault(fam, _empty_family())
        n_sig = n_fill = n_miss = n_win = 0
        abs_h = abs_l = abs_c = 0.0
        n_fc = 0
        for i, card in enumerate(payload.get("cards") or []):
            bar = _next_bar(ohlcv, card["ticker"], asof)
            if bar is None:
                continue
            scored = _score_card(card, bar)
            n_fc += 1
            abs_h += scored["err_high"]
            abs_l += scored["err_low"]
            abs_c += scored["err_close"]
            if scored["side"]:
                n_sig += 1
                if scored.get("fill") == "miss" or scored.get("fill") is None:
                    if scored.get("fill") == "miss":
                        n_miss += 1
                else:
                    n_fill += 1
                    if scored.get("ret", 0) > 0:
                        n_win += 1

            # Paper book: shared, first N actionable only
            if fam == PAPER_FAMILY and scored["side"] and scored.get("fill") not in (None,) and i < PAPER_MAX_POSITIONS:
                if scored.get("fill") == "miss":
                    continue
                entry = float(scored["entry"])
                notional_usd = (equity / fx) * PAPER_NOTIONAL_FRAC
                shares = int(math.floor(notional_usd / entry))
                if shares < 1:
                    continue
                side = scored["side"]
                exit_px = float(scored["exit"])
                gross_usd = shares * (exit_px - entry) * side
                pnl_usd = gross_usd - fee_rt
                pnl_hkd = pnl_usd * fx
                paper_pnl += pnl_hkd
                paper_fills += 1
                ledger["fills"].append(
                    {
                        "asof": asof_s,
                        "session": session,
                        "family": fam,
                        "ticker": card["ticker"],
                        "side": side,
                        "shares": shares,
                        "entry": entry,
                        "exit": exit_px,
                        "reason": scored["fill"],
                        "ret": scored.get("ret"),
                        "fee_usd": fee_rt,
                        "pnl_usd": pnl_usd,
                        "pnl_hkd": pnl_hkd,
                        "fx": fx,
                    }
                )

        fam_state["n_signals"] += n_sig
        fam_state["n_fills"] += n_fill
        fam_state["n_miss"] += n_miss
        fam_state["n_wins"] += n_win
        fam_state["n_forecast"] += n_fc
        fam_state["abs_high"] += abs_h
        fam_state["abs_low"] += abs_l
        fam_state["abs_close"] += abs_c
        day["families"][fam] = {
            "n_signals": n_sig,
            "n_fills": n_fill,
            "n_miss": n_miss,
            "n_wins": n_win,
            "n_forecast": n_fc,
            "mae_high": (abs_h / n_fc) if n_fc else None,
            "mae_low": (abs_l / n_fc) if n_fc else None,
            "mae_close": (abs_c / n_fc) if n_fc else None,
        }

    ledger["account"]["equity_hkd"] = equity + paper_pnl
    ledger["account"]["cash_hkd"] = ledger["account"]["equity_hkd"]
    day["paper_pnl_hkd"] = paper_pnl
    day["paper_fills"] = paper_fills
    day["equity_hkd"] = ledger["account"]["equity_hkd"]
    ledger["days"].append(day)
    ledger["realized_asofs"].append(asof_s)
    ledger["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger["headlines"] = {fam: _family_headline(ledger["families"][fam]) for fam in FAMILY_PATHS}
    return ledger


def update_ledger() -> dict:
    ledger = realize_once()
    save_ledger(ledger)
    print(
        f"ledger equity_hkd={ledger['account']['equity_hkd']:.2f} "
        f"asofs={ledger.get('realized_asofs')} fills={len(ledger.get('fills') or [])}"
    )
    return ledger


def ledger_view(ledger: dict | None = None) -> dict:
    ledger = ledger or load_ledger()
    cards = _read_cards(FAMILY_PATHS[PAPER_FAMILY])
    acct = ledger["account"]
    return {
        "ledger": ledger,
        "headlines": ledger.get("headlines")
        or {fam: _family_headline(ledger["families"][fam]) for fam in FAMILY_PATHS},
        "planned": planned_orders(cards, float(acct["equity_hkd"]), float(acct["fx_hkd_per_usd"])),
        "cards_asof": cards.get("asof") if cards else None,
    }
