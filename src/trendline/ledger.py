"""Paper ledger: realize last night's cards against the next session's bar.

Not a broker. Three HKD books (shared / sector / stock), each starts at the same capital.
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
    PAPER_BROKER,
    PAPER_DAILY_RISK_FRAC,
    PAPER_FAMILY,
    PAPER_FX_HKD_PER_USD,
    PAPER_GROSS_FRAC,
    PAPER_MAX_NAME_FRAC,
    PAPER_MAX_NAME_RISK,
    PAPER_MAX_POSITIONS,
    PAPER_MIN_NOTIONAL_USD,
    PAPER_STARTING_HKD,
)
from trendline.data.intraday import fetch_rth_5m
from trendline.data.store import load_ohlcv
from trendline.ibkr_fees import roundtrip_fees
from trendline.range_touch import FillResult, fill_fade, fill_fade_bars

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
        "starting_equity_hkd": PAPER_STARTING_HKD,
        "equity_hkd": PAPER_STARTING_HKD,
        "cash_hkd": PAPER_STARTING_HKD,
        "pnl_hkd": 0.0,
    }


def new_ledger() -> dict:
    return {
        "account": {
            "name": "Kk paper",
            "started": "2026-09-13",
            "disclaimer": "模擬戶口。按當日權益分倉（唔係開戶本金）；支出不可超過當日權益。即日平倉。",
            "starting_equity_hkd": PAPER_STARTING_HKD,
            "fx_hkd_per_usd": PAPER_FX_HKD_PER_USD,
            "broker": PAPER_BROKER,
            "traded_families": list(FAMILY_PATHS),
            "max_positions": PAPER_MAX_POSITIONS,
            "daily_risk_frac": PAPER_DAILY_RISK_FRAC,
            "max_name_risk": PAPER_MAX_NAME_RISK,
            "max_name_frac": PAPER_MAX_NAME_FRAC,
            "min_notional_usd": PAPER_MIN_NOTIONAL_USD,
        },
        "realized_asofs": [],
        "families": {fam: _empty_family() for fam in FAMILY_PATHS},
        "fills": [],
        "days": [],
        "updated_at_utc": None,
    }


def _ensure_books(ledger: dict) -> dict:
    """Upgrade a single-book ledger to three equal-start books (no fills yet / reset books)."""
    if not ledger.get("families"):
        ledger["families"] = {fam: _empty_family() for fam in FAMILY_PATHS}
    for fam in FAMILY_PATHS:
        book = ledger["families"].setdefault(fam, _empty_family())
        for k, v in _empty_family().items():
            book.setdefault(k, v)
        # Same starting line: if this book never traded, pin equity to the shared start.
        if not ledger.get("fills"):
            book["starting_equity_hkd"] = PAPER_STARTING_HKD
            book["equity_hkd"] = PAPER_STARTING_HKD
            book["cash_hkd"] = PAPER_STARTING_HKD
            book["pnl_hkd"] = 0.0
    acct = ledger.setdefault("account", {})
    acct["traded_families"] = list(FAMILY_PATHS)
    acct["starting_equity_hkd"] = PAPER_STARTING_HKD
    acct["broker"] = PAPER_BROKER
    acct["daily_risk_frac"] = PAPER_DAILY_RISK_FRAC
    acct["max_name_risk"] = PAPER_MAX_NAME_RISK
    acct.pop("traded_family", None)
    acct.pop("equity_hkd", None)
    acct.pop("cash_hkd", None)
    acct.pop("fee_usd_per_order", None)
    acct.pop("orders_per_roundtrip", None)
    return ledger


def load_ledger(path: Path | None = None) -> dict:
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        return new_ledger()
    return _ensure_books(json.loads(path.read_text(encoding="utf-8")))


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
    start = float(fam.get("starting_equity_hkd") or PAPER_STARTING_HKD)
    eq = float(fam.get("equity_hkd") or start)
    return {
        "n_signals": fam.get("n_signals", 0),
        "n_fills": fills,
        "n_miss": fam.get("n_miss", 0),
        "hit_rate": (wins / fills) if fills else None,
        "mae_high": (fam["abs_high"] / n) if n else None,
        "mae_low": (fam["abs_low"] / n) if n else None,
        "mae_close": (fam["abs_close"] / n) if n else None,
        "n_forecast": n,
        "equity_hkd": eq,
        "pnl_hkd": eq - start,
        "ret": (eq / start - 1.0) if start else None,
    }


def fade_score(card: dict) -> float:
    """Confidence = fade room back to prior close, in ATR units."""
    try:
        entry = float(card.get("entry_px") or 0)
        prior = float(card.get("prior_close") or 0)
        atr = float(card.get("atr") or 0)
        side = int(card.get("side") or 0)
    except (TypeError, ValueError):
        return 0.0
    if entry <= 0 or prior <= 0 or atr <= 0 or side == 0:
        return 0.0
    room = (prior - entry) if side == 1 else (entry - prior)
    if room <= 0:
        return 0.0
    return room / atr


def _sl_distance(card: dict) -> float:
    try:
        entry = float(card.get("entry_px") or 0)
        sl = float(card.get("sl") or 0)
    except (TypeError, ValueError):
        return 0.0
    if entry <= 0 or sl <= 0:
        return 0.0
    return abs(entry - sl)


def _size_book(ranked: list, equity_usd: float) -> list[dict]:
    """Risk-first: fade-score shares a 5% daily SL budget; 12% name notional cap."""
    if not ranked or equity_usd <= 0:
        return []
    score_sum = sum(s for s, _ in ranked)
    if score_sum <= 0:
        return []
    raw = []
    for s, c in ranked:
        dist = _sl_distance(c)
        entry = float(c["entry_px"])
        if dist <= 1e-9 or entry <= 0:
            continue
        risk_frac = min(PAPER_MAX_NAME_RISK, PAPER_DAILY_RISK_FRAC * (s / score_sum))
        raw.append((s, c, entry, dist, risk_frac))
    tot = sum(r[-1] for r in raw)
    if tot > PAPER_DAILY_RISK_FRAC and tot > 0:
        scale = PAPER_DAILY_RISK_FRAC / tot
        raw = [(s, c, e, d, rf * scale) for s, c, e, d, rf in raw]
    out = []
    for s, c, entry, dist, risk_frac in raw:
        risk_usd = equity_usd * risk_frac
        shares = int(math.floor(risk_usd / dist))
        cap_shares = int(math.floor(equity_usd * PAPER_MAX_NAME_FRAC / entry))
        shares = min(shares, cap_shares)
        if shares < 1:
            continue
        notional = shares * entry
        if notional < PAPER_MIN_NOTIONAL_USD:
            continue
        out.append(
            {
                "ticker": c["ticker"],
                "side": int(c["side"]),
                "action": c.get("action"),
                "entry": entry,
                "tp": c.get("tp"),
                "sl": c.get("sl"),
                "shares": shares,
                "notional_usd": notional,
                "weight": notional / equity_usd,
                "risk_frac": shares * dist / equity_usd,
                "score": s,
                "fee_usd": roundtrip_fees(int(c["side"]), shares, entry, float(c.get("tp") or entry))["total"],
            }
        )
    def _spend(rows):
        return sum(float(r["notional_usd"]) + float(r["fee_usd"]) for r in rows)

    # Today's equity only — not the opening HK$500k.
    while len(out) > 1 and _spend(out) > equity_usd + 1e-6:
        out = out[:-1]
    if out and _spend(out) > equity_usd + 1e-6:
        row = out[0]
        entry = float(row["entry"])
        dist = abs(entry - float(row["sl"] or entry))
        budget = max(0.0, equity_usd - float(row["fee_usd"]))
        cap_shares = int(math.floor(budget / entry)) if entry else 0
        row["shares"] = max(0, min(int(row["shares"]), cap_shares))
        if row["shares"] < 1 or row["shares"] * entry < PAPER_MIN_NOTIONAL_USD:
            return []
        row["notional_usd"] = row["shares"] * entry
        row["weight"] = row["notional_usd"] / equity_usd
        row["risk_frac"] = (row["shares"] * dist / equity_usd) if dist else 0.0
        row["fee_usd"] = roundtrip_fees(int(row["side"]), row["shares"], entry, float(row.get("tp") or entry))["total"]
        if _spend(out) > equity_usd + 1e-6:
            return []
    return out


def planned_orders(cards_payload: dict | None, equity_hkd: float, fx: float) -> list[dict]:
    """Size so a stop costs ~score-weighted share of a 5% daily risk budget."""
    if not cards_payload or fx <= 0:
        return []
    ranked = []
    for c in cards_payload.get("cards") or []:
        if not c.get("side") or not c.get("entry_px"):
            continue
        s = fade_score(c)
        if s > 0 and _sl_distance(c) > 0:
            ranked.append((s, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[:PAPER_MAX_POSITIONS]
    equity_usd = float(equity_hkd) / float(fx)
    while ranked:
        out = _size_book(ranked, equity_usd)
        kept = {r["ticker"] for r in out}
        if kept == {c["ticker"] for _, c in ranked} or (out and len(out) < len(ranked) and sum(r["weight"] for r in out) <= PAPER_GROSS_FRAC + 1e-9):
            return out
        ranked = ranked[:-1]
    return []


def _bars_usable(bars) -> bool:
    if bars is None:
        return False
    try:
        return len(bars) > 0
    except TypeError:
        return False


def _score_card(card: dict, bar: pd.Series, bars=None) -> dict:
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
    rec["fill_source"] = None
    if side and card.get("entry_px"):
        entry = float(card["entry_px"])
        tp = float(card["tp"])
        sl = float(card["sl"])
        if _bars_usable(bars):
            filled = fill_fade_bars(side, entry, tp, sl, bars)
            fill_source = "5m"
        else:
            filled = fill_fade(
                side,
                entry,
                tp,
                sl,
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
            )
            fill_source = "daily"
        rec["fill_source"] = fill_source
        if filled is None:
            rec["fill"] = "miss"
            rec["entry_ts"] = None
            rec["exit_ts"] = None
        else:
            if not isinstance(filled, FillResult):
                ret, exit_px, reason = filled
                filled = FillResult(float(ret), float(exit_px), str(reason))
            rec["fill"] = filled.reason
            rec["ret"] = float(filled.ret)
            rec["exit"] = float(filled.exit_px)
            rec["entry"] = entry
            rec["entry_ts"] = filled.entry_ts
            rec["exit_ts"] = filled.exit_ts
    return rec


def realize_once(
    ledger: dict | None = None,
    ohlcv: pd.DataFrame | None = None,
    bars_by_ticker: dict | None = None,
) -> dict:
    """Score current on-disk cards against the next session in ohlcv. Idempotent per asof.

    Paper fills prefer America/New_York RTH 5-minute bars; if 5m is missing for a
    ticker, fall back to daily OHLC ``fill_fade``. Pass ``bars_by_ticker`` to inject
    bars (tests) and skip Yahoo.
    """
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
    ledger["account"]["broker"] = PAPER_BROKER
    ledger = _ensure_books(ledger)

    # Collect signal tickers once; fetch 5m for the session (or use inject).
    side_tickers: set[str] = set()
    payloads: dict[str, dict] = {}
    for fam, path in FAMILY_PATHS.items():
        payload = _read_cards(path)
        if not payload:
            continue
        payloads[fam] = payload
        for card in payload.get("cards") or []:
            if int(card.get("side") or 0) and card.get("entry_px"):
                side_tickers.add(card["ticker"])
    if bars_by_ticker is None:
        try:
            bars_by_ticker = fetch_rth_5m(sorted(side_tickers), session) if side_tickers else {}
        except Exception:
            bars_by_ticker = {}

    day = {"asof": asof_s, "session": session, "families": {}, "equity_hkd": {}}

    for fam, path in FAMILY_PATHS.items():
        payload = payloads.get(fam) or _read_cards(path)
        if not payload:
            continue
        fam_state = ledger["families"].setdefault(fam, _empty_family())
        equity = float(fam_state.get("equity_hkd") or PAPER_STARTING_HKD)
        plan = {p["ticker"]: p for p in planned_orders(payload, equity, fx)}
        n_sig = n_fill = n_miss = n_win = 0
        abs_h = abs_l = abs_c = 0.0
        n_fc = 0
        book_pnl = 0.0
        book_fills = 0
        for i, card in enumerate(payload.get("cards") or []):
            bar = _next_bar(ohlcv, card["ticker"], asof)
            if bar is None:
                continue
            bars = (bars_by_ticker or {}).get(card["ticker"])
            scored = _score_card(card, bar, bars=bars)
            n_fc += 1
            abs_h += scored["err_high"]
            abs_l += scored["err_low"]
            abs_c += scored["err_close"]
            if scored["side"]:
                n_sig += 1
                if scored.get("fill") == "miss":
                    n_miss += 1
                elif scored.get("fill"):
                    n_fill += 1
                    if scored.get("ret", 0) > 0:
                        n_win += 1

            if card["ticker"] in plan and scored.get("fill") not in (None, "miss"):
                entry = float(scored["entry"])
                shares = int(plan[card["ticker"]]["shares"])
                if shares < 1:
                    continue
                side = scored["side"]
                exit_px = float(scored["exit"])
                fees = roundtrip_fees(side, shares, entry, exit_px)
                fee_rt = float(fees["total"])
                pnl_usd = shares * (exit_px - entry) * side - fee_rt
                pnl_hkd = pnl_usd * fx
                book_pnl += pnl_hkd
                book_fills += 1
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
                        "entry_ts": scored.get("entry_ts"),
                        "exit_ts": scored.get("exit_ts"),
                        "reason": scored["fill"],
                        "fill_source": scored.get("fill_source") or "daily",
                        "ret": scored.get("ret"),
                        "fee_usd": fee_rt,
                        "commission_usd": fees["commission"],
                        "regulatory_usd": fees["regulatory"],
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
        fam_state["equity_hkd"] = equity + book_pnl
        fam_state["cash_hkd"] = fam_state["equity_hkd"]
        fam_state["pnl_hkd"] = float(fam_state["equity_hkd"]) - float(fam_state.get("starting_equity_hkd") or PAPER_STARTING_HKD)
        day["families"][fam] = {
            "n_signals": n_sig,
            "n_fills": n_fill,
            "n_miss": n_miss,
            "n_wins": n_win,
            "n_forecast": n_fc,
            "paper_fills": book_fills,
            "paper_pnl_hkd": book_pnl,
            "equity_hkd": fam_state["equity_hkd"],
            "mae_high": (abs_h / n_fc) if n_fc else None,
            "mae_low": (abs_l / n_fc) if n_fc else None,
            "mae_close": (abs_c / n_fc) if n_fc else None,
        }
        day["equity_hkd"][fam] = fam_state["equity_hkd"]

    ledger["days"].append(day)
    ledger["realized_asofs"].append(asof_s)
    ledger["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger["headlines"] = {fam: _family_headline(ledger["families"][fam]) for fam in FAMILY_PATHS}
    return ledger


def update_ledger() -> dict:
    ledger = realize_once()
    save_ledger(ledger)
    eq = {fam: ledger["families"][fam]["equity_hkd"] for fam in FAMILY_PATHS}
    print(f"ledger equities={eq} asofs={ledger.get('realized_asofs')} fills={len(ledger.get('fills') or [])}")
    return ledger


def ledger_view(ledger: dict | None = None) -> dict:
    ledger = _ensure_books(ledger or load_ledger())
    fx = float(ledger["account"]["fx_hkd_per_usd"])
    planned = {}
    asofs = {}
    for fam, path in FAMILY_PATHS.items():
        cards = _read_cards(path)
        asofs[fam] = cards.get("asof") if cards else None
        eq = float(ledger["families"][fam]["equity_hkd"])
        planned[fam] = planned_orders(cards, eq, fx)
    return {
        "ledger": ledger,
        "headlines": {fam: _family_headline(ledger["families"][fam]) for fam in FAMILY_PATHS},
        "planned": planned,
        "cards_asof": asofs,
    }
