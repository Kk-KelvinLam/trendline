"""Intraday fade: touch predicted High/Low, take profit at prior close.

High-win-rate setup. Close q50 gates fade side: long needs pred close >= prior,
short needs pred close <= prior (fade room alone is not enough).
Paper ledger prefers RTH 5-minute bars (``fill_fade_bars``); daily OHLC
``fill_fade`` remains the walk-forward / missing-5m fallback.
Conservative fill: if stop and target both print in the same bar, stop wins.
If the trigger fills but neither TP nor SL prints, flatten at the last bar close
(same-day book; no overnight). Skip if the session open gapped through the trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd

from trendline.config import ATR_SL_MULT, CLOSE_GATE_MIN_RET, FADE_MIN_ATR, RANGE_ATR_MIN


@dataclass(frozen=True)
class FadeSetup:
    side: int  # +1 long, -1 short, 0 none
    entry: float
    tp: float
    sl: float
    room: float
    reason: str

@dataclass(frozen=True)
class FillResult:
    ret: float
    exit_px: float
    reason: str
    entry_ts: str | None = None  # America/New_York ISO when known (5m)
    exit_ts: str | None = None


def _fmt_ts(ts) -> str | None:
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return str(ts)
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York")
        return t.isoformat()
    return t.isoformat()



def predicted_range(close: float, q90h: float, q10l: float) -> float:
    return (q90h - q10l) * close


def range_too_tight(close: float, atr: float, q90h: float, q10l: float) -> bool:
    rng = predicted_range(close, q90h, q10l)
    return (not np.isfinite(rng)) or (not np.isfinite(atr)) or (rng < RANGE_ATR_MIN * atr)


def _long_levels(close: float, atr: float, q50l: float, q10l: float) -> tuple[float, float, float, float]:
    entry = close * (1.0 + q50l)
    tp = close
    sl = min(close * (1.0 + q10l), entry - ATR_SL_MULT * atr)
    room = close - entry
    return entry, tp, sl, room


def _short_levels(close: float, atr: float, q50h: float, q90h: float) -> tuple[float, float, float, float]:
    entry = close * (1.0 + q50h)
    tp = close
    sl = max(close * (1.0 + q90h), entry + ATR_SL_MULT * atr)
    room = entry - close
    return entry, tp, sl, room


def choose_setup(
    close: float,
    atr: float,
    q50h: float,
    q50l: float,
    q10l: float,
    q90h: float,
    allowed: bool,
    q50c: float | None = None,
) -> FadeSetup:
    """Pick fade side with more room back to prior close, gated by Close q50.

    Long only if Close q50 is at least +CLOSE_GATE_MIN_RET above prior.
    Short only if Close q50 is at least CLOSE_GATE_MIN_RET below prior.
    Near-zero Close predictions are treated as no edge (cannot open either side via Close gate).
    """
    if not allowed:
        return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "range_model_does_not_beat_baseline")
    if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
        return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "bad_price")
    if range_too_tight(close, atr, q90h, q10l):
        return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "range_too_tight")

    le, ltp, lsl, lroom = _long_levels(close, atr, q50l, q10l)
    se, stp, ssl, sroom = _short_levels(close, atr, q50h, q90h)
    min_room = FADE_MIN_ATR * atr
    long_ok = np.isfinite(le) and lroom >= min_room and lsl < le < close
    short_ok = np.isfinite(se) and sroom >= min_room and close < se < ssl

    close_gate = q50c is not None and np.isfinite(q50c)
    if close_gate:
        # Require meaningful |Close| edge — tiny q50c is noise, not permission to fade.
        long_ok = long_ok and float(q50c) >= float(CLOSE_GATE_MIN_RET)
        short_ok = short_ok and float(q50c) <= -float(CLOSE_GATE_MIN_RET)

    if long_ok and (not short_ok or lroom >= sroom):
        return FadeSetup(1, float(le), float(ltp), float(lsl), float(lroom), "fade_to_prior_close")
    if short_ok:
        return FadeSetup(-1, float(se), float(stp), float(ssl), float(sroom), "fade_to_prior_close")
    if close_gate and (np.isfinite(le) or np.isfinite(se)):
        # Had a room-eligible side that Close vetoed (or both sides vetoed).
        return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "close_disagrees_with_fade")
    return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "fade_room_too_small")


def gapped_through(side: int, next_open: float, entry: float) -> bool:
    if not np.isfinite(next_open) or next_open <= 0:
        return True
    if side == 1:
        return next_open <= entry
    if side == -1:
        return next_open >= entry
    return True


def fill_fade(
    side: int,
    entry: float,
    tp: float,
    sl: float,
    next_open: float,
    next_high: float,
    next_low: float,
    next_close: float,
) -> FillResult | None:
    """Daily OHLC fade fill. No intraday timestamps (entry_ts/exit_ts left None)."""
    if side == 0 or gapped_through(side, next_open, entry):
        return None
    if side == 1:
        if next_low > entry:
            return None
        hit_sl = next_low <= sl
        hit_tp = next_high >= tp
        if hit_sl:
            return FillResult(sl / entry - 1.0, sl, "sl")
        if hit_tp:
            return FillResult(tp / entry - 1.0, tp, "tp")
        return FillResult(next_close / entry - 1.0, next_close, "close")
    if next_high < entry:
        return None
    hit_sl = next_high >= sl
    hit_tp = next_low <= tp
    if hit_sl:
        return FillResult((entry - sl) / entry, sl, "sl")
    if hit_tp:
        return FillResult((entry - tp) / entry, tp, "tp")
    return FillResult((entry - next_close) / entry, next_close, "close")


def _ohlc(bar) -> tuple[float, float, float, float]:
    if isinstance(bar, dict):
        return float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    return float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])



def _as_bar_seq(bars) -> list[tuple]:
    """Normalize to list of (timestamp|None, OHLC row). Avoid list(df)=columns."""
    if bars is None:
        return []
    if hasattr(bars, "iterrows"):
        return [(idx, row) for idx, row in bars.iterrows()]
    if hasattr(bars, "iloc") and not isinstance(bars, (list, tuple, dict)):
        out = []
        for i in range(len(bars)):
            row = bars.iloc[i]
            idx = bars.index[i] if hasattr(bars, "index") else None
            out.append((idx, row))
        return out
    seq = []
    for item in list(bars):
        if isinstance(item, (tuple, list)) and len(item) == 2 and not isinstance(item[0], (int, float)):
            seq.append((item[0], item[1]))
        else:
            seq.append((None, item))
    return seq


def fill_fade_bars(
    side: int,
    entry: float,
    tp: float,
    sl: float,
    bars,
) -> FillResult | None:
    """Walk RTH (or any ordered) OHLC bars for a same-session fade fill.

    Gap-through uses the *first* bar open (session open). Until filled, long
    triggers on low<=entry / short on high>=entry at price=entry. After fill
    (including the fill bar): SL if hit else TP; both in the same bar → SL.
    Still open on the last bar → flatten at that close (reason=close).
    Prints of TP/SL *before* the entry fill are ignored.

    Returns ``FillResult`` with America/New_York ``entry_ts`` / ``exit_ts`` when
    the bar index carries timestamps (Yahoo 5m).
    """
    if side == 0:
        return None
    seq = _as_bar_seq(bars)
    if not seq:
        return None

    first_ts, first_bar = seq[0]
    first_open, _, _, _ = _ohlc(first_bar)
    if gapped_through(side, first_open, entry):
        return None

    def _done(ret: float, exit_px: float, reason: str, entry_ts, exit_ts) -> FillResult:
        return FillResult(ret, exit_px, reason, _fmt_ts(entry_ts), _fmt_ts(exit_ts))

    filled = False
    entry_ts = None
    for ts, bar in seq:
        _o, h, l, _c = _ohlc(bar)
        if not filled:
            if side == 1:
                if l > entry:
                    continue
                filled = True
            else:
                if h < entry:
                    continue
                filled = True
            entry_ts = ts
            if side == 1:
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp
            if hit_sl:
                ret = (sl / entry - 1.0) if side == 1 else (entry - sl) / entry
                return _done(ret, sl, "sl", entry_ts, ts)
            if hit_tp:
                ret = (tp / entry - 1.0) if side == 1 else (entry - tp) / entry
                return _done(ret, tp, "tp", entry_ts, ts)
            continue

        if side == 1:
            hit_sl = l <= sl
            hit_tp = h >= tp
            if hit_sl:
                return _done(sl / entry - 1.0, sl, "sl", entry_ts, ts)
            if hit_tp:
                return _done(tp / entry - 1.0, tp, "tp", entry_ts, ts)
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp
            if hit_sl:
                return _done((entry - sl) / entry, sl, "sl", entry_ts, ts)
            if hit_tp:
                return _done((entry - tp) / entry, tp, "tp", entry_ts, ts)

    if not filled:
        return None
    last_ts, last_bar = seq[-1]
    last_close = _ohlc(last_bar)[3]
    if side == 1:
        return _done(last_close / entry - 1.0, last_close, "close", entry_ts, last_ts)
    return _done((entry - last_close) / entry, last_close, "close", entry_ts, last_ts)
