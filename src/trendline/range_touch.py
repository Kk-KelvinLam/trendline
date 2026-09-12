"""Intraday fade: touch predicted High/Low, take profit at prior close.

High-win-rate setup. Close direction is not used.
Conservative daily-OHLC fill: if stop and target both print, stop wins.
Skip if the next open already gapped through the trigger (no chase).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trendline.config import ATR_SL_MULT, FADE_MIN_ATR, RANGE_ATR_MIN


@dataclass(frozen=True)
class FadeSetup:
    side: int  # +1 long, -1 short, 0 none
    entry: float
    tp: float
    sl: float
    room: float
    reason: str


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
) -> FadeSetup:
    """Pick the fade side with more room back to prior close."""
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

    if long_ok and (not short_ok or lroom >= sroom):
        return FadeSetup(1, float(le), float(ltp), float(lsl), float(lroom), "fade_to_prior_close")
    if short_ok:
        return FadeSetup(-1, float(se), float(stp), float(ssl), float(sroom), "fade_to_prior_close")
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
) -> tuple[float, float, str] | None:
    """Return (ret, exit, reason) or None if no fill / gapped through."""
    if side == 0 or gapped_through(side, next_open, entry):
        return None
    if side == 1:
        if next_low > entry:
            return None
        hit_sl = next_low <= sl
        hit_tp = next_high >= tp
        if hit_sl:
            return sl / entry - 1.0, sl, "sl"
        if hit_tp:
            return tp / entry - 1.0, tp, "tp"
        return next_close / entry - 1.0, next_close, "close"
    if next_high < entry:
        return None
    hit_sl = next_high >= sl
    hit_tp = next_low <= tp
    if hit_sl:
        return (entry - sl) / entry, sl, "sl"
    if hit_tp:
        return (entry - tp) / entry, tp, "tp"
    return (entry - next_close) / entry, next_close, "close"
