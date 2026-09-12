"""Next-session trade cards. Long/short only if the model beat baseline OOS."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trendline.config import (
    ATR_SL_MULT,
    DIR_RET_MIN,
    RANGE_ATR_MIN,
    SHOW_TOP_N,
)
from trendline.models.baseline import BaselineModel
from trendline.models.lightgbm_quantile import QuantileLGBM
from trendline.universe import rank_by_dollar_volume

SOURCE_LABEL = {
    "yfinance": "Yahoo Finance",
    "yahoo": "Yahoo Finance",
    "stooq": "Stooq",
    "unknown": "未知",
}


def _source_label(raw: str | None) -> str:
    key = (raw or "unknown").strip().lower()
    return SOURCE_LABEL.get(key, raw or "未知")


def _ticker_source(ohlcv: pd.DataFrame, ticker: str, asof: pd.Timestamp) -> str:
    if ohlcv is None or ohlcv.empty or "source" not in ohlcv.columns:
        return "未知"
    g = ohlcv[(ohlcv["ticker"] == ticker) & (pd.to_datetime(ohlcv["date"]) == asof)]
    if g.empty:
        g = ohlcv[ohlcv["ticker"] == ticker]
    if g.empty:
        return "未知"
    return _source_label(str(g.iloc[-1]["source"]))


def _recent_error(oos: pd.DataFrame | None, ticker: str, n: int = 20) -> dict:
    if oos is None or getattr(oos, "empty", True):
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None}
    g = oos.loc[oos["ticker"] == ticker].sort_values("date").tail(n)
    if g.empty:
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None}
    err = (g["y_close"] - g["pred_close_q50"]).abs()
    px_err = (g["close"] * err).abs()
    return {
        "n": int(len(g)),
        "mae_close_ret": float(err.mean()),
        "mae_close_px": float(px_err.mean()),
    }


def build_cards(
    featured: pd.DataFrame,
    ohlcv: pd.DataFrame,
    oos: pd.DataFrame,
    per_ticker: pd.DataFrame,
    overall_beats: bool,
    model: QuantileLGBM | None = None,
    asof: pd.Timestamp | None = None,
) -> dict:
    """Score the latest completed session and emit dashboard cards."""
    feat = featured.copy()
    feat["date"] = pd.to_datetime(feat["date"])
    if asof is None:
        # Latest date that still has a full feature vector (target may be NaN — that's the live row).
        ready = feat.dropna(subset=["ret_20d", "atr", "dist_ma50"])
        asof = ready["date"].max()
    asof = pd.Timestamp(asof)
    day = feat[feat["date"] == asof].copy()
    if day.empty:
        raise RuntimeError(f"no feature rows on {asof.date()}")

    ranked = rank_by_dollar_volume(ohlcv, asof=asof, top_n=SHOW_TOP_N)
    show = set(ranked["ticker"])
    day = day[day["ticker"].isin(show)].copy()
    if day.empty:
        raise RuntimeError("no S&P names in the top dollar-volume set for this session")

    if model is None:
        model = QuantileLGBM().load()
    preds = model.predict(day)
    base = BaselineModel().predict(day)
    day = pd.concat([day.reset_index(drop=True), preds.reset_index(drop=True), base.reset_index(drop=True)], axis=1)

    beat = dict(zip(per_ticker["ticker"], per_ticker["beats_baseline"], strict=False))
    rank_map = dict(zip(ranked["ticker"], ranked["dvol_rank"], strict=False))
    dvol_map = dict(zip(ranked["ticker"], ranked["dollar_volume"], strict=False))

    cards = []
    for row in day.itertuples(index=False):
        ticker = row.ticker
        allowed = bool(beat.get(ticker, False) or overall_beats)
        close = float(row.close)
        atr = float(row.atr) if np.isfinite(row.atr) else float("nan")
        q50c = float(row.pred_close_q50)
        q50h = float(row.pred_high_q50)
        q50l = float(row.pred_low_q50)
        q10l = float(row.pred_low_q10)
        q90h = float(row.pred_high_q90)
        rng = (q90h - q10l) * close
        tight = (not np.isfinite(rng)) or (not np.isfinite(atr)) or (rng < RANGE_ATR_MIN * atr)
        weak = abs(q50c) < DIR_RET_MIN

        action = "觀望"
        side = 0
        reason = "weak_or_tight"
        if not allowed:
            reason = "model_does_not_beat_baseline"
        elif tight:
            reason = "range_too_tight"
        elif weak:
            reason = "probability_too_close"
        elif q50c > 0:
            action, side, reason = "做多", 1, "model_beats_baseline"
        elif q50c < 0:
            action, side, reason = "做空", -1, "model_beats_baseline"

        # Entry is the next regular open (unknown until the bell). Levels use prior close / ATR.
        tp = sl = None
        if side == 1:
            tp = close * (1.0 + q50h)
            sl = max(close * (1.0 + q10l), close - ATR_SL_MULT * atr)
        elif side == -1:
            tp = close * (1.0 + q50l)
            sl = min(close * (1.0 + q90h), close + ATR_SL_MULT * atr)

        cards.append(
            {
                "ticker": ticker,
                "name": getattr(row, "sector", None),
                "sector": getattr(row, "sector", None),
                "dvol_rank": int(rank_map.get(ticker, 0) or 0),
                "dollar_volume": float(dvol_map.get(ticker, 0) or 0),
                "action": action,
                "side": side,
                "reason": reason,
                "asof": str(asof.date()),
                "prior_close": close,
                "entry": "下一開市",
                "tp": None if tp is None else float(tp),
                "sl": None if sl is None else float(sl),
                "pred": {
                    "high": {
                        "q10": float(close * (1 + row.pred_high_q10)),
                        "q50": float(close * (1 + q50h)),
                        "q90": float(close * (1 + row.pred_high_q90)),
                        "q10_ret": float(row.pred_high_q10),
                        "q50_ret": float(q50h),
                        "q90_ret": float(row.pred_high_q90),
                    },
                    "low": {
                        "q10": float(close * (1 + q10l)),
                        "q50": float(close * (1 + q50l)),
                        "q90": float(close * (1 + row.pred_low_q90)),
                        "q10_ret": float(q10l),
                        "q50_ret": float(q50l),
                        "q90_ret": float(row.pred_low_q90),
                    },
                    "close": {
                        "q10": float(close * (1 + row.pred_close_q10)),
                        "q50": float(close * (1 + q50c)),
                        "q90": float(close * (1 + row.pred_close_q90)),
                        "q10_ret": float(row.pred_close_q10),
                        "q50_ret": float(q50c),
                        "q90_ret": float(row.pred_close_q90),
                    },
                },
                "baseline": {
                    "high_q50": float(close * (1 + row.base_high_q50)),
                    "low_q50": float(close * (1 + row.base_low_q50)),
                    "close_q50": float(close * (1 + row.base_close_q50)),
                },
                "atr": float(atr) if np.isfinite(atr) else None,
                "high_confidence": bool(allowed and (not tight) and abs(q50c) >= 2 * DIR_RET_MIN),
                "beats_baseline": bool(beat.get(ticker, False)),
                "recent_error": _recent_error(oos, ticker),
                "data_source": _ticker_source(ohlcv, ticker, asof),
                "universe_source": "S&P 500 · Wikipedia 2026-09-12",
            }
        )

    cards.sort(key=lambda c: (0 if c["action"] != "觀望" else 1, c["dvol_rank"]))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof": str(asof.date()),
        "next_session": "下一常規交易時段",
        "n_cards": len(cards),
        "n_long": sum(1 for c in cards if c["action"] == "做多"),
        "n_short": sum(1 for c in cards if c["action"] == "做空"),
        "n_flat": sum(1 for c in cards if c["action"] == "觀望"),
        "disclaimer_zh": "本頁為量化模型輸出，並非投資建議。過往回測不代表未來表現。",
        "disclaimer_en": "Model output, not investment advice. Past backtests do not predict future results.",
        "universe_source": "S&P 500 · Wikipedia 2026-09-12",
        "cards": cards,
    }
    return payload
