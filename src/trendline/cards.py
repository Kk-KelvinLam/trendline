"""Next-session trade cards. Long/short only if the model beat baseline OOS."""

from __future__ import annotations

from typing import Protocol

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trendline.config import FADE_MIN_ATR, SHOW_TOP_N
from trendline.range_touch import choose_setup
from trendline.models.baseline import BaselineModel
from trendline.models.families import SectorBundle, SharedBundle, StockBundle, pred_col
from trendline.models.lightgbm_quantile import QuantileLGBM
from trendline.universe import load_sp500, rank_by_dollar_volume

SOURCE_LABEL = {
    "yfinance": "Yahoo Finance",
    "yahoo": "Yahoo Finance",
    "stooq": "Stooq",
    "unknown": "未知",
}


class Predictor(Protocol):
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame: ...


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


def _recent_error(oos: pd.DataFrame | None, ticker: str, family: str = "shared", n: int = 20) -> dict:
    if oos is None or getattr(oos, "empty", True):
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None}
    g = oos.loc[oos["ticker"] == ticker].sort_values("date").tail(n)
    if g.empty:
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None}
    col = pred_col("close", 0.50, family)
    if col not in g.columns:
        col = pred_col("close", 0.50)
    if col not in g.columns:
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None}
    err = (g["y_close"] - g[col]).abs()
    px_err = (g["close"] * err).abs()
    return {
        "n": int(len(g)),
        "mae_close_ret": float(err.mean()),
        "mae_close_px": float(px_err.mean()),
    }


class FamilyPredictor:
    """Wraps shared/sector/stock bundles so cards always see unsuffixed pred_* cols."""

    def __init__(
        self,
        family: str,
        shared: SharedBundle,
        sector: SectorBundle | None = None,
        stock: StockBundle | None = None,
    ) -> None:
        self.family = family
        self.shared = shared
        self.sector = sector
        self.stock = stock

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        shared_preds = self.shared.predict(frame)
        if self.family == "shared":
            return shared_preds
        if self.family == "sector":
            if self.sector is None:
                return shared_preds
            return self.sector.predict(frame, fallback=shared_preds)
        if self.family == "stock":
            if self.stock is None:
                return shared_preds
            return self.stock.predict(frame, fallback=shared_preds)
        raise ValueError(f"unknown family {self.family}")


def _naive_midnight(values) -> pd.Series | pd.Timestamp:
    """Normalize dates to tz-naive midnight for reliable equality joins."""
    if isinstance(values, pd.Series):
        out = pd.to_datetime(values)
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        return out.dt.normalize()
    ts = pd.Timestamp(values)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def latest_card_asof(featured: pd.DataFrame) -> pd.Timestamp:
    """Latest session with ready *S&P* features — skip macro-only / thin days.

    yfinance often publishes ^VIX on holidays or partial-fetch days when no
    equity bars landed. Taking max(ready date) then intersects empty with the
    S&P dollar-volume shortlist.
    """
    feat = featured.copy()
    feat["date"] = _naive_midnight(feat["date"])
    ready = feat.dropna(subset=["ret_20d", "atr", "dist_ma50"])
    members = set(load_sp500()["ticker"])
    ready_sp = ready[ready["ticker"].isin(members)]
    if ready_sp.empty:
        raise RuntimeError("no S&P names with ready features for card asof")
    return pd.Timestamp(ready_sp["date"].max())


def build_cards(
    featured: pd.DataFrame,
    ohlcv: pd.DataFrame,
    oos: pd.DataFrame,
    per_ticker: pd.DataFrame,
    overall_beats: bool,
    model: Predictor | QuantileLGBM | None = None,
    asof: pd.Timestamp | None = None,
    family: str = "shared",
) -> dict:
    """Score the latest completed session and emit dashboard cards for one family."""
    feat = featured.copy()
    feat["date"] = _naive_midnight(feat["date"])
    if asof is None:
        asof = latest_card_asof(feat)
    asof = _naive_midnight(asof)
    day = feat[feat["date"] == asof].copy()
    if day.empty:
        raise RuntimeError(f"no feature rows on {asof.date()}")

    ranked = rank_by_dollar_volume(ohlcv, asof=asof, top_n=SHOW_TOP_N)
    show = set(ranked["ticker"])
    day = day[day["ticker"].isin(show)].copy()
    if day.empty:
        raise RuntimeError(
            f"no S&P names in the top dollar-volume set for session {asof.date()} "
            f"(ranked={len(ranked)}; macro-only/thin session?)"
        )

    if model is None:
        shared = SharedBundle().load()
        sector = SectorBundle().load() if family != "shared" else None
        stock = StockBundle().load() if family == "stock" else None
        if family == "sector" and sector is None:
            sector = SectorBundle().load()
        model = FamilyPredictor(family, shared, sector, stock)

    preds = model.predict(day)
    base = BaselineModel().predict(day)
    day = pd.concat([day.reset_index(drop=True), preds.reset_index(drop=True), base.reset_index(drop=True)], axis=1)

    beat_col = "beats_range" if "beats_range" in per_ticker.columns else "beats_baseline"
    beat = dict(zip(per_ticker["ticker"], per_ticker[beat_col], strict=False))
    rank_map = dict(zip(ranked["ticker"], ranked["dvol_rank"], strict=False))
    dvol_map = dict(zip(ranked["ticker"], ranked["dollar_volume"], strict=False))

    cards = []
    for row in day.itertuples(index=False):
        ticker = row.ticker
        close = float(row.close)
        atr = float(row.atr) if np.isfinite(row.atr) else float("nan")
        q50c = float(row.pred_close_q50)
        q50h = float(row.pred_high_q50)
        q50l = float(row.pred_low_q50)
        q10l = float(row.pred_low_q10)
        q90h = float(row.pred_high_q90)
        range_ok = bool(beat.get(ticker, False) or overall_beats)
        setup = choose_setup(close, atr, q50h, q50l, q10l, q90h, range_ok)
        action = "觀望"
        side = setup.side
        reason = setup.reason
        tp = sl = None
        entry_px = None
        if setup.side == 1:
            action, tp, sl, entry_px = "做多", setup.tp, setup.sl, setup.entry
        elif setup.side == -1:
            action, tp, sl, entry_px = "做空", setup.tp, setup.sl, setup.entry

        cards.append(
            {
                "ticker": ticker,
                "name": getattr(row, "sector", None),
                "sector": getattr(row, "sector", None),
                "model_family": family,
                "dvol_rank": int(rank_map.get(ticker, 0) or 0),
                "dollar_volume": float(dvol_map.get(ticker, 0) or 0),
                "action": action,
                "side": side,
                "reason": reason,
                "asof": str(asof.date()),
                "prior_close": close,
                "entry": "盤中觸價" if side else "無",
                "entry_px": None if entry_px is None else float(entry_px),
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
                "high_confidence": bool(setup.side != 0 and setup.room >= 2 * FADE_MIN_ATR * atr),
                "beats_baseline": bool(beat.get(ticker, False)),
                "beats_range": bool(beat.get(ticker, False)),
                "strategy": "fade_to_prior_close",
                "recent_error": _recent_error(oos, ticker, family=family),
                "data_source": _ticker_source(ohlcv, ticker, asof),
                "universe_source": "S&P 500 · Wikipedia 2026-09-12",
            }
        )

    cards.sort(key=lambda c: (0 if c["action"] != "觀望" else 1, c["dvol_rank"]))
    family_label = {"shared": "共用模型", "sector": "行業模型", "stock": "個股模型"}.get(family, family)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof": str(asof.date()),
        "next_session": "下一常規交易時段",
        "model_family": family,
        "model_family_zh": family_label,
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


def build_all_family_cards(
    featured: pd.DataFrame,
    ohlcv: pd.DataFrame,
    oos: pd.DataFrame,
    family_reports: dict[str, dict],
    shared: SharedBundle,
    sector: SectorBundle,
    stock: StockBundle,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for family in ("shared", "sector", "stock"):
        rep = family_reports[family]
        predictor = FamilyPredictor(family, shared, sector, stock)
        out[family] = build_cards(
            featured=featured,
            ohlcv=ohlcv,
            oos=oos,
            per_ticker=rep["per_ticker"],
            overall_beats=bool(rep.get("overall_beats_range", rep.get("overall_beats_baseline"))),
            model=predictor,
            family=family,
        )
    return out
