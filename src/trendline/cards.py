"""Next-session trade cards. Long/short only if the model beat baseline OOS."""

from __future__ import annotations

from typing import Protocol

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trendline.config import (
    ARTIFACT_DIR,
    FADE_MIN_ATR,
    CLOSE_GATE_FAMILIES,
    RECENT_CLOSE_MAE_MAX,
    RECENT_CLOSE_MAE_SOFT,
    RECENT_ERROR_MIN_N,
    RECENT_ERROR_WINDOW,
    SHOW_TOP_N,
    MAE_RANK_W_CLOSE,
    MAE_RANK_W_HIGH,
    MAE_RANK_W_LOW,
    recent_close_error_path,
)
from trendline.range_touch import FadeSetup, choose_setup
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


def _load_recent_error_artifact(family: str) -> dict[str, dict]:
    path = recent_close_error_path(family)
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    tickers = payload.get("tickers") or {}
    return {str(k): dict(v) for k, v in tickers.items() if isinstance(v, dict)}


def _write_recent_error_artifact(family: str, mapping: dict[str, dict], asof: pd.Timestamp) -> Path:
    import json
    from datetime import datetime, timezone

    path = recent_close_error_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof": str(pd.Timestamp(asof).date()),
        "n_window": int(RECENT_ERROR_WINDOW),
        "family": family,
        "tickers": mapping,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _pred_q50_col(pred_frame: pd.DataFrame, target: str, family: str) -> str | None:
    for col in (f"pred_{target}_q50", pred_col(target, 0.50, family), pred_col(target, 0.50)):
        if col in pred_frame.columns:
            return col
    return None


def _mae_score(err: dict) -> float | None:
    parts = []
    for key, w in (
        ("mae_high_ret", MAE_RANK_W_HIGH),
        ("mae_low_ret", MAE_RANK_W_LOW),
        ("mae_close_ret", MAE_RANK_W_CLOSE),
    ):
        val = err.get(key)
        if val is None or not np.isfinite(val):
            return None
        parts.append(w * float(val))
    return float(sum(parts))


def rank_by_recent_mae(
    recent_map: dict[str, dict],
    dvol_by_ticker: dict[str, float] | None = None,
    *,
    top_n: int = SHOW_TOP_N,
    min_n: int = RECENT_ERROR_MIN_N,
) -> list[dict]:
    """Best-first shortlist: 0.4 High + 0.4 Low + 0.2 Close recent MAE; n then dollar volume."""
    dvol_by_ticker = dvol_by_ticker or {}
    rows: list[tuple] = []
    for ticker, err in (recent_map or {}).items():
        if int(err.get("n") or 0) < int(min_n):
            continue
        score = _mae_score(err)
        if score is None:
            continue
        dv = float(dvol_by_ticker.get(ticker) or 0.0)
        rows.append((score, -int(err.get("n") or 0), -dv, str(ticker)))
    rows.sort()
    out = []
    for i, (score, neg_n, _neg_dv, ticker) in enumerate(rows[: int(top_n)], start=1):
        out.append(
            {
                "ticker": ticker,
                "mae_rank": i,
                "mae_score": float(score),
                "n": int(-neg_n),
            }
        )
    return out


def compute_recent_close_errors(
    featured: pd.DataFrame,
    model: "Predictor | FamilyPredictor",
    family: str = "shared",
    n: int = RECENT_ERROR_WINDOW,
    asof: pd.Timestamp | None = None,
) -> dict[str, dict]:
    """Score last-n labeled sessions with the *current* saved models (true recent MAE).

    High / Low / Close vs ``y_*`` before ``asof``. Does not need oos parquet.
    """
    if featured is None or getattr(featured, "empty", True):
        return {}
    df = featured.copy()
    df["date"] = _naive_midnight(df["date"])
    need = ["y_close", "close", "ticker", "date"]
    for col in need:
        if col not in df.columns:
            return {}
    label_cols = [c for c in ("y_high", "y_low", "y_close") if c in df.columns]
    ready = df.dropna(subset=label_cols + ["close"]).copy()
    if asof is not None:
        ready = ready[ready["date"] < _naive_midnight(asof)]
    if ready.empty:
        return {}
    dates = sorted(ready["date"].unique())
    use_dates = dates[-int(n) :]
    window = ready[ready["date"].isin(use_dates)].copy()
    if window.empty:
        return {}
    try:
        preds = model.predict(window)
    except Exception:
        return {}
    pred_frame = preds.reset_index(drop=True) if hasattr(preds, "reset_index") else preds
    keep = ["date", "ticker", "close", *label_cols]
    base = window[keep].reset_index(drop=True)
    col_c = _pred_q50_col(pred_frame, "close", family)
    if col_c is None:
        return {}
    use_pred = [col_c]
    col_h = _pred_q50_col(pred_frame, "high", family) if "y_high" in base.columns else None
    col_l = _pred_q50_col(pred_frame, "low", family) if "y_low" in base.columns else None
    if col_h:
        use_pred.append(col_h)
    if col_l:
        use_pred.append(col_l)
    merged = pd.concat([base, pred_frame[use_pred]], axis=1)
    out: dict[str, dict] = {}
    for ticker, g in merged.groupby("ticker", sort=False):
        close_err = (g["y_close"] - g[col_c]).abs()
        px_err = (g["close"] * close_err).abs()
        row = {
            "n": int(len(g)),
            "mae_close_ret": float(close_err.mean()),
            "mae_close_px": float(px_err.mean()),
            "scope": "recent",
        }
        if col_h and "y_high" in g.columns:
            high_err = (g["y_high"] - g[col_h]).abs()
            row["mae_high_ret"] = float(high_err.mean())
            row["mae_high_px"] = float((g["close"] * high_err).abs().mean())
        if col_l and "y_low" in g.columns:
            low_err = (g["y_low"] - g[col_l]).abs()
            row["mae_low_ret"] = float(low_err.mean())
            row["mae_low_px"] = float((g["close"] * low_err).abs().mean())
        score = _mae_score(row)
        if score is not None:
            row["mae_score"] = score
        out[str(ticker)] = row
    return out


def _recent_error(
    oos: pd.DataFrame | None,
    ticker: str,
    family: str = "shared",
    n: int = RECENT_ERROR_WINDOW,
    *,
    per_ticker_row: dict | None = None,
    prior_close: float | None = None,
    recent_lookup: dict | None = None,
) -> dict:
    """Resolve Close MAE for a card: live recent → artifact → OOS parquet → walk-forward."""
    if recent_lookup and ticker in recent_lookup:
        row = dict(recent_lookup[ticker])
        row.setdefault("scope", "recent")
        return row
    if oos is not None and not getattr(oos, "empty", True):
        g = oos.loc[oos["ticker"] == ticker].sort_values("date").tail(n)
        if not g.empty:
            col = pred_col("close", 0.50, family)
            if col not in g.columns:
                col = pred_col("close", 0.50)
            if col in g.columns:
                err = (g["y_close"] - g[col]).abs()
                px_err = (g["close"] * err).abs()
                return {
                    "n": int(len(g)),
                    "mae_close_ret": float(err.mean()),
                    "mae_close_px": float(px_err.mean()),
                    "scope": "recent",
                }
    artifact = _load_recent_error_artifact(family).get(ticker)
    if artifact and artifact.get("mae_close_ret") is not None:
        out = dict(artifact)
        out.setdefault("scope", "recent")
        return out
    row = per_ticker_row or {}
    mae_ret = row.get("model_mae_close")
    if mae_ret is None:
        return {"n": 0, "mae_close_ret": None, "mae_close_px": None, "scope": "none"}
    px = None
    if prior_close is not None and np.isfinite(prior_close):
        px = float(abs(prior_close) * float(mae_ret))
    return {
        "n": int(row.get("n") or 0),
        "mae_close_ret": float(mae_ret),
        "mae_close_px": px,
        "scope": "walk_forward",
    }


def _apply_recent_mae_decision(setup: FadeSetup, err: dict) -> FadeSetup:
    """Hard-flat when trusted recent High/Low/Close MAE is too high."""
    if setup.side == 0:
        return setup
    if err.get("scope") != "recent":
        return setup
    if int(err.get("n") or 0) < int(RECENT_ERROR_MIN_N):
        return setup
    mae = _mae_score(err)
    if mae is None:
        mae = err.get("mae_close_ret")
    if mae is None or not np.isfinite(mae):
        return setup
    if float(mae) > float(RECENT_CLOSE_MAE_MAX):
        return FadeSetup(0, float("nan"), float("nan"), float("nan"), 0.0, "recent_mae_too_high")
    return setup


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


def _last_bar_dates(ohlcv: pd.DataFrame) -> dict[str, str]:
    """Map ticker → max OHLCV date (ISO YYYY-MM-DD), dates normalized like asof."""
    if ohlcv is None or getattr(ohlcv, "empty", True):
        return {}
    if "ticker" not in ohlcv.columns or "date" not in ohlcv.columns:
        return {}
    df = ohlcv[["ticker", "date"]].copy()
    df["date"] = _naive_midnight(df["date"])
    grouped = df.groupby("ticker", sort=False)["date"].max()
    return {str(t): str(pd.Timestamp(d).date()) for t, d in grouped.items()}


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

    if model is None:
        shared = SharedBundle().load()
        sector = SectorBundle().load() if family != "shared" else None
        stock = StockBundle().load() if family == "stock" else None
        if family == "sector" and sector is None:
            sector = SectorBundle().load()
        model = FamilyPredictor(family, shared, sector, stock)

    # Dollar volume for display + MAE tie-break; do not use it to cut the pool.
    ranked_dvol = rank_by_dollar_volume(ohlcv, asof=asof, top_n=10_000)
    dvol_map = dict(zip(ranked_dvol["ticker"], ranked_dvol["dollar_volume"], strict=False)) if not ranked_dvol.empty else {}
    rank_map = dict(zip(ranked_dvol["ticker"], ranked_dvol["dvol_rank"], strict=False)) if not ranked_dvol.empty else {}

    recent_map = compute_recent_close_errors(
        feat, model, family=family, n=RECENT_ERROR_WINDOW, asof=asof
    )
    if not recent_map:
        recent_map = _load_recent_error_artifact(family)
    else:
        _write_recent_error_artifact(family, recent_map, asof)

    mae_ranked = rank_by_recent_mae(recent_map, dvol_map, top_n=SHOW_TOP_N)
    if mae_ranked:
        show = {r["ticker"] for r in mae_ranked}
        mae_rank_map = {r["ticker"]: r["mae_rank"] for r in mae_ranked}
        mae_score_map = {r["ticker"]: r["mae_score"] for r in mae_ranked}
    else:
        # No trusted recent High/Low/Close MAE yet — keep a 100-name card file.
        fallback = ranked_dvol.head(int(SHOW_TOP_N)) if not ranked_dvol.empty else ranked_dvol
        show = set(fallback["ticker"]) if not fallback.empty else set(day["ticker"])
        mae_rank_map = {}
        mae_score_map = {}
    day = day[day["ticker"].isin(show)].copy()
    if day.empty:
        raise RuntimeError(
            f"no S&P names in the MAE shortlist for session {asof.date()} "
            f"(mae_ranked={len(mae_ranked)}; dvol={len(ranked_dvol)})"
        )

    preds = model.predict(day)
    base = BaselineModel().predict(day)
    day = pd.concat([day.reset_index(drop=True), preds.reset_index(drop=True), base.reset_index(drop=True)], axis=1)

    beat_col = "beats_range" if "beats_range" in per_ticker.columns else "beats_baseline"
    beat = dict(zip(per_ticker["ticker"], per_ticker[beat_col], strict=False))
    per_map = {r["ticker"]: r for r in per_ticker.to_dict(orient="records")}
    last_bar_map = _last_bar_dates(ohlcv)

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
        err = _recent_error(
            oos,
            ticker,
            family=family,
            per_ticker_row=per_map.get(ticker),
            prior_close=close,
            recent_lookup=recent_map,
        )
        q50c_gate = q50c if family in CLOSE_GATE_FAMILIES else None
        setup = choose_setup(close, atr, q50h, q50l, q10l, q90h, range_ok, q50c=q50c_gate)
        setup = _apply_recent_mae_decision(setup, err)
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
                "mae_rank": int(mae_rank_map.get(ticker, 0) or 0),
                "mae_score": mae_score_map.get(ticker),
                "dollar_volume": float(dvol_map.get(ticker, 0) or 0),
                "action": action,
                "side": side,
                "reason": reason,
                "asof": str(asof.date()),
                "last_bar_date": last_bar_map.get(ticker),
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
                "high_confidence": bool(
                    setup.side != 0
                    and setup.room >= 2 * FADE_MIN_ATR * atr
                    and not (
                        err.get("scope") == "recent"
                        and int(err.get("n") or 0) >= int(RECENT_ERROR_MIN_N)
                        and (
                            (_mae_score(err) is not None and float(_mae_score(err)) > float(RECENT_CLOSE_MAE_SOFT))
                            or (
                                _mae_score(err) is None
                                and err.get("mae_close_ret") is not None
                                and float(err["mae_close_ret"]) > float(RECENT_CLOSE_MAE_SOFT)
                            )
                        )
                    )
                ),
                "beats_baseline": bool(beat.get(ticker, False)),
                "beats_range": bool(beat.get(ticker, False)),
                "strategy": "fade_to_prior_close",
                "recent_error": err,
                "data_source": _ticker_source(ohlcv, ticker, asof),
                "universe_source": "S&P 500 · Wikipedia 2026-09-12",
            }
        )

    cards.sort(
        key=lambda c: (
            0 if c["action"] != "觀望" else 1,
            c.get("mae_rank") or 10_000,
            c.get("dvol_rank") or 10_000,
        )
    )
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
