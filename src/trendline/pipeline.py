"""End-to-end: features → walk-forward → final fit → cards → artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trendline.backtest import evaluate_forecasts, run_walk_forward, simulate_trades, summarize_backtest
from trendline.cards import build_cards
from trendline.config import ARTIFACT_DIR, CARDS_PATH, FEATURE_PATH, METRICS_PATH, MODEL_DIR, OOS_PATH
from trendline.data.store import load_ohlcv, summarize
from trendline.features import build_features
from trendline.models.lightgbm_quantile import QuantileLGBM


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, float):
        if pd.isna(obj):
            return None
        return obj
    return obj


def run(ohlcv: pd.DataFrame | None = None) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if ohlcv is None:
        ohlcv = load_ohlcv()
    data_info = summarize(ohlcv)
    featured = build_features(ohlcv)
    featured.to_parquet(FEATURE_PATH, index=False)

    oos, fold_logs = run_walk_forward(featured)
    oos.to_parquet(OOS_PATH, index=False)
    forecast = evaluate_forecasts(oos)
    trades, daily = simulate_trades(oos, forecast["per_ticker"], forecast["overall_beats_baseline"])
    if not trades.empty:
        trades.to_parquet(ARTIFACT_DIR / "trades.parquet", index=False)
    if not daily.empty:
        daily.to_parquet(ARTIFACT_DIR / "daily_returns.parquet", index=False)

    metrics = summarize_backtest(forecast, trades, daily, fold_logs)
    metrics["data"] = data_info
    per_path = ARTIFACT_DIR / "per_ticker.json"
    forecast["per_ticker"].to_json(per_path, orient="records", date_format="iso")

    # Final production fit on all complete rows (for the live card, not for metrics).
    ready = featured.dropna(subset=["y_close", "ret_20d", "atr"]).copy()
    from trendline.universe import is_sp500

    ready = ready[ready["ticker"].map(is_sp500)]
    cut = ready["date"].quantile(0.90)
    model = QuantileLGBM().fit(ready[ready["date"] <= cut], ready[ready["date"] > cut])
    model.save(MODEL_DIR)

    cards = build_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=oos,
        per_ticker=forecast["per_ticker"],
        overall_beats=forecast["overall_beats_baseline"],
        model=model,
    )
    CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": metrics, "cards": cards, "data": data_info}


def load_metrics(path: Path | None = None) -> dict:
    p = Path(path or METRICS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def load_cards(path: Path | None = None) -> dict:
    p = Path(path or CARDS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))
