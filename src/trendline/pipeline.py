"""End-to-end: features → three-family walk-forward → final fit → cards → artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from trendline.backtest import (
    build_scoreboard,
    evaluate_family,
    run_walk_forward,
    simulate_trades,
    summarize_backtest,
)
from trendline.cards import FamilyPredictor, build_all_family_cards, build_cards
from trendline.config import (
    ARTIFACT_DIR,
    CARDS_PATH,
    CARDS_SECTOR_PATH,
    CARDS_SHARED_PATH,
    CARDS_STOCK_PATH,
    FEATURE_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_SECTOR_DIR,
    MODEL_SHARED_DIR,
    MODEL_STOCK_DIR,
    OOS_PATH,
    SCOREBOARD_PATH,
)
from trendline.data.store import load_ohlcv, summarize
from trendline.features import build_features
from trendline.models.families import SectorBundle, SharedBundle, StockBundle
from trendline.universe import is_sp500


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


def _write_cards(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _persist_open_plan() -> None:
    try:
        from trendline.ledger import load_ledger, save_ledger, snapshot_open_plan

        save_ledger(snapshot_open_plan(load_ledger()))
    except Exception as exc:
        print(f"open_plan snapshot failed: {exc}", flush=True)


def fit_production_models(featured: pd.DataFrame) -> tuple[SharedBundle, SectorBundle, StockBundle]:
    ready = featured.dropna(subset=["y_close", "ret_20d", "atr"]).copy()
    ready = ready[ready["ticker"].map(is_sp500)]
    cut = ready["date"].quantile(0.90)
    tr = ready[ready["date"] <= cut]
    va = ready[ready["date"] > cut]
    va_use = va if len(va) > 50 else None

    print(f"production fit: rows={len(ready)} tickers={ready['ticker'].nunique()}", flush=True)
    shared = SharedBundle().fit(tr, va_use)
    shared.save(MODEL_SHARED_DIR)

    sector = SectorBundle().fit(tr, va_use)
    sector.save(MODEL_SECTOR_DIR)
    print(f"  sector models: {sorted(sector.models)}", flush=True)

    stock = StockBundle().fit(tr, va_use)
    stock.save(MODEL_STOCK_DIR)
    print(f"  stock models: {len(stock.models)}", flush=True)

    # Legacy flat MODEL_DIR copy of shared (older nightly / scripts)
    for p in MODEL_DIR.glob("*.txt"):
        p.unlink()
    for src in MODEL_SHARED_DIR.glob("*.txt"):
        shutil.copy2(src, MODEL_DIR / src.name)

    return shared, sector, stock


def run(ohlcv: pd.DataFrame | None = None) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if ohlcv is None:
        ohlcv = load_ohlcv()
    data_info = summarize(ohlcv)
    featured = build_features(ohlcv)
    featured.to_parquet(FEATURE_PATH, index=False)

    oos, fold_logs = run_walk_forward(featured)
    oos.to_parquet(OOS_PATH, index=False)

    family_reports = {fam: evaluate_family(oos, fam) for fam in ("shared", "sector", "stock")}
    scoreboard = build_scoreboard(oos, fold_logs)
    SCOREBOARD_PATH.write_text(
        json.dumps(_jsonable(scoreboard), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Primary metrics / trading sim remain on shared for backward compat
    forecast = family_reports["shared"]
    trades, daily = simulate_trades(oos, forecast["per_ticker"], forecast["overall_beats_range"], family="shared")
    if not trades.empty:
        trades.to_parquet(ARTIFACT_DIR / "trades.parquet", index=False)
    if not daily.empty:
        daily.to_parquet(ARTIFACT_DIR / "daily_returns.parquet", index=False)

    metrics = summarize_backtest(forecast, trades, daily, fold_logs)
    metrics["data"] = data_info
    metrics["scoreboard_headline"] = {
        fam: {
            t: {
                "mae_px": family_reports[fam][f"model_{t}"].get("mae_px"),
                "mape_px": family_reports[fam][f"model_{t}"].get("mape_px"),
                "coverage": family_reports[fam][f"model_{t}"].get("coverage_q10_q90"),
            }
            for t in ("high", "low", "close")
        }
        for fam in ("shared", "sector", "stock")
    }
    metrics["scoreboard_headline"]["baseline"] = {
        t: {
            "mae_px": forecast[f"baseline_{t}"].get("mae_px"),
            "mape_px": forecast[f"baseline_{t}"].get("mape_px"),
            "coverage": forecast[f"baseline_{t}"].get("coverage_q10_q90"),
        }
        for t in ("high", "low", "close")
    }

    # Per-ticker JSON: shared + thin family summary columns
    per_path = ARTIFACT_DIR / "per_ticker.json"
    per = forecast["per_ticker"].copy()
    for fam in ("sector", "stock"):
        other = family_reports[fam]["per_ticker"].set_index("ticker")
        per[f"mae_close_{fam}"] = per["ticker"].map(other["model_mae_close"])
        per[f"beats_baseline_{fam}"] = per["ticker"].map(other["beats_baseline"])
    per.to_json(per_path, orient="records", date_format="iso")
    for fam in ("shared", "sector", "stock"):
        family_reports[fam]["per_ticker"].to_json(
            ARTIFACT_DIR / f"per_ticker_{fam}.json", orient="records", date_format="iso"
        )

    shared, sector, stock = fit_production_models(featured)

    cards_by_family = build_all_family_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=oos,
        family_reports=family_reports,
        shared=shared,
        sector=sector,
        stock=stock,
    )
    _write_cards(CARDS_SHARED_PATH, cards_by_family["shared"])
    _write_cards(CARDS_SECTOR_PATH, cards_by_family["sector"])
    _write_cards(CARDS_STOCK_PATH, cards_by_family["stock"])
    _write_cards(CARDS_PATH, cards_by_family["shared"])  # backward compat
    _persist_open_plan()

    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "cards": cards_by_family["shared"],
        "cards_by_family": cards_by_family,
        "scoreboard": scoreboard,
        "data": data_info,
    }


def refresh_cards_from_saved_models(ohlcv: pd.DataFrame | None = None) -> dict[str, dict]:
    """Weekday nightly path: infer three card files without re-running walk-forward."""
    if ohlcv is None:
        ohlcv = load_ohlcv()
    featured = build_features(ohlcv)
    oos = pd.read_parquet(OOS_PATH) if OOS_PATH.exists() else pd.DataFrame()

    family_reports = {}
    for fam in ("shared", "sector", "stock"):
        per_path = ARTIFACT_DIR / f"per_ticker_{fam}.json"
        if not per_path.exists():
            per_path = ARTIFACT_DIR / "per_ticker.json"
        per = pd.read_json(per_path)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        headline = (metrics.get("scoreboard_headline") or {}).get(fam) or {}
        base = (metrics.get("scoreboard_headline") or {}).get("baseline") or {}
        try:
            beats_range = float(headline["high"]["mae_px"]) < float(base["high"]["mae_px"]) and float(
                headline["low"]["mae_px"]
            ) < float(base["low"]["mae_px"])
        except (KeyError, TypeError, ValueError):
            if "beats_range" in per.columns:
                beats_range = bool(per["beats_range"].mean() > 0.5)
            else:
                beats_range = bool(metrics.get("overall_beats_range", False))
        family_reports[fam] = {
            "per_ticker": per,
            "overall_beats_baseline": bool(metrics.get("overall_beats_baseline")),
            "overall_beats_range": bool(beats_range),
        }

    shared = SharedBundle().load(MODEL_SHARED_DIR)
    sector = SectorBundle().load(MODEL_SECTOR_DIR)
    stock = StockBundle().load(MODEL_STOCK_DIR)

    cards_by_family = build_all_family_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=oos,
        family_reports=family_reports,
        shared=shared,
        sector=sector,
        stock=stock,
    )
    _write_cards(CARDS_SHARED_PATH, cards_by_family["shared"])
    _write_cards(CARDS_SECTOR_PATH, cards_by_family["sector"])
    _write_cards(CARDS_STOCK_PATH, cards_by_family["stock"])
    _write_cards(CARDS_PATH, cards_by_family["shared"])
    _persist_open_plan()
    return cards_by_family


def load_metrics(path: Path | None = None) -> dict:
    p = Path(path or METRICS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def load_cards(path: Path | None = None) -> dict:
    p = Path(path or CARDS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))
