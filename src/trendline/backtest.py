"""Purged walk-forward backtest and next-open trading simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendline.config import (
    ATR_SL_MULT,
    DIR_RET_MIN,
    FEATURE_COLS,
    MAX_POSITIONS,
    RANGE_ATR_MIN,
    TARGETS,
    WF_MIN_TRAIN_DAYS,
    WF_PURGE_DAYS,
    WF_TEST_DAYS,
)
from trendline.metrics import directional_accuracy, forecast_block, trade_stats
from trendline.models.baseline import BaselineModel
from trendline.models.lightgbm_quantile import QuantileLGBM
from trendline.universe import is_sp500


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def walk_forward_folds(dates: pd.Series) -> list[Fold]:
    uniq = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    folds: list[Fold] = []
    i = WF_MIN_TRAIN_DAYS
    fold_id = 1
    while i + WF_TEST_DAYS <= len(uniq):
        train_end_idx = i - WF_PURGE_DAYS
        if train_end_idx < WF_MIN_TRAIN_DAYS // 2:
            i += WF_TEST_DAYS
            continue
        test_start_idx = i
        test_end_idx = i + WF_TEST_DAYS - 1
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=uniq[0],
                train_end=uniq[train_end_idx],
                test_start=uniq[test_start_idx],
                test_end=uniq[test_end_idx],
                n_train=train_end_idx + 1,
                n_test=WF_TEST_DAYS,
            )
        )
        fold_id += 1
        i += WF_TEST_DAYS
    return folds


def _ready(df: pd.DataFrame) -> pd.DataFrame:
    need = FEATURE_COLS + ["y_high", "y_low", "y_close", "close", "atr"]
    return df.dropna(subset=need).copy()


def run_walk_forward(featured: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Train LightGBM on expanding windows; return OOS prediction frame + fold log."""
    df = _ready(featured)
    # Train on S&P members only (macros stay as features via merge, not as rows).
    df = df[df["ticker"].map(is_sp500)].copy()
    if df.empty:
        raise RuntimeError("no S&P rows with complete features/targets")

    folds = walk_forward_folds(df["date"])
    if not folds:
        raise RuntimeError(
            "not enough history for walk-forward "
            f"(need ~{WF_MIN_TRAIN_DAYS + WF_TEST_DAYS} sessions)"
        )

    parts: list[pd.DataFrame] = []
    fold_logs: list[dict] = []
    baseline = BaselineModel()

    for fold in folds:
        train = df[(df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)]
        test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
        if len(train) < 500 or test.empty:
            continue
        # Small tail of train as early-stopping valid (still before purge gap).
        cut = train["date"].quantile(0.88)
        tr = train[train["date"] <= cut]
        va = train[train["date"] > cut]
        model = QuantileLGBM().fit(tr, va if len(va) > 50 else None)
        pred = model.predict(test)
        base = baseline.predict(test)
        block = pd.concat([test.reset_index(drop=True), pred.reset_index(drop=True), base.reset_index(drop=True)], axis=1)
        block["fold_id"] = fold.fold_id
        parts.append(block)
        fold_logs.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            }
        )

    if not parts:
        raise RuntimeError("walk-forward produced no OOS predictions")
    oos = pd.concat(parts, ignore_index=True)
    return oos, fold_logs


def _px(close: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return close * (1.0 + ret)


def evaluate_forecasts(oos: pd.DataFrame) -> dict:
    """MAE/RMSE/MAPE, coverage, pinball, directional accuracy — model vs baseline."""
    close = oos["close"].to_numpy(dtype=float)
    report: dict = {"n_rows": int(len(oos)), "n_tickers": int(oos["ticker"].nunique())}

    for target in TARGETS:
        yt = oos[f"y_{target}"].to_numpy(dtype=float)
        q50 = oos[f"pred_{target}_q50"].to_numpy(dtype=float)
        q10 = oos[f"pred_{target}_q10"].to_numpy(dtype=float)
        q90 = oos[f"pred_{target}_q90"].to_numpy(dtype=float)
        b50 = oos[f"base_{target}_q50"].to_numpy(dtype=float)
        b10 = oos[f"base_{target}_q10"].to_numpy(dtype=float)
        b90 = oos[f"base_{target}_q90"].to_numpy(dtype=float)
        actual_px = _px(close, yt)
        pred_px = _px(close, q50)
        base_px = _px(close, b50)
        report[f"model_{target}"] = forecast_block(yt, q50, q10, q90, actual_px, pred_px)
        report[f"baseline_{target}"] = forecast_block(yt, b50, b10, b90, actual_px, base_px)

    report["dir_acc_vs_prior_close"] = directional_accuracy(
        oos["y_close"].to_numpy(), oos["pred_close_q50"].to_numpy()
    )
    report["dir_acc_vs_prior_close_baseline"] = directional_accuracy(
        oos["y_close"].to_numpy(), oos["base_close_q50"].to_numpy()
    )
    # Direction vs next open: sign(close - open) vs sign(pred_close_px - next_open)
    if "next_open" in oos.columns:
        pred_c_px = _px(close, oos["pred_close_q50"].to_numpy())
        act_from_open = oos["next_close"].to_numpy(dtype=float) - oos["next_open"].to_numpy(dtype=float)
        pred_from_open = pred_c_px - oos["next_open"].to_numpy(dtype=float)
        report["dir_acc_vs_next_open"] = directional_accuracy(act_from_open, pred_from_open)
        base_c_px = _px(close, oos["base_close_q50"].to_numpy())
        base_from_open = base_c_px - oos["next_open"].to_numpy(dtype=float)
        report["dir_acc_vs_next_open_baseline"] = directional_accuracy(act_from_open, base_from_open)

    # Per-ticker close MAE (return space) for the card gate.
    rows = []
    for ticker, g in oos.groupby("ticker"):
        rows.append(
            {
                "ticker": ticker,
                "n": int(len(g)),
                "model_mae_close": float(np.mean(np.abs(g["y_close"] - g["pred_close_q50"]))),
                "base_mae_close": float(np.mean(np.abs(g["y_close"] - g["base_close_q50"]))),
                "model_mae_high": float(np.mean(np.abs(g["y_high"] - g["pred_high_q50"]))),
                "base_mae_high": float(np.mean(np.abs(g["y_high"] - g["base_high_q50"]))),
                "model_mae_low": float(np.mean(np.abs(g["y_low"] - g["pred_low_q50"]))),
                "base_mae_low": float(np.mean(np.abs(g["y_low"] - g["base_low_q50"]))),
            }
        )
    per = pd.DataFrame(rows)
    per["beats_baseline"] = per["model_mae_close"] < per["base_mae_close"]
    report["per_ticker"] = per
    report["overall_beats_baseline"] = bool(
        report["model_close"]["mae_ret"] < report["baseline_close"]["mae_ret"]
    )
    return report


def simulate_trades(oos: pd.DataFrame, per_ticker: pd.DataFrame, overall_beats: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enter next open; TP at q50 extreme; SL at q10/q90 or 1×ATR. Conservative OHLC fill.

    A long/short is allowed only when the model beats the baseline on that ticker
    *or* overall. Otherwise the name is 觀望 (flat).
    """
    beat = dict(zip(per_ticker["ticker"], per_ticker["beats_baseline"], strict=False))
    recs: list[dict] = []

    work = oos.dropna(subset=["next_open", "next_high", "next_low", "next_close", "atr"]).copy()
    for date, day in work.groupby("date"):
        scored = []
        for row in day.itertuples(index=False):
            ticker = row.ticker
            allowed = bool(beat.get(ticker, False) or overall_beats)
            q50c = float(row.pred_close_q50)
            q50h = float(row.pred_high_q50)
            q50l = float(row.pred_low_q50)
            q10l = float(row.pred_low_q10)
            q90h = float(row.pred_high_q90)
            atr = float(row.atr)
            close = float(row.close)
            rng = (q90h - q10l) * close
            tight = (not np.isfinite(rng)) or (rng < RANGE_ATR_MIN * atr)
            weak = abs(q50c) < DIR_RET_MIN
            side = 0
            if allowed and (not tight) and (not weak):
                if q50c > 0:
                    side = 1
                elif q50c < 0:
                    side = -1
            scored.append((abs(q50c), side, row, q50h, q50l, q10l, q90h, atr, close))

        scored.sort(key=lambda x: x[0], reverse=True)
        taken = 0
        for _, side, row, q50h, q50l, q10l, q90h, atr, close in scored:
            if side == 0 or taken >= MAX_POSITIONS:
                continue
            entry = float(row.next_open)
            if not np.isfinite(entry) or entry <= 0:
                continue
            if side == 1:
                tp = close * (1.0 + q50h)
                sl_q = close * (1.0 + q10l)
                sl_atr = entry - ATR_SL_MULT * atr
                sl = max(sl_q, sl_atr)  # tighter (higher) stop
                ret, exit_px, reason = _fill_long(entry, float(row.next_high), float(row.next_low), float(row.next_close), tp, sl)
            else:
                tp = close * (1.0 + q50l)
                sl_q = close * (1.0 + q90h)
                sl_atr = entry + ATR_SL_MULT * atr
                sl = min(sl_q, sl_atr)  # tighter (lower) stop
                ret, exit_px, reason = _fill_short(entry, float(row.next_high), float(row.next_low), float(row.next_close), tp, sl)
            recs.append(
                {
                    "date": pd.Timestamp(date),
                    "next_date": row.next_date,
                    "ticker": row.ticker,
                    "side": side,
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "exit": exit_px,
                    "ret": ret,
                    "reason": reason,
                    "fold_id": getattr(row, "fold_id", None),
                }
            )
            taken += 1

    trades = pd.DataFrame(recs)
    if trades.empty:
        daily = pd.DataFrame(columns=["date", "ret"])
        return trades, daily
    daily = trades.groupby("date", as_index=False)["ret"].mean()
    daily = daily.sort_values("date")
    return trades, daily


def _fill_long(entry: float, high: float, low: float, close: float, tp: float, sl: float) -> tuple[float, float, str]:
    """If both TP and SL print, assume stop first (conservative)."""
    hit_sl = low <= sl
    hit_tp = high >= tp
    if hit_sl:
        return sl / entry - 1.0, sl, "sl"
    if hit_tp:
        return tp / entry - 1.0, tp, "tp"
    return close / entry - 1.0, close, "close"


def _fill_short(entry: float, high: float, low: float, close: float, tp: float, sl: float) -> tuple[float, float, str]:
    hit_sl = high >= sl
    hit_tp = low <= tp
    if hit_sl:
        return entry / sl - 1.0 if False else (entry - sl) / entry, sl, "sl"
    if hit_tp:
        return (entry - tp) / entry, tp, "tp"
    return (entry - close) / entry, close, "close"


def summarize_backtest(forecast: dict, trades: pd.DataFrame, daily: pd.DataFrame, fold_logs: list[dict]) -> dict:
    ts = trade_stats(trades, daily)
    out = {
        "n_rows": forecast["n_rows"],
        "n_tickers": forecast["n_tickers"],
        "overall_beats_baseline": forecast["overall_beats_baseline"],
        "folds": fold_logs,
        "model": {t: forecast[f"model_{t}"] for t in TARGETS},
        "baseline": {t: forecast[f"baseline_{t}"] for t in TARGETS},
        "dir_acc_vs_prior_close": forecast.get("dir_acc_vs_prior_close"),
        "dir_acc_vs_prior_close_baseline": forecast.get("dir_acc_vs_prior_close_baseline"),
        "dir_acc_vs_next_open": forecast.get("dir_acc_vs_next_open"),
        "dir_acc_vs_next_open_baseline": forecast.get("dir_acc_vs_next_open_baseline"),
        "trading": ts,
        "tickers_beating_baseline": int(forecast["per_ticker"]["beats_baseline"].sum()),
        "tickers_evaluated": int(len(forecast["per_ticker"])),
    }
    return out
