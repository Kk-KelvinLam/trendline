"""Forecast and trading metrics. All numbers come from arrays — nothing is hardcoded."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _finite(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _finite(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _finite(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    mask = np.abs(yt) > 1e-12
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs((yt - yp) / yt)))


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    yt, yp = _finite(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if len(yt) == 0:
        return float("nan")
    e = yt - yp
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])))


def directional_accuracy(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float:
    a = np.asarray(actual_ret, dtype=float)
    p = np.asarray(pred_ret, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p) & (np.abs(a) > 1e-12) & (np.abs(p) > 1e-12)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(p[mask])))


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    if len(eq) == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd))


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods))


def forecast_block(
    y_true: np.ndarray,
    y_q50: np.ndarray,
    y_q10: np.ndarray | None = None,
    y_q90: np.ndarray | None = None,
    price_true: np.ndarray | None = None,
    price_q50: np.ndarray | None = None,
) -> dict:
    block = {
        "mae_ret": mae(y_true, y_q50),
        "rmse_ret": rmse(y_true, y_q50),
        "mape_ret": mape(y_true, y_q50),
        "n": int(np.isfinite(y_true).sum()),
    }
    if price_true is not None and price_q50 is not None:
        block["mae_px"] = mae(price_true, price_q50)
        block["rmse_px"] = rmse(price_true, price_q50)
        block["mape_px"] = mape(price_true, price_q50)
    if y_q10 is not None and y_q90 is not None:
        block["coverage_q10_q90"] = coverage(y_true, y_q10, y_q90)
        block["pinball_q10"] = pinball(y_true, y_q10, 0.10)
        block["pinball_q50"] = pinball(y_true, y_q50, 0.50)
        block["pinball_q90"] = pinball(y_true, y_q90, 0.90)
    return block


def trade_stats(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "hit_rate": float("nan"),
            "payoff": float("nan"),
            "avg_return": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "turnover": 0.0,
        }
    r = trades["ret"].to_numpy(dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    hit = float(np.mean(r > 0)) if len(r) else float("nan")
    payoff = float(np.mean(wins) / abs(np.mean(losses))) if len(wins) and len(losses) else float("nan")
    dret = daily["ret"].to_numpy(dtype=float) if daily is not None and not daily.empty else r
    eq = np.cumprod(1.0 + np.nan_to_num(dret, nan=0.0))
    n_days = max(int(daily["date"].nunique()) if daily is not None and not daily.empty else 1, 1)
    return {
        "n_trades": int(len(trades)),
        "hit_rate": hit,
        "payoff": payoff,
        "avg_return": float(np.nanmean(r)) if len(r) else float("nan"),
        "sharpe": sharpe(dret),
        "max_dd": max_drawdown(eq),
        "turnover": float(len(trades) / n_days),
    }
