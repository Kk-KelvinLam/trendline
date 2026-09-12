"""Sector assignment and thin-ticker fallback (toy frames only — not market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendline.config import FEATURE_COLS, STOCK_MIN_TRAIN_ROWS
from trendline.models.families import SectorBundle, SharedBundle, StockBundle, pred_col
from trendline.universe import large_sectors, model_sector_for, model_sector_map, sector_map


def _toy_ready(n_dates: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    rows = []
    for t_i, ticker in enumerate(tickers):
        for d_i, dt in enumerate(dates):
            feat = {c: float(rng.normal(0, 0.01)) for c in FEATURE_COLS}
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "close": 100.0 + t_i + 0.01 * d_i,
                    "atr": 2.0,
                    "y_high": float(rng.normal(0.01, 0.005)),
                    "y_low": float(rng.normal(-0.01, 0.005)),
                    "y_close": float(rng.normal(0.0, 0.008)),
                    **feat,
                }
            )
    return pd.DataFrame(rows)


def test_model_sector_assignment_uses_snapshot_threshold():
    sm = sector_map()
    assert sm
    large = large_sectors()
    assert "Information Technology" in large or len(large) >= 1
    # Every mapped ticker resolves to a large GICS name or Other
    for ticker, sec in model_sector_map().items():
        if sm.get(ticker) in large:
            assert sec == sm[ticker]
        else:
            assert sec == "Other"
    # AAPL should map to IT (present in snapshot)
    assert model_sector_for("AAPL") in large or model_sector_for("AAPL") == "Other"


def test_stock_bundle_falls_back_to_shared_for_thin_ticker():
    # Thick ticker gets its own model; thin ticker (< STOCK_MIN_TRAIN_ROWS) uses shared.
    thick = "THICK"
    thin = "THIN"
    n_thick = STOCK_MIN_TRAIN_ROWS + 40
    n_thin = 80
    assert n_thin < STOCK_MIN_TRAIN_ROWS

    dates = pd.bdate_range("2024-01-02", periods=n_thick)
    rng = np.random.default_rng(42)

    def rows_for(ticker: str, n: int) -> list[dict]:
        out = []
        for d_i, dt in enumerate(dates[:n]):
            feat = {c: float(rng.normal(0, 0.01)) for c in FEATURE_COLS}
            out.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "close": 50.0 + 0.01 * d_i,
                    "atr": 1.5,
                    "y_high": float(rng.normal(0.01, 0.005)),
                    "y_low": float(rng.normal(-0.01, 0.005)),
                    "y_close": float(rng.normal(0.0, 0.008)),
                    **feat,
                }
            )
        return out

    train = pd.DataFrame(rows_for(thick, n_thick) + rows_for(thin, n_thin))
    # Test frame: one day, both tickers
    last = dates[n_thin - 1]
    test = train[train["date"] == last].copy()
    assert set(test["ticker"]) == {thick, thin}

    shared = SharedBundle().fit(train)
    shared_preds = shared.predict(test)
    stock = StockBundle(min_train_rows=STOCK_MIN_TRAIN_ROWS).fit(train)
    assert thick in stock.models
    assert thin not in stock.models

    preds = stock.predict(test, fallback=shared_preds)
    # Thin rows equal shared; thick rows may differ
    thin_idx = test["ticker"] == thin
    thick_idx = test["ticker"] == thick
    col = pred_col("close", 0.50)
    assert np.allclose(
        preds.loc[thin_idx, col].to_numpy(),
        shared_preds.loc[thin_idx, col].to_numpy(),
        equal_nan=True,
    )
    # Thick has a dedicated model so prediction is finite
    assert np.isfinite(preds.loc[thick_idx, col].to_numpy()).all()


def test_sector_bundle_predicts_with_fallback():
    # Use real sector map tickers if present; otherwise synthetic Other.
    smap = model_sector_map()
    tickers = list(smap.keys())[:3]
    if len(tickers) < 2:
        tickers = ["AAA", "BBB"]
    train = _toy_ready(120, tickers, seed=1)
    test = train[train["date"] == train["date"].max()].copy()
    shared = SharedBundle().fit(train)
    shared_preds = shared.predict(test)
    sector = SectorBundle().fit(train)
    preds = sector.predict(test, fallback=shared_preds)
    col = pred_col("close", 0.50)
    assert col in preds.columns
    assert preds[col].notna().all()
