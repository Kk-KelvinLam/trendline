from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _gbm_ohlc(n: int, seed: int, start_px: float) -> pd.DataFrame:
    """Deterministic toy path for unit tests only — never used as product data."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, size=n)
    close = start_px * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_px], close[:-1]]) * (1 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, size=n))
    volume = rng.integers(1_000_000, 8_000_000, size=n)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def toy_ohlcv() -> pd.DataFrame:
    frames = []
    specs = [
        ("AAPL", 1, 150.0),
        ("MSFT", 2, 280.0),
        ("SPY", 3, 400.0),
        ("^VIX", 4, 18.0),
        ("XLK", 5, 160.0),
    ]
    for ticker, seed, px in specs:
        part = _gbm_ohlc(320, seed, px)
        part["ticker"] = ticker
        frames.append(part)
    return pd.concat(frames, ignore_index=True)
