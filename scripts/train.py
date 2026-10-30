#!/usr/bin/env python3
"""Fit production LightGBM quantile models on all complete history."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.config import MODEL_DIR  # noqa: E402
from trendline.data.store import load_ohlcv  # noqa: E402
from trendline.features import build_features  # noqa: E402
from trendline.models.lightgbm_quantile import QuantileLGBM  # noqa: E402
from trendline.universe import is_sp500  # noqa: E402


def main() -> int:
    ohlcv = load_ohlcv()
    feat = build_features(ohlcv)
    ready = feat.dropna(subset=["y_close", "ret_20d", "atr"]).copy()
    ready = ready[ready["ticker"].map(is_sp500)]
    if ready.empty:
        print("ERROR: no complete feature rows")
        return 2
    cut = ready["date"].quantile(0.90)
    model = QuantileLGBM().fit(ready[ready["date"] <= cut], ready[ready["date"] > cut])
    path = model.save(MODEL_DIR)
    print(f"saved {len(model.models)} models under {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
