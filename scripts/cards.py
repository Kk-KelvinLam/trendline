#!/usr/bin/env python3
"""Refresh next-session cards from saved models + latest parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.cards import build_cards  # noqa: E402
from trendline.config import ARTIFACT_DIR, CARDS_PATH  # noqa: E402
from trendline.data.store import load_ohlcv  # noqa: E402
from trendline.features import build_features  # noqa: E402
from trendline.models.lightgbm_quantile import QuantileLGBM  # noqa: E402
from trendline.pipeline import load_metrics  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ohlcv = load_ohlcv()
    featured = build_features(ohlcv)
    oos = pd.read_parquet(ARTIFACT_DIR / "oos_predictions.parquet")
    per = pd.read_json(ARTIFACT_DIR / "per_ticker.json")
    metrics = load_metrics()
    cards = build_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=oos,
        per_ticker=per,
        overall_beats=bool(metrics["overall_beats_baseline"]),
        model=QuantileLGBM().load(),
    )
    CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {CARDS_PATH}  n={cards['n_cards']} asof={cards['asof']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
