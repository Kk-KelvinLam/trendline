#!/usr/bin/env python3
"""CI/local nightly: fetch prices, then refresh cards (or full retrain)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _refresh_cards() -> int:
    import pandas as pd

    from trendline.cards import build_cards
    from trendline.config import ARTIFACT_DIR, CARDS_PATH, METRICS_PATH, MODEL_DIR, OOS_PATH
    from trendline.data.store import load_ohlcv
    from trendline.features import build_features
    from trendline.models.lightgbm_quantile import QuantileLGBM

    ohlcv = load_ohlcv()
    featured = build_features(ohlcv)
    oos = pd.read_parquet(OOS_PATH) if OOS_PATH.exists() else pd.DataFrame()
    per_path = ARTIFACT_DIR / "per_ticker.json"
    if not per_path.exists() or not METRICS_PATH.exists() or not any(MODEL_DIR.glob("*.txt")):
        print("missing backtest artifacts; running full walk-forward")
        return _run("backtest.py")
    per = pd.read_json(per_path)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cards = build_cards(
        featured=featured,
        ohlcv=ohlcv,
        oos=oos,
        per_ticker=per,
        overall_beats=bool(metrics.get("overall_beats_baseline")),
        model=QuantileLGBM().load(MODEL_DIR),
    )
    CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {CARDS_PATH}  n={cards['n_cards']} asof={cards['asof']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true", help="Full walk-forward + refit")
    p.add_argument("--start", default="2023-01-01")
    args = p.parse_args()

    rc = _run("fetch.py", ["--start", args.start])
    if rc != 0:
        return rc
    if args.retrain:
        return _run("backtest.py")
    return _refresh_cards()


if __name__ == "__main__":
    raise SystemExit(main())
