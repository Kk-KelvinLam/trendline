#!/usr/bin/env python3
"""Walk-forward backtest. Prints computed metrics only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.pipeline import run  # noqa: E402


def main() -> int:
    result = run()
    metrics = result["metrics"]
    print(json.dumps({k: metrics[k] for k in metrics if k != "folds"}, indent=2, default=str)[:4000])
    print("…")
    print(f"folds: {len(metrics.get('folds', []))}")
    print(f"cards: {result['cards']['n_cards']}  long={result['cards']['n_long']} short={result['cards']['n_short']} flat={result['cards']['n_flat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
