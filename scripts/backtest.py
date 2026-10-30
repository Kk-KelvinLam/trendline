#!/usr/bin/env python3
"""Three-family walk-forward backtest. Prints computed metrics only."""

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
    print(json.dumps({k: metrics[k] for k in metrics if k not in ("folds",)}, indent=2, default=str)[:5000])
    print("…")
    print(f"folds: {len(metrics.get('folds', []))}")
    hl = metrics.get("scoreboard_headline") or {}
    for fam in ("shared", "sector", "stock", "baseline"):
        block = hl.get(fam) or {}
        if not block:
            continue
        print(
            f"{fam:8s}  High MAE$={ (block.get('high') or {}).get('mae_px') }  "
            f"Low={ (block.get('low') or {}).get('mae_px') }  "
            f"Close={ (block.get('close') or {}).get('mae_px') }"
        )
    for fam, cards in (result.get("cards_by_family") or {"shared": result["cards"]}).items():
        print(
            f"cards_{fam}: {cards['n_cards']}  "
            f"long={cards['n_long']} short={cards['n_short']} flat={cards['n_flat']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
