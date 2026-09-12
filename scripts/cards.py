#!/usr/bin/env python3
"""Refresh next-session cards from saved models + latest parquet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.pipeline import refresh_cards_from_saved_models  # noqa: E402


def main() -> int:
    cards = refresh_cards_from_saved_models()
    for fam, payload in cards.items():
        print(f"wrote cards_{fam}.json  n={payload['n_cards']} asof={payload['asof']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
