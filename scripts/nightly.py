#!/usr/bin/env python3
"""CI/local nightly: fetch prices, then refresh cards (or full retrain)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run(script: str, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(extra or [])]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _refresh_cards() -> int:
    from trendline.config import (
        ARTIFACT_DIR,
        METRICS_PATH,
        MODEL_SECTOR_DIR,
        MODEL_SHARED_DIR,
        MODEL_STOCK_DIR,
    )
    from trendline.pipeline import refresh_cards_from_saved_models

    per_ok = (ARTIFACT_DIR / "per_ticker.json").exists()
    models_ok = (
        any(MODEL_SHARED_DIR.glob("*.txt"))
        and MODEL_SECTOR_DIR.exists()
        and MODEL_STOCK_DIR.exists()
    )
    if not per_ok or not METRICS_PATH.exists() or not models_ok:
        print("missing backtest artifacts; running full three-family walk-forward")
        return _run("backtest.py")
    cards = refresh_cards_from_saved_models()
    for fam, payload in cards.items():
        print(f"wrote cards_{fam}.json  n={payload['n_cards']} asof={payload['asof']}")
    return 0


def _is_first_sunday(d: date | None = None) -> bool:
    d = d or date.today()
    return d.weekday() == 6 and d.day <= 7


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true", help="Full three-family walk-forward + refit")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--update-universe", action="store_true", help="Refresh sp500.csv from Wikipedia")
    args = p.parse_args()

    if args.update_universe or (args.retrain and _is_first_sunday()):
        print("refreshing S&P 500 membership from Wikipedia")
        rc = _run("update_universe.py")
        if rc != 0:
            print("universe refresh failed; keeping the committed list")

    fetch_args = ["--start", args.start]
    if args.update_universe or (args.retrain and _is_first_sunday()):
        print("first-Sunday / universe refresh: full OHLCV pull")
        fetch_args.append("--full")
    rc = _run("fetch.py", fetch_args)
    if rc != 0:
        return rc
    if args.retrain:
        return _run("backtest.py")
    return _refresh_cards()


if __name__ == "__main__":
    raise SystemExit(main())
