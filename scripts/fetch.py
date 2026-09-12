#!/usr/bin/env python3
"""Download real daily OHLCV. Never synthesizes prices.

Keeps a single data/parquet/ohlcv.parquet:
  --full or missing file → history from --start
  otherwise → last 15 days merged onto the existing file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.config import OHLCV_PATH  # noqa: E402
from trendline.data.providers import CombinedProvider  # noqa: E402
from trendline.data.store import (  # noqa: E402
    delta_start,
    load_ohlcv,
    merge_ohlcv,
    save_ohlcv,
    summarize,
)
from trendline.universe import default_fetch_tickers  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch daily OHLCV into data/parquet/ohlcv.parquet")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument(
        "--full",
        action="store_true",
        help="Full history from --start (ignore existing parquet for the download window)",
    )
    p.add_argument("--tickers", default="", help="Comma-separated override")
    args = p.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = default_fetch_tickers()

    stored = load_ohlcv(OHLCV_PATH) if OHLCV_PATH.exists() else None
    existing = None if args.full else stored

    start = args.start
    mode = "full"
    if existing is not None and not existing.empty:
        start = delta_start(existing, args.start)
        mode = "delta"

    print(f"fetching {len(tickers)} symbols {mode} from {start} …")
    raw = CombinedProvider().download(tickers, start=start, end=args.end)
    if raw.empty:
        if stored is not None and not stored.empty:
            print("WARNING: download empty; keeping existing ohlcv.parquet")
            print(summarize(stored))
            return 0
        print("ERROR: no real prices downloaded. Check network / provider availability.")
        print(f"Drop a parquet with columns date,ticker,open,high,low,close,adj_close,volume at {OHLCV_PATH}")
        return 2

    combined = merge_ohlcv(existing if existing is not None else raw.iloc[0:0], raw)
    path = save_ohlcv(combined, OHLCV_PATH)
    info = summarize(combined)
    print(f"wrote {path} mode={mode}")
    print(info)
    missing = [t for t in tickers if t not in set(combined["ticker"])]
    if missing:
        print(f"missing {len(missing)} tickers (not invented): {missing[:15]}{'…' if len(missing) > 15 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
