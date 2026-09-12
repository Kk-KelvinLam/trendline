#!/usr/bin/env python3
"""Download real daily OHLCV. Never synthesizes prices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.config import OHLCV_PATH  # noqa: E402
from trendline.data.providers import CombinedProvider  # noqa: E402
from trendline.data.store import save_ohlcv, summarize  # noqa: E402
from trendline.universe import all_member_tickers, default_fetch_tickers  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch daily OHLCV into data/parquet/ohlcv.parquet")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--full", action="store_true", help="All S&P 500 names (slow; rate-limited)")
    p.add_argument("--tickers", default="", help="Comma-separated override")
    args = p.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.full:
        tickers = all_member_tickers()
        # macros still needed for features
        from trendline.config import MACRO_TICKERS

        for m in MACRO_TICKERS:
            if m not in tickers:
                tickers.append(m)
    else:
        tickers = default_fetch_tickers()

    print(f"fetching {len(tickers)} symbols from {args.start} …")
    raw = CombinedProvider().download(tickers, start=args.start, end=args.end)
    if raw.empty:
        print("ERROR: no real prices downloaded. Check network / provider availability.")
        print(f"Drop a parquet with columns date,ticker,open,high,low,close,adj_close,volume at {OHLCV_PATH}")
        return 2
    path = save_ohlcv(raw, OHLCV_PATH)
    info = summarize(raw)
    print(f"wrote {path}")
    print(info)
    missing = [t for t in tickers if t not in set(raw["ticker"])]
    if missing:
        print(f"missing {len(missing)} tickers (not invented): {missing[:15]}{'…' if len(missing) > 15 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
