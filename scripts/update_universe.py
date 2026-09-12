#!/usr/bin/env python3
"""Refresh data/universe/sp500.csv from Wikipedia. Never invents members."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = ROOT / "data" / "universe" / "sp500.csv"


def _yahoo_ticker(symbol: str) -> str:
    # Wikipedia uses BRK.B; Yahoo uses BRK-B.
    return str(symbol).strip().replace(".", "-")


def fetch_sp500(today: date | None = None) -> "object":
    import pandas as pd

    today = today or date.today()
    import io
    import requests

    r = requests.get(
        WIKI,
        headers={"User-Agent": "trendline/0.2 (research; +https://github.com/Kk-KelvinLam/trendline)"},
        timeout=30,
    )
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text), flavor="lxml")
    if not tables:
        raise RuntimeError("Wikipedia returned no tables")
    raw = tables[0]
    cols = {str(c).strip().lower(): c for c in raw.columns}
    def col(*names: str):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"missing column {names} in {list(raw.columns)}")

    out = pd.DataFrame(
        {
            "ticker": raw[col("symbol")].map(_yahoo_ticker),
            "name": raw[col("security")],
            "sector": raw[col("gics sector", "sector")],
            "sub_industry": raw[col("gics sub-industry", "sub-industry", "gics sub industry")],
            "asof": today.isoformat(),
            "source": f"Wikipedia List of S&P 500 companies, retrieved {today.isoformat()}",
        }
    )
    out = out.dropna(subset=["ticker", "sector"])
    out = out[out["ticker"].str.len() > 0]
    out = out.drop_duplicates("ticker", keep="first")
    if len(out) < 480 or len(out) > 520:
        raise RuntimeError(f"unexpected constituent count {len(out)}")
    return out


def main() -> int:
    import pandas as pd

    new = fetch_sp500()
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["ticker"])
    old_set = set(old["ticker"].astype(str))
    new_set = set(new["ticker"].astype(str))
    added = sorted(new_set - old_set)
    dropped = sorted(old_set - new_set)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT, index=False)
    print(f"wrote {OUT} n={len(new)} asof={new['asof'].iloc[0]}")
    print(f"added {len(added)}: {added[:12]}{'…' if len(added) > 12 else ''}")
    print(f"dropped {len(dropped)}: {dropped[:12]}{'…' if len(dropped) > 12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
