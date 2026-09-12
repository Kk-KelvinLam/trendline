#!/usr/bin/env python3
"""One command: fetch (if needed) → walk-forward → cards.

    python scripts/run_pipeline.py
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendline.config import OHLCV_PATH  # noqa: E402


def main() -> int:
    py = sys.executable
    if not OHLCV_PATH.exists():
        print("no parquet — fetching real prices…")
        rc = subprocess.call([py, str(ROOT / "scripts" / "fetch.py")])
        if rc != 0:
            return rc
    rc = subprocess.call([py, str(ROOT / "scripts" / "backtest.py")])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
