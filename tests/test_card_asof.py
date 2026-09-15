"""Card asof must skip macro-only / thin sessions (e.g. Labor Day ^VIX-only)."""

from __future__ import annotations

import pandas as pd

from trendline.cards import latest_card_asof


def test_latest_card_asof_skips_vix_only_thin_day():
    # 2026-09-11 has ready S&P features; 2026-09-14 is ^VIX-only (partial fetch / holiday).
    rows = [
        {"date": "2026-09-11", "ticker": "AAPL", "ret_20d": 0.01, "atr": 1.0, "dist_ma50": 0.02},
        {"date": "2026-09-11", "ticker": "MSFT", "ret_20d": 0.02, "atr": 1.2, "dist_ma50": 0.01},
        {"date": "2026-09-11", "ticker": "^VIX", "ret_20d": 0.05, "atr": 0.8, "dist_ma50": 0.03},
        {"date": "2026-09-14", "ticker": "^VIX", "ret_20d": 0.12, "atr": 1.3, "dist_ma50": 0.06},
    ]
    featured = pd.DataFrame(rows)
    asof = latest_card_asof(featured)
    assert asof == pd.Timestamp("2026-09-11")


def test_latest_card_asof_normalizes_tz_aware_dates():
    featured = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-09-11", tz="America/New_York"),
                "ticker": "AAPL",
                "ret_20d": 0.01,
                "atr": 1.0,
                "dist_ma50": 0.0,
            },
            {
                "date": pd.Timestamp("2026-09-14", tz="America/New_York"),
                "ticker": "^VIX",
                "ret_20d": 0.1,
                "atr": 1.0,
                "dist_ma50": 0.0,
            },
        ]
    )
    asof = latest_card_asof(featured)
    assert asof.tzinfo is None
    assert asof == pd.Timestamp("2026-09-11")
