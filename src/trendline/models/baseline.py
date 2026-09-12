"""ATR-band baseline.

Close forecast = prior close (return 0).
High / Low = prior close ± k × ATR, expressed as returns vs prior close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendline.config import BASELINE_ATR_K


class BaselineModel:
    name = "baseline"

    def __init__(self, k: float = BASELINE_ATR_K) -> None:
        self.k = float(k)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """frame must include close, atr (same-day, known at close)."""
        atr = frame["atr"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        atr_ret = np.where(close > 0, self.k * atr / close, np.nan)
        out = pd.DataFrame(index=frame.index)
        out["base_high_q50"] = atr_ret
        out["base_low_q50"] = -atr_ret
        out["base_close_q50"] = 0.0
        # Wide bands for coverage comparison: ±1.6 ATR ~ roughly 10/90-ish
        out["base_high_q10"] = 0.4 * atr_ret
        out["base_high_q90"] = 1.6 * atr_ret
        out["base_low_q10"] = -1.6 * atr_ret
        out["base_low_q90"] = -0.4 * atr_ret
        out["base_close_q10"] = -atr_ret
        out["base_close_q90"] = atr_ret
        return out
