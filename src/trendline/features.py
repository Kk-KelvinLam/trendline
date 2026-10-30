"""Close-as-of features and next-day targets. No future leakage.

Every feature on date t uses only information known at t's regular-session close.
Targets are the *next* session's high / low / close as returns vs t's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendline.config import (
    ATR_PERIOD,
    DVOL_Z_WINDOW,
    FEATURE_COLS,
    PARKINSON_WINDOW,
    SECTOR_ETF,
)
from trendline.universe import sector_map


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).std(ddof=0)


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add features + next-day targets. Rows with incomplete features are kept;
    callers should dropna on FEATURE_COLS / targets as needed.
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", sort=False)
    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_20d"] = g["close"].pct_change(20)
    df["prev_close"] = g["close"].shift(1)
    df["overnight_gap"] = df["open"] / df["prev_close"] - 1.0

    prev_c = df["prev_close"]
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"] - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.groupby(df["ticker"]).transform(lambda s: _rolling_mean(s, ATR_PERIOD))
    df["atr_pct"] = df["atr"] / df["close"]

    hl2 = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan) ** 2
    park = hl2.groupby(df["ticker"]).transform(
        lambda s: np.sqrt(_rolling_mean(s, PARKINSON_WINDOW) / (4.0 * np.log(2.0)))
    )
    df["parkinson_20"] = park

    dvol = df["close"] * df["volume"]
    df["dollar_volume"] = dvol
    dvol_mean = dvol.groupby(df["ticker"]).transform(lambda s: _rolling_mean(s, DVOL_Z_WINDOW))
    dvol_std = dvol.groupby(df["ticker"]).transform(lambda s: _rolling_std(s, DVOL_Z_WINDOW))
    df["dvol_z_20"] = (dvol - dvol_mean) / dvol_std.replace(0.0, np.nan)

    for win, name in ((20, "dist_ma20"), (50, "dist_ma50"), (200, "dist_ma200")):
        ma = g["close"].transform(lambda s, w=win: _rolling_mean(s, w))
        df[name] = df["close"] / ma - 1.0

    df = _attach_macro(df)
    df = _attach_sector(df)

    # Targets: next regular session vs *today's* close. Shift is within ticker.
    df["next_open"] = g["open"].shift(-1)
    df["next_high"] = g["high"].shift(-1)
    df["next_low"] = g["low"].shift(-1)
    df["next_close"] = g["close"].shift(-1)
    df["next_date"] = g["date"].shift(-1)
    df["y_high"] = df["next_high"] / df["close"] - 1.0
    df["y_low"] = df["next_low"] / df["close"] - 1.0
    df["y_close"] = df["next_close"] / df["close"] - 1.0
    return df


def _attach_macro(df: pd.DataFrame) -> pd.DataFrame:
    spy = df.loc[df["ticker"] == "SPY", ["date", "close", "ret_1d", "ret_5d", "ret_20d"]].copy()
    if spy.empty:
        df["spy_ret_1d"] = np.nan
        df["spy_ret_5d"] = np.nan
        df["spy_ret_20d"] = np.nan
    else:
        spy = spy.rename(
            columns={
                "close": "spy_close",
                "ret_1d": "spy_ret_1d",
                "ret_5d": "spy_ret_5d",
                "ret_20d": "spy_ret_20d",
            }
        )
        df = df.merge(spy[["date", "spy_ret_1d", "spy_ret_5d", "spy_ret_20d"]], on="date", how="left")

    vix = df.loc[df["ticker"].isin(["^VIX", "VIX"]), ["date", "close"]].copy()
    if vix.empty:
        df["vix_level"] = np.nan
        df["vix_chg_1d"] = np.nan
    else:
        vix = vix.drop_duplicates("date", keep="last").sort_values("date")
        vix["vix_level"] = vix["close"]
        vix["vix_chg_1d"] = vix["close"].pct_change(1)
        df = df.merge(vix[["date", "vix_level", "vix_chg_1d"]], on="date", how="left")
    return df


def _attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    smap = sector_map()
    df["sector"] = df["ticker"].map(smap)
    df["sector_etf"] = df["sector"].map(SECTOR_ETF)

    etf_tickers = sorted({e for e in SECTOR_ETF.values() if e in set(df["ticker"])})
    if not etf_tickers:
        df["sector_ret_1d"] = np.nan
        df["sector_ret_5d"] = np.nan
        return df

    etf = df.loc[df["ticker"].isin(etf_tickers), ["date", "ticker", "ret_1d", "ret_5d"]].copy()
    etf = etf.rename(columns={"ticker": "sector_etf", "ret_1d": "sector_ret_1d", "ret_5d": "sector_ret_5d"})
    df = df.merge(etf, on=["date", "sector_etf"], how="left")
    return df


def feature_matrix(df: pd.DataFrame, dropna: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y) aligned. y has y_high/y_low/y_close."""
    cols = ["date", "ticker", *FEATURE_COLS, "y_high", "y_low", "y_close"]
    extra = [c for c in ("close", "atr", "next_open", "next_high", "next_low", "next_close", "next_date") if c in df.columns]
    use = df[cols + extra].copy()
    if dropna:
        use = use.dropna(subset=FEATURE_COLS + ["y_high", "y_low", "y_close"])
    X = use[["date", "ticker", *FEATURE_COLS]]
    y = use[["date", "ticker", "y_high", "y_low", "y_close"]]
    return X, y


def assert_no_future_in_features(raw: pd.DataFrame, featured: pd.DataFrame, ticker: str) -> None:
    """Test helper: mutating t+1 OHLC must not change features at t."""
    tdf = featured.loc[featured["ticker"] == ticker].sort_values("date")
    if len(tdf) < 5:
        raise AssertionError("not enough rows to test leakage")
    mid = tdf.iloc[len(tdf) // 2]
    asof = mid["date"]
    feat_before = mid[FEATURE_COLS].to_numpy(dtype=float)

    mutated = raw.copy()
    nxt = mutated[(mutated["ticker"] == ticker) & (mutated["date"] > asof)]
    if nxt.empty:
        raise AssertionError("no future row to mutate")
    idx = nxt.index[0]
    mutated.loc[idx, ["open", "high", "low", "close", "volume"]] = (
        mutated.loc[idx, ["open", "high", "low", "close", "volume"]] * 1.25
    )
    rebuilt = build_features(mutated)
    after = rebuilt[(rebuilt["ticker"] == ticker) & (rebuilt["date"] == asof)].iloc[0]
    feat_after = after[FEATURE_COLS].to_numpy(dtype=float)
    if not np.allclose(feat_before, feat_after, equal_nan=True):
        raise AssertionError("feature leakage: t+1 mutation changed features at t")
