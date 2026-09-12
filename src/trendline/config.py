"""Paths and model/backtest constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "universe" / "sp500.csv"
PARQUET_DIR = DATA_DIR / "parquet"
OHLCV_PATH = PARQUET_DIR / "ohlcv.parquet"
ARTIFACT_DIR = DATA_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
CARDS_PATH = ARTIFACT_DIR / "cards.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
OOS_PATH = ARTIFACT_DIR / "oos_predictions.parquet"
FEATURE_PATH = ARTIFACT_DIR / "features.parquet"

QUANTILES = (0.10, 0.50, 0.90)
TARGETS = ("high", "low", "close")

# Baseline: High/Low = prior close ± k * ATR
BASELINE_ATR_K = 1.0
ATR_PERIOD = 14
PARKINSON_WINDOW = 20
DVOL_Z_WINDOW = 20

# LightGBM — small enough for a laptop walk-forward
LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "force_col_wise": True,
}
LGB_N_ESTIMATORS = 180
LGB_EARLY_STOPPING = 30

# Walk-forward (trading days)
WF_MIN_TRAIN_DAYS = 252
WF_TEST_DAYS = 63
WF_PURGE_DAYS = 5

# Trade filters
DIR_RET_MIN = 0.0015  # stand-aside if |q50 close ret| below this
RANGE_ATR_MIN = 0.45  # stand-aside if (q90h-q10l)*close < this * ATR
MAX_POSITIONS = 10
ATR_SL_MULT = 1.0

SHOW_TOP_N = 100

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

MACRO_TICKERS = (
    "SPY",
    "^VIX",
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLY",
    "XLP",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
)

# High-liquidity S&P names used as the default fetch/train subset.
# Full membership list lives in data/universe/sp500.csv.
DEFAULT_TRAIN_TICKERS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B",
    "AVGO", "JPM", "UNH", "XOM", "LLY", "V", "MA", "COST", "HD", "PG", "JNJ",
    "WMT", "NFLX", "ORCL", "ABBV", "CRM", "BAC", "KO", "PEP", "CVX", "MRK",
    "AMD", "ADBE", "TMO", "CSCO", "ACN", "LIN", "MCD", "ABT", "WFC", "DIS",
    "GE", "CAT", "INTU", "QCOM", "IBM", "AMAT", "NOW", "TXN", "VZ", "NEE",
    "PM", "RTX", "SPGI", "ISRG", "BKNG", "HON", "PFE", "GS", "LOW", "UNP",
    "AMGN", "BLK", "PLD", "C", "MS", "ETN", "UBER", "TJX", "CMCSA", "SCHW",
    "COP", "MDT", "ADP", "DE", "LMT", "SBUX", "GILD", "PANW", "BMY", "MMC",
)

FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "overnight_gap",
    "atr_pct",
    "parkinson_20",
    "dvol_z_20",
    "dist_ma20",
    "dist_ma50",
    "dist_ma200",
    "spy_ret_1d",
    "spy_ret_5d",
    "spy_ret_20d",
    "vix_level",
    "vix_chg_1d",
    "sector_ret_1d",
    "sector_ret_5d",
]
