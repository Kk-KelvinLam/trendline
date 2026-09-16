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
MODEL_SHARED_DIR = MODEL_DIR / "shared"
MODEL_SECTOR_DIR = MODEL_DIR / "sector"
MODEL_STOCK_DIR = MODEL_DIR / "stock"
CARDS_PATH = ARTIFACT_DIR / "cards.json"
CARDS_SHARED_PATH = ARTIFACT_DIR / "cards_shared.json"
CARDS_SECTOR_PATH = ARTIFACT_DIR / "cards_sector.json"
CARDS_STOCK_PATH = ARTIFACT_DIR / "cards_stock.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SCOREBOARD_PATH = ARTIFACT_DIR / "scoreboard.json"
OOS_PATH = ARTIFACT_DIR / "oos_predictions.parquet"
FEATURE_PATH = ARTIFACT_DIR / "features.parquet"
LEDGER_PATH = ARTIFACT_DIR / "ledger.json"

QUANTILES = (0.10, 0.50, 0.90)
TARGETS = ("high", "low", "close")
MODEL_FAMILIES = ("shared", "sector", "stock")

# Baseline: High/Low = prior close ± k * ATR
BASELINE_ATR_K = 1.0
ATR_PERIOD = 14
PARKINSON_WINDOW = 20
DVOL_Z_WINDOW = 20

# LightGBM — panel / sector (slightly lean for full-universe walk-forward)
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
LGB_N_ESTIMATORS = 120
LGB_EARLY_STOPPING = 25

# Per-stock: smaller trees; skip thin history
STOCK_LGB_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_data_in_leaf": 25,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": 1,
    "force_col_wise": True,
}
STOCK_N_ESTIMATORS = 80
STOCK_MIN_TRAIN_ROWS = 400
STOCK_EARLY_STOPPING = 20

# Sector split: GICS with enough Wikipedia-snapshot members get own model
SECTOR_MIN_NAMES = 20

# Walk-forward (trading days)
WF_MIN_TRAIN_DAYS = 252
WF_TEST_DAYS = 63
WF_PURGE_DAYS = 5

# Trade filters
DIR_RET_MIN = 0.0015  # unused for entries; kept for reference
RANGE_ATR_MIN = 0.45  # stand-aside if (q90h-q10l)*close < this * ATR
FADE_MIN_ATR = 0.30  # need this much room from High/Low trigger back to prior close
CLOSE_GATE_MIN_RET = 0.001  # |Close q50| min for sector/stock Close gate (shared skips Close gate)
CLOSE_GATE_FAMILIES = ("sector", "stock")  # shared Close model has no usable |q50c|
RECENT_ERROR_WINDOW = 20  # trading days for card recent Close MAE
RECENT_ERROR_MIN_N = 10  # need this many labeled days to trust recent MAE
RECENT_CLOSE_MAE_MAX = 0.025  # hard flat if recent Close MAE (ret) above this
RECENT_CLOSE_MAE_SOFT = 0.018  # strip high_confidence above this
MAX_POSITIONS = 10
PAPER_STARTING_HKD = 500_000.0
PAPER_FX_HKD_PER_USD = 7.80
PAPER_FEE_USD_PER_ORDER = 2.0  # unused; paper costs follow IBKR Pro Fixed
PAPER_ORDERS_PER_ROUNDTRIP = 2  # unused
PAPER_BROKER = "IBKR Pro Fixed"
PAPER_MAX_POSITIONS = 20
PAPER_NOTIONAL_FRAC = 0.07  # unused; kept for reference
PAPER_GROSS_FRAC = 1.00  # never deploy more than today's equity
PAPER_MAX_NAME_FRAC = 0.12
PAPER_MIN_NOTIONAL_USD = 2500.0
PAPER_DAILY_RISK_FRAC = 0.05
PAPER_MAX_NAME_RISK = 0.006
PAPER_FAMILY = "shared"
ATR_SL_MULT = 1.0
# Paper ledger: only open if the trigger prints before this NY time (bar start).
# Management (TP/SL/close) after fill can continue through the RTH close.
PAPER_ENTRY_CUTOFF_ET = (12, 30)

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

PAPER_BENCHMARK_TICKER = "VOO"  # buy-and-hold mark-to-close vs the three fade books

MACRO_TICKERS = (
    "SPY",
    "VOO",
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

# Kept for reference / tests; default fetch is now the full S&P list.
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


def recent_close_error_path(family: str = "shared") -> Path:
    """Compact per-ticker recent Close MAE JSON (committed; not full OOS parquet)."""
    return ARTIFACT_DIR / f"recent_close_error_{family}.json"
