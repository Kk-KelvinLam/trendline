# Trendline

Post-US-close dashboard that forecasts the **next regular session** High / Low / Close for S&P 500 names.

美股收市後儀表板：預測 **下一常規交易時段** 的高 / 低 / 收。**全數 S&P 500 成分股訓練**，按前一交易日成交額（收市價 × 成交量）排序，**顯示最多 100 隻**。

**免責聲明 / Disclaimer:** 本頁為量化模型輸出，**並非投資建議**。過往回測不代表未來表現。
Model output, not investment advice. Past backtests do not predict future results.

---

## What it does

| 項目 | 說明 |
| --- | --- |
| Universe | `data/universe/sp500.csv` from Wikipedia. Refreshed on the **first Sunday of each month** during the Sunday retrain. Default fetch = **all members + macros** (SPY, VIX, sector ETFs). |
| Train / show | Train on the full downloaded membership; dashboard cards shortlist **top 100** by prior-day dollar volume. |
| Model families | Three LightGBM quantile families on the **same** purged walk-forward folds: **shared** (panel), **sector** (one model per large GICS sector; small sectors → `Other`), **per-stock** (smaller trees; tickers with &lt; 400 train rows fall back to shared). |
| Quantiles | q10 / q50 / q90 on *returns vs prior close*, then converted to price levels. No LSTM. |
| Features | Known at prior close only: 1/5/20d returns, overnight gap, ATR, Parkinson vol, dollar-volume z-score, distance to 20/50/200 MAs, plus SPY / VIX / sector ETF. |
| Baseline | Close = prior close; High / Low = prior close ± 1×ATR. |
| Cards | Three JSON sets. **Fade to prior close:** touch predicted High/Low q50 during the next session, TP = prior close, SL = q10/q90 or 1×ATR. Gate on High/Low beating ATR, not Close direction. No chase if the open gaps through. |
| Scoreboard | `scoreboard.json`: per family × High/Low/Close × fold — MAE$, MAPE, coverage (+ overall). |
| Backtest | Expanding walk-forward with a **5-session purge** between train and test. |

---

## UI pages (zh-HK)

Streamlit radio:

1. **共用模型** — shared panel cards  
2. **行業模型** — sector-family cards  
3. **個股模型** — per-stock (with shared fallback) cards  
4. **計分板** — fold-by-fold High/Low/Close MAE horse race  
5. **流水** — 模擬戶口 HK$500,000；共用淡區間、每邊 $2、三族實績每日更新  

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py   # fetch (if needed) → three-family WF → cards
streamlit run app/streamlit_app.py
```

Step-by-step:

```bash
python scripts/fetch.py          # full S&P + macros → data/parquet/ohlcv.parquet
python scripts/backtest.py       # three-family walk-forward + metrics + cards + scoreboard
streamlit run app/streamlit_app.py
```

`python scripts/fetch.py --full` is kept as an alias for the same full universe.

If the network blocks Yahoo/Stooq, **do not invent prices**. Drop a parquet with columns `date,ticker,open,high,low,close,adj_close,volume` at `data/parquet/ohlcv.parquet` and re-run `backtest.py`.

---

## Artifacts

```
data/artifacts/
  cards.json              # = cards_shared (backward compat)
  cards_shared.json
  cards_sector.json
  cards_stock.json
  metrics.json            # primary (shared) + scoreboard_headline
  scoreboard.json         # full horse race
  per_ticker.json
  per_ticker_{shared,sector,stock}.json
  models/
    shared/*.txt
    sector/<Sector>/*.txt
    stock/<TICKER>/*.txt
```

Model dumps under `data/artifacts/models/` are **tracked** (needed for weekday nightly infer). Large parquets stay gitignored.

---

## Nightly update (GitHub Actions)

| When | What |
| --- | --- |
| 23:00 UTC Mon–Fri | **Delta** OHLCV in the runner + infer three card files. Does **not** commit `ohlcv.parquet`. |
| 10:00 UTC Sunday | Three-family walk-forward retrain. **First Sunday of the month:** refresh S&P list + **full** OHLCV pull and commit the single parquet (housekeep: ~12 blobs/year). Other Sundays keep delta and skip the parquet commit. |
| Actions → Nightly cards → Run workflow | Manual, optional full retrain |

No market-data API key. Yahoo first, Stooq fallback.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers no-leakage feature timing, backtest helpers, sector assignment, and thin-ticker fallback to shared.

---

## v2 backtest (computed, not invented)

Run on this machine, 2026-09-12. Real Yahoo/Stooq daily OHLCV: **516 symbols** (full S&P membership + macros), 473,671 rows, 2023-01-03 → 2026-09-11. Walk-forward: **7** purged folds, 220,055 OOS rows, 500 names. Three families on the same folds.

| Family | High MAE $ | Low MAE $ | Close MAE $ |
| --- | ---: | ---: | ---: |
| shared | 2.43 | 2.56 | 3.26 |
| sector | 2.43 | 2.57 | 3.26 |
| stock | 2.44 | 2.57 | 3.27 |
| baseline (ATR / RW) | 4.06 | 4.21 | 3.25 |

High/Low: all three families beat the ATR baseline by a wide margin. Close remains near random-walk (shared 3.261 vs baseline 3.253). Shared is the slight High/Low winner on this run. Live top-100 cards (asof 2026-09-11): shared all 觀望; sector 17 long; stock 20 long / 1 short.

---

## Layout

```
src/trendline/     universe, data, features, models (shared/sector/stock), backtest, cards
app/               Streamlit dashboard (zh-HK)
scripts/           fetch, train, backtest, cards, nightly, run_pipeline
data/universe/     S&P 500 ticker list
data/artifacts/    cards*, metrics, scoreboard, models/
tests/
```

## Railway

Production image serves Streamlit with precomputed card/metrics/scoreboard JSON. Dockerfile copies all three card files + metrics + scoreboard.
