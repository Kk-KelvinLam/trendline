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
| Cards | Three JSON sets. **Fade to prior close:** touch predicted High/Low q50 during the next session, TP = prior close, SL = q10/q90 or 1×ATR. Range gate: High/Low must beat the ATR baseline (ticker or overall). **Close gate differs by family:** shared skips it; sector/stock need Close q50 ≥ +10bps to go long or ≤ −10bps to go short. **Recent Close MAE** (20 sessions, ≥10 labels) above 2.5% hard-flats the card; above 1.8% strips `high_confidence`. No chase if the open gaps through. Families use different rules — do not rank models off the live books. |
| Paper ledger | Three HK$500k books. Realize against the **next** session using RTH **5-minute** bars only. Entry must print **before 12:30 America/New_York**; later prints and missing 5m are misses (no daily-OHLC fallback). Flatten same day. Headline hit rate is on all directional *signals* (pre-fee, including names not sized into the book). |
| Scoreboard | `scoreboard.json`: per family × High/Low/Close × fold — MAE$, MAPE, coverage (+ overall). |
| Backtest | Expanding walk-forward with a **5-session purge** between train and test. WF trade sim still uses daily OHLC fills and does not apply the paper 12:30 cutoff. |

---

## UI pages (zh-HK)

Streamlit radio:

1. **共用模型** — shared panel cards（無 Close 方向閘；Close MAE hard-flat 仍適用）  
2. **行業模型** — sector-family cards（Close q50 ±10bps 閘）  
3. **個股模型** — per-stock（Close q50 ±10bps 閘；薄歷史 fallback shared）  
4. **釘選對照** — pin tickers and compare the three families side by side  
5. **計分板** — fold-by-fold High/Low/Close MAE horse race  
6. **流水** — 三本模擬戶口各 HK$500,000；只認 NY RTH 5 分鐘 bar，12:30 ET 前到價先入場；缺 5m 當錯過  

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
  ledger.json             # paper books (three families)
  recent_close_error_{shared,sector,stock}.json
  metrics.json            # primary (shared) + scoreboard_headline
  scoreboard.json         # full horse race
  per_ticker.json
  per_ticker_{shared,sector,stock}.json
  models/
    shared/*.txt
    sector/<Sector>/*.txt
    stock/<TICKER>/*.txt
```

Model dumps under `data/artifacts/models/` are **tracked** (needed for weekday nightly infer). `oos_predictions.parquet` / features / trades stay gitignored. `data/parquet/ohlcv.parquet` is force-tracked and committed only on the monthly full pull.

---

## Nightly update (GitHub Actions)

| When | What |
| --- | --- |
| 04:15 UTC / 12:15 HKT Tue–Sat | **Delta** OHLCV in the runner + infer three card files + **paper ledger** realize (RTH **5-minute** fills only; entry before 12:30 ET). Later slot so Yahoo usually has real Close. Does **not** commit `ohlcv.parquet`. |
| 11:15 UTC / 19:15 HKT Sunday | Three-family walk-forward retrain. **First Sunday of the month:** refresh S&P list + **full** OHLCV pull and commit the single parquet (housekeep: ~12 blobs/year). Other Sundays keep delta and skip the parquet commit. |
| Actions → Nightly cards → Run workflow | Manual, optional full retrain |

The workflow file still checks out and pushes `feat/v1-dashboard`. Dashboard code already lives on `main`; until that ref is flipped, weekday card/ledger commits may not land on the deployed branch.

No market-data API key. Yahoo first, Stooq fallback.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers no-leakage feature timing, backtest helpers, sector assignment, and thin-ticker fallback to shared.

---

## v2 backtest snapshot (computed, not invented)

Snapshot from 2026-09-12 (not live cards). Real Yahoo/Stooq daily OHLCV: **516 symbols** (full S&P membership + macros), 473,671 rows, 2023-01-03 → 2026-09-11. Walk-forward: **7** purged folds, 220,055 OOS rows, 500 names. Three families on the same folds.

| Family | High MAE $ | Low MAE $ | Close MAE $ |
| --- | ---: | ---: | ---: |
| shared | 2.43 | 2.56 | 3.26 |
| sector | 2.43 | 2.57 | 3.26 |
| stock | 2.44 | 2.57 | 3.27 |
| baseline (ATR / RW) | 4.06 | 4.21 | 3.25 |

High/Low: all three families beat the ATR baseline by a wide margin. Close remains near random-walk (shared 3.261 vs baseline 3.253). Shared is the slight High/Low winner on this run. Live top-100 cards **that day** (asof 2026-09-11): shared all 觀望; sector 17 long; stock 20 long / 1 short — a snapshot, not a live count.

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

Production image serves Streamlit with precomputed JSON. Dockerfile copies the three family card files, `cards.json`, metrics, scoreboard, `per_ticker.json`, and `ledger.json`.
