# Trendline

Post-US-close dashboard that forecasts the **next regular session** High / Low / Close for S&P 500 names.

美股收市後儀表板：預測 **下一常規交易時段** 的高 / 低 / 收。只顯示 S&P 500 成分股，按前一交易日成交額（收市價 × 成交量）排序，最多 100 隻。

**免責聲明 / Disclaimer:** 本頁為量化模型輸出，**並非投資建議**。過往回測不代表未來表現。
Model output, not investment advice. Past backtests do not predict future results.

---

## What it does

| 項目 | 說明 |
| --- | --- |
| Universe | Static S&P 500 membership list (`data/universe/sp500.csv`, Wikipedia snapshot **2026-09-12**). Macros: SPY, VIX, sector ETFs. |
| Model | LightGBM **quantile regression** q10 / q50 / q90 on *returns vs prior close*, then converted to price levels. No LSTM. |
| Features | Known at prior close only: 1/5/20d returns, overnight gap, ATR, Parkinson vol, dollar-volume z-score, distance to 20/50/200 MAs, plus SPY / VIX / sector ETF. |
| Baseline | Close = prior close; High / Low = prior close ± 1×ATR. |
| Cards | 做多 / 做空 / 觀望. Entry = next open. TP = q50 high (long) or q50 low (short). SL = q10 low / q90 high **or** 1×ATR (tighter). Long/short only if the model beat the baseline out-of-sample on that ticker **or** overall; otherwise 觀望. Stand-aside when \|q50 close\| is tiny or the q10–q90 range is tight vs ATR. |
| Backtest | Expanding walk-forward with a **5-session purge** between train and test. |

---

## One-command dashboard

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py   # fetch (if needed) → walk-forward → cards
streamlit run app/streamlit_app.py
```

If `data/artifacts/cards.json` and `metrics.json` are already present, the UI can start immediately. Re-run the pipeline to refresh on new prices.

Step-by-step:

```bash
python scripts/fetch.py          # real OHLCV → data/parquet/ohlcv.parquet
python scripts/backtest.py       # walk-forward + metrics + cards
# or separately:
python scripts/train.py
python scripts/cards.py
streamlit run app/streamlit_app.py
```

`python scripts/fetch.py --full` downloads every S&P name (slow, rate-limited). Default is ~80 liquid members + SPY/VIX/sector ETFs.

If the network blocks Yahoo/Stooq, **do not invent prices**. Drop a parquet with columns `date,ticker,open,high,low,close,adj_close,volume` at `data/parquet/ohlcv.parquet` and re-run `backtest.py`.

---

## v1 backtest (computed, not invented)

Run on this machine, 2026-09-12. Real Yahoo daily OHLCV: 92 symbols, 75,718 rows, 2023-06-01 → 2026-09-11. Walk-forward: 5 purged folds, 24,885 OOS rows, 79 names after 200-day warmup.

| Target | Model MAE $ | Baseline MAE $ | Model MAPE | q10–q90 coverage |
| --- | ---: | ---: | ---: | ---: |
| High | 2.90 | 4.81 | 1.05% | 76.3% |
| Low | 3.04 | 4.96 | 1.14% | 76.5% |
| Close | 3.81 | 3.81 | 1.40% | 75.5% |

Close is a near random walk (MAE 3.813 vs 3.814). Directional accuracy vs prior close 51.6%; vs next open 50.8%. Trading sim: 90 trades, hit 43.3%, payoff 0.72, Sharpe −8.0, max DD −2.8%. Live cards for 2026-09-11: all 觀望 (`probability_too_close`) — High/Low is where the model adds value.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers no-leakage feature timing and backtest/metric helpers.

---

## Layout

```
src/trendline/     universe, data providers, features, models, backtest, cards
app/               Streamlit dashboard (zh-HK)
scripts/           fetch, train, backtest, cards, run_pipeline
data/universe/     S&P 500 ticker list
data/artifacts/    metrics.json, cards.json, per_ticker.json
tests/
```

OHLCV parquet and model dumps are gitignored (re-fetch / retrain locally).

## Railway

Production image serves the Streamlit board only (precomputed `data/artifacts/*.json`). Dockerfile + `railway.toml` are in the repo. Start command binds `0.0.0.0:$PORT`.
