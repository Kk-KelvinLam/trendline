"""Post-US-close next-day dashboard (zh-HK)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from trendline.config import ARTIFACT_DIR, CARDS_PATH, METRICS_PATH


st.set_page_config(page_title="Trendline · 美股翌日預測", page_icon="📈", layout="wide")

DISCLAIMER = "本頁為量化模型輸出，並非投資建議。過往回測不代表未來表現。Model output, not investment advice."


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_px(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:,.2f}"


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_num(x, digits=4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{digits}f}"


def main() -> None:
    st.title("Trendline")
    st.caption("美股收市後 · 下一常規交易時段預測（S&P 500）")
    st.warning(DISCLAIMER)

    cards_payload = _load_json(CARDS_PATH)
    metrics = _load_json(METRICS_PATH)
    if cards_payload is None or metrics is None:
        st.error(
            "尚未找到回測產物。請先在專案根目錄執行：\n\n"
            "`python scripts/run_pipeline.py`\n\n"
            "然後重新整理本頁。"
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("資料截止", cards_payload.get("asof", "—"))
    c2.metric("預測對象", cards_payload.get("next_session", "下一常規時段"))
    c3.metric("做多", cards_payload.get("n_long", 0))
    c4.metric("做空 / 觀望", f"{cards_payload.get('n_short', 0)} / {cards_payload.get('n_flat', 0)}")

    filt = st.radio("篩選", ["全部", "做多", "做空", "高信心"], horizontal=True)

    cards = cards_payload.get("cards", [])
    if filt == "做多":
        cards = [c for c in cards if c["action"] == "做多"]
    elif filt == "做空":
        cards = [c for c in cards if c["action"] == "做空"]
    elif filt == "高信心":
        cards = [c for c in cards if c.get("high_confidence")]

    st.subheader("翌日交易卡")
    if not cards:
        st.info("此篩選沒有卡片。")
    else:
        cols = st.columns(3)
        for i, card in enumerate(cards):
            with cols[i % 3]:
                _render_card(card)

    st.divider()
    st.subheader("樣本外回測（真實計算，非虛構）")
    data = metrics.get("data", {})
    st.caption(
        f"資料：{data.get('tickers', '?')} 隻 × 中位 {data.get('days_median', '?')} 日 "
        f"（{data.get('min_date', '?')} → {data.get('max_date', '?')}），"
        f"{metrics.get('n_rows', '?')} 條樣本外列，{metrics.get('n_tickers', '?')} 隻股票。"
    )

    m = metrics.get("model", {})
    b = metrics.get("baseline", {})
    t1, t2, t3 = st.tabs(["預測誤差 q50", "分位覆蓋 / Pinball", "交易模擬"])

    with t1:
        rows = []
        for tgt, label in (("high", "高"), ("low", "低"), ("close", "收")):
            mm, bb = m.get(tgt, {}), b.get(tgt, {})
            rows.append(
                {
                    "標的": label,
                    "模型 MAE 價": mm.get("mae_px"),
                    "基準 MAE 價": bb.get("mae_px"),
                    "模型 RMSE 價": mm.get("rmse_px"),
                    "基準 RMSE 價": bb.get("rmse_px"),
                    "模型 MAPE 價": mm.get("mape_px"),
                    "基準 MAPE 價": bb.get("mape_px"),
                    "模型 MAE 報酬": mm.get("mae_ret"),
                    "基準 MAE 報酬": bb.get("mae_ret"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"收市方向準確率（相對前收）：模型 {_fmt_pct(metrics.get('dir_acc_vs_prior_close'))}　"
            f"基準 {_fmt_pct(metrics.get('dir_acc_vs_prior_close_baseline'))}　|　"
            f"相對次日開市：模型 {_fmt_pct(metrics.get('dir_acc_vs_next_open'))}　"
            f"基準 {_fmt_pct(metrics.get('dir_acc_vs_next_open_baseline'))}"
        )
        st.caption(
            f"模型整體是否優於基準（收市 MAE）：{'是' if metrics.get('overall_beats_baseline') else '否'}　·　"
            f"個股優於基準：{metrics.get('tickers_beating_baseline')}/{metrics.get('tickers_evaluated')}"
        )

    with t2:
        rows = []
        for tgt, label in (("high", "高"), ("low", "低"), ("close", "收")):
            mm = m.get(tgt, {})
            rows.append(
                {
                    "標的": label,
                    "q10–q90 覆蓋": mm.get("coverage_q10_q90"),
                    "pinball q10": mm.get("pinball_q10"),
                    "pinball q50": mm.get("pinball_q50"),
                    "pinball q90": mm.get("pinball_q90"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t3:
        tr = metrics.get("trading", {})
        a, b1, c, d = st.columns(4)
        a.metric("成交筆數", tr.get("n_trades", 0))
        b1.metric("命中率", _fmt_pct(tr.get("hit_rate")))
        c.metric("盈虧比", _fmt_num(tr.get("payoff"), 2))
        d.metric("Sharpe（年化）", _fmt_num(tr.get("sharpe"), 2))
        e, f, g = st.columns(3)
        e.metric("最大回撤", _fmt_pct(tr.get("max_dd")))
        f.metric("周轉（筆/日）", _fmt_num(tr.get("turnover"), 2))
        g.metric("平均報酬", _fmt_pct(tr.get("avg_return")))
        daily_path = ARTIFACT_DIR / "daily_returns.parquet"
        if daily_path.exists():
            daily = pd.read_parquet(daily_path)
            daily["equity"] = (1 + daily["ret"].fillna(0)).cumprod()
            st.line_chart(daily.set_index("date")["equity"], height=220)

    folds = metrics.get("folds") or []
    if folds:
        with st.expander("Walk-forward 摺疊"):
            st.dataframe(pd.DataFrame(folds), hide_index=True, use_container_width=True)

    st.caption(f"產物產生時間（UTC）：{cards_payload.get('generated_at_utc', '—')}")


def _render_card(card: dict) -> None:
    action = card["action"]
    color = {"做多": "green", "做空": "red", "觀望": "gray"}.get(action, "gray")
    conf = " · 高信心" if card.get("high_confidence") else ""
    st.markdown(f"### {card['ticker']}  :{color}[{action}]{conf}")
    st.caption(
        f"#{card.get('dvol_rank', '—')} 成交額　·　{card.get('sector') or '—'}　·　"
        f"{'模型優於基準' if card.get('beats_baseline') else '模型未優於該股基準'}"
    )
    p = card.get("pred", {})
    st.write(
        f"前收 **{_fmt_px(card.get('prior_close'))}**　·　"
        f"預測高 {_fmt_px(p.get('high', {}).get('q50'))}　"
        f"低 {_fmt_px(p.get('low', {}).get('q50'))}　"
        f"收 {_fmt_px(p.get('close', {}).get('q50'))}"
    )
    st.caption(
        f"收市 q10/q50/q90：{_fmt_px(p.get('close', {}).get('q10'))} / "
        f"{_fmt_px(p.get('close', {}).get('q50'))} / {_fmt_px(p.get('close', {}).get('q90'))}　"
        f"（{_fmt_pct(p.get('close', {}).get('q50_ret'))}）"
    )
    if action == "觀望":
        st.info(f"入場：無　·　原因：{card.get('reason')}")
    else:
        st.success(
            f"入場：{card.get('entry')}　·　止盈 {_fmt_px(card.get('tp'))}　·　止損 {_fmt_px(card.get('sl'))}"
        )
    err = card.get("recent_error") or {}
    st.caption(
        f"近期樣本外收市誤差（{err.get('n', 0)} 日）："
        f"{_fmt_px(err.get('mae_close_px'))}　（{_fmt_pct(err.get('mae_close_ret'))}）"
    )
    st.markdown("---")


if __name__ == "__main__":
    main()
