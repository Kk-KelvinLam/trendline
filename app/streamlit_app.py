"""Post-US-close next-day dashboard (zh-HK) — three model families + scoreboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from trendline.config import (
    ARTIFACT_DIR,
    CARDS_PATH,
    CARDS_SECTOR_PATH,
    CARDS_SHARED_PATH,
    CARDS_STOCK_PATH,
    METRICS_PATH,
    SCOREBOARD_PATH,
    LEDGER_PATH,
)
from trendline.ledger import ledger_view


st.set_page_config(page_title="Trendline · 美股翌日預測", page_icon="📈", layout="wide")

DISCLAIMER = "本頁為量化模型輸出，並非投資建議。過往回測不代表未來表現。Model output, not investment advice."

FAMILY_PAGES = {
    "共用模型": ("shared", CARDS_SHARED_PATH),
    "行業模型": ("sector", CARDS_SECTOR_PATH),
    "個股模型": ("stock", CARDS_STOCK_PATH),
    "計分板": ("scoreboard", None),
    "流水": ("ledger", None),
}


def _load_json(path: Path) -> dict | None:
    if path is None or not path.exists():
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



def _inject_back_to_top(*, jump: bool) -> None:
    """Fixed ↑ in the Streamlit page. Hide only when we know we are at the top."""
    import streamlit.components.v1 as components

    st.markdown(
        """
<style>
#tl-top { position: relative; top: -8px; height: 1px; }
#tl-arrow {
  position: fixed !important;
  right: 20px;
  bottom: 20px;
  z-index: 999999;
  width: 44px;
  height: 44px;
  display: flex !important;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #1f2937;
  color: #fff !important;
  text-decoration: none !important;
  font-size: 22px;
  line-height: 1;
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
}
#tl-arrow.tl-hide { display: none !important; }
</style>
<div id="tl-top"></div>
<a id="tl-arrow" href="#tl-top" aria-label="回到頁頂">↑</a>
""",
        unsafe_allow_html=True,
    )
    flag = "1" if jump else "0"
    components.html(
        f"""
<script>
(function() {{
  const win = window.parent;
  const doc = win.document;
  try {{ win.history.scrollRestoration = "manual"; }} catch (e) {{}}
  var orphan = doc.getElementById("tl-back-top");
  if (orphan) orphan.remove();
  function arrow() {{
    return doc.getElementById("tl-arrow") || document.getElementById("tl-arrow");
  }}
  function scrollers() {{
    const found = [
      doc.scrollingElement, doc.documentElement, doc.body,
      doc.querySelector(".stApp"),
      doc.querySelector('[data-testid="stAppViewContainer"]'),
      doc.querySelector('[data-testid="stAppScrollToBottomContainer"]'),
      doc.querySelector("section.main"),
    ];
    return found.filter(Boolean);
  }}
  function hardTop() {{
    scrollers().forEach(function(el) {{ el.scrollTop = 0; }});
    try {{ win.scrollTo(0, 0); }} catch (e) {{}}
    var top = doc.getElementById("tl-top");
    if (top && top.scrollIntoView) top.scrollIntoView();
  }}
  function maxY() {{
    var m = win.scrollY || 0;
    scrollers().forEach(function(el) {{ m = Math.max(m, el.scrollTop || 0); }});
    return m;
  }}
  function sync() {{
    var a = arrow();
    if (!a) return;
    var tall = scrollers().some(function(el) {{ return el.scrollHeight > el.clientHeight + 80; }});
    if (tall && maxY() <= 80) a.classList.add("tl-hide");
    else a.classList.remove("tl-hide");
  }}
  if (!win.__tlArrowTimer) {{
    win.__tlArrowTimer = win.setInterval(sync, 250);
  }}
  if (!win.__tlPinnedTop) {{
    win.__tlPinnedTop = true;
    hardTop();
    [80, 250, 600].forEach(function(ms) {{ win.setTimeout(hardTop, ms); }});
  }}
  if ("{flag}" === "1") hardTop();
  sync();
}})();
</script>
""",
        height=0,
    )


def main() -> None:
    _inject_back_to_top(jump=False)
    st.title("Trendline")
    st.caption("美股收市後 · 盤中觸價淡區間（止盈前收）。S&P 500 全數訓練 / 顯示前 100 成交額")
    st.warning(DISCLAIMER)

    page = st.radio("頁面", list(FAMILY_PAGES.keys()), horizontal=True)
    jumped = st.session_state.get("_page") != page
    st.session_state["_page"] = page
    _inject_back_to_top(jump=jumped)
    key, cards_path = FAMILY_PAGES[page]

    metrics = _load_json(METRICS_PATH)
    scoreboard = _load_json(SCOREBOARD_PATH)

    if key == "scoreboard":
        _render_scoreboard(metrics, scoreboard)
        return
    if key == "ledger":
        _render_ledger()
        return

    cards_payload = _load_json(cards_path) or ( _load_json(CARDS_PATH) if key == "shared" else None )
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
    st.caption(f"模型家族：{cards_payload.get('model_family_zh') or page}")

    filt = st.radio("篩選", ["全部", "做多", "做空", "高信心"], horizontal=True, key=f"filt_{key}")

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
    _render_family_metrics(metrics, key)
    st.caption(f"產物產生時間（UTC）：{cards_payload.get('generated_at_utc', '—')}")


def _render_family_metrics(metrics: dict, family: str) -> None:
    st.subheader("樣本外回測（真實計算，非虛構）")
    data = metrics.get("data", {})
    st.caption(
        f"資料：{data.get('tickers', '?')} 隻 × 中位 {data.get('days_median', '?')} 日 "
        f"（{data.get('min_date', '?')} → {data.get('max_date', '?')}），"
        f"{metrics.get('n_rows', '?')} 條樣本外列，{metrics.get('n_tickers', '?')} 隻股票。"
    )

    headline = (metrics.get("scoreboard_headline") or {}).get(family)
    base_h = (metrics.get("scoreboard_headline") or {}).get("baseline")
    if headline:
        rows = []
        for tgt, label in (("high", "高"), ("low", "低"), ("close", "收")):
            mm = headline.get(tgt, {})
            bb = (base_h or {}).get(tgt, {})
            rows.append(
                {
                    "標的": label,
                    "模型 MAE 價": mm.get("mae_px"),
                    "基準 MAE 價": bb.get("mae_px"),
                    "模型 MAPE 價": mm.get("mape_px"),
                    "基準 MAPE 價": bb.get("mape_px"),
                    "覆蓋 q10–q90": mm.get("coverage"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif family == "shared":
        m = metrics.get("model", {})
        b = metrics.get("baseline", {})
        rows = []
        for tgt, label in (("high", "高"), ("low", "低"), ("close", "收")):
            mm, bb = m.get(tgt, {}), b.get(tgt, {})
            rows.append(
                {
                    "標的": label,
                    "模型 MAE 價": mm.get("mae_px"),
                    "基準 MAE 價": bb.get("mae_px"),
                    "模型 MAPE 價": mm.get("mape_px"),
                    "基準 MAPE 價": bb.get("mape_px"),
                    "覆蓋 q10–q90": mm.get("coverage_q10_q90"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if family == "shared":
        tr = metrics.get("trading", {})
        a, b1, c, d = st.columns(4)
        a.metric("成交筆數", tr.get("n_trades", 0))
        b1.metric("命中率", _fmt_pct(tr.get("hit_rate")))
        c.metric("盈虧比", _fmt_num(tr.get("payoff"), 2))
        d.metric("Sharpe（年化）", _fmt_num(tr.get("sharpe"), 2))


def _render_scoreboard(metrics: dict | None, scoreboard: dict | None) -> None:
    st.subheader("計分板 · 三族模型樣本外誤差")
    st.caption("同一 walk-forward 摺疊；焦點在 High / Low，同時報告 Close。數字全部來自真實回測。")

    if scoreboard is None and metrics and metrics.get("scoreboard_headline"):
        # Minimal board from metrics headline
        headline = metrics["scoreboard_headline"]
        rows = []
        for fam, label in (
            ("shared", "共用"),
            ("sector", "行業"),
            ("stock", "個股"),
            ("baseline", "基準"),
        ):
            block = headline.get(fam) or {}
            rows.append(
                {
                    "模型": label,
                    "High MAE$": (block.get("high") or {}).get("mae_px"),
                    "Low MAE$": (block.get("low") or {}).get("mae_px"),
                    "Close MAE$": (block.get("close") or {}).get("mae_px"),
                    "High MAPE": (block.get("high") or {}).get("mape_px"),
                    "Low MAPE": (block.get("low") or {}).get("mape_px"),
                    "Close MAPE": (block.get("close") or {}).get("mape_px"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        return

    if scoreboard is None:
        st.error("尚未找到 scoreboard.json。請先執行 `python scripts/backtest.py`。")
        return

    families = scoreboard.get("families") or {}
    baseline = scoreboard.get("baseline") or {}

    # Overall horse race
    rows = []
    for fam, label in (("shared", "共用"), ("sector", "行業"), ("stock", "個股")):
        ov = (families.get(fam) or {}).get("overall") or {}
        rows.append(
            {
                "模型": label,
                "High MAE$": (ov.get("high") or {}).get("mae_px"),
                "Low MAE$": (ov.get("low") or {}).get("mae_px"),
                "Close MAE$": (ov.get("close") or {}).get("mae_px"),
                "High 覆蓋": (ov.get("high") or {}).get("coverage_q10_q90"),
                "Low 覆蓋": (ov.get("low") or {}).get("coverage_q10_q90"),
                "Close 覆蓋": (ov.get("close") or {}).get("coverage_q10_q90"),
            }
        )
    bov = baseline.get("overall") or {}
    rows.append(
        {
            "模型": "基準 ATR/RW",
            "High MAE$": (bov.get("high") or {}).get("mae_px"),
            "Low MAE$": (bov.get("low") or {}).get("mae_px"),
            "Close MAE$": (bov.get("close") or {}).get("mae_px"),
            "High 覆蓋": (bov.get("high") or {}).get("coverage_q10_q90"),
            "Low 覆蓋": (bov.get("low") or {}).get("coverage_q10_q90"),
            "Close 覆蓋": (bov.get("close") or {}).get("coverage_q10_q90"),
        }
    )
    st.markdown("#### 整體")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Fold-by-fold
    st.markdown("#### 各摺疊 High / Low / Close MAE$")
    fold_rows = []
    # Collect fold ids
    fold_ids = set()
    for fam in ("shared", "sector", "stock"):
        for fr in (families.get(fam) or {}).get("folds") or []:
            fold_ids.add(fr["fold_id"])
    for fid in sorted(fold_ids):
        row = {"摺疊": fid}
        for fam, label in (("shared", "共用"), ("sector", "行業"), ("stock", "個股")):
            frs = {f["fold_id"]: f for f in (families.get(fam) or {}).get("folds") or []}
            fr = frs.get(fid) or {}
            for tgt, short in (("high", "H"), ("low", "L"), ("close", "C")):
                row[f"{label}{short}"] = ((fr.get(tgt) or {}).get("mae_px"))
        brs = {f["fold_id"]: f for f in baseline.get("folds") or []}
        br = brs.get(fid) or {}
        for tgt, short in (("high", "H"), ("low", "L"), ("close", "C")):
            row[f"基準{short}"] = ((br.get(tgt) or {}).get("mae_px"))
        fold_rows.append(row)
    if fold_rows:
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)

    logs = scoreboard.get("fold_logs") or (metrics or {}).get("folds") or []
    if logs:
        with st.expander("Walk-forward 摺疊設定"):
            st.dataframe(pd.DataFrame(logs), hide_index=True, use_container_width=True)


def _fmt_hkd(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"HK${x:,.0f}"


def _render_ledger() -> None:
    st.subheader("流水 · 三戶口賽馬")
    st.warning("紙上模擬，未接券商。三個模型各 HK$500,000。入場當日一定平倉，未中止盈／止損就用當日收市價出場，唔留過夜。")
    st.caption("對賬路徑：America/New_York 常規時段（09:30–16:00）5 分鐘 bar 成交；若個別股份缺 5m 資料則回退日 K OHLC（fill_source=5m / daily）。Walk-forward OOS 仍用日 K。")
    view = ledger_view()
    ledger = view["ledger"]
    acct = ledger.get("account") or {}
    headlines = view.get("headlines") or {}
    labels = (("shared", "共用"), ("sector", "行業"), ("stock", "個股"))
    cols = st.columns(3)
    for col, (fam, label) in zip(cols, labels):
        h = headlines.get(fam) or {}
        with col:
            st.metric(f"{label}權益", _fmt_hkd(h.get("equity_hkd")), delta=_fmt_hkd(h.get("pnl_hkd")))
    st.caption(
        f"各本金 {_fmt_hkd(acct.get('starting_equity_hkd'))}　·　按止損風險分倉　·　"
        f"按當日權益、支出不可超過當日權益　·　全日 SL 5%、單隻風險 0.6%、名義 12%　·　"
        f"{acct.get('broker') or 'IBKR Pro Fixed'}　$0.005/股（每單最少 $1，賣出加 SEC/FINRA）　·　"
        f"匯率 {float(acct.get('fx_hkd_per_usd') or 0):.3f} HKD/USD　·　"
        f"已實現時段 {len(ledger.get('realized_asofs') or [])}"
    )
    st.caption(f"更新（UTC）：{ledger.get('updated_at_utc') or '尚未有實盤時段。下一個有新 bar 嘅 Nightly 會記第一筆。'}")

    st.markdown("#### 三族戶口")
    rows = []
    for fam, label in labels:
        h = headlines.get(fam) or {}
        rows.append(
            {
                "模型": label,
                "權益HKD": h.get("equity_hkd"),
                "損益HKD": h.get("pnl_hkd"),
                "回報": h.get("ret"),
                "信號": h.get("n_signals"),
                "成交": h.get("n_fills"),
                "錯過": h.get("n_miss"),
                "勝率": h.get("hit_rate"),
                "High MAE$": h.get("mae_high"),
                "Low MAE$": h.get("mae_low"),
                "Close MAE$": h.get("mae_close"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### 下一轉計劃（未成交）")
    planned = view.get("planned") or {}
    asofs = view.get("cards_asof") or {}
    tabs = st.tabs(["共用", "行業", "個股"])
    for tab, fam in zip(tabs, ("shared", "sector", "stock")):
        with tab:
            rows_p = planned.get(fam) or []
            if not rows_p:
                st.info("未有可觸價信號。")
            else:
                st.caption(f"asof {asofs.get(fam)} · {len(rows_p)} 隻")
                st.dataframe(pd.DataFrame(rows_p), hide_index=True, use_container_width=True)

    st.markdown("#### 成交流水")
    fills = ledger.get("fills") or []
    if not fills:
        st.info("未有成交。美股下一個完整時段收市後，Nightly 會自動記帳。")
    else:
        st.caption("出入場時間為 America/New_York（5 分鐘 bar）；日 K 回退則可能空白。")
        rows_f = []
        for f in fills[::-1]:
            rows_f.append(
                {
                    "session": f.get("session"),
                    "asof": f.get("asof"),
                    "family": f.get("family"),
                    "ticker": f.get("ticker"),
                    "side": f.get("side"),
                    "shares": f.get("shares"),
                    "entry": f.get("entry"),
                    "exit": f.get("exit"),
                    "入場時間": f.get("entry_ts") or "—",
                    "出場時間": f.get("exit_ts") or "—",
                    "reason": f.get("reason"),
                    "fill_source": f.get("fill_source"),
                    "pnl_hkd": f.get("pnl_hkd"),
                    "fee_usd": f.get("fee_usd"),
                }
            )
        st.dataframe(pd.DataFrame(rows_f), hide_index=True, use_container_width=True)

    days = ledger.get("days") or []
    if days:
        st.markdown("#### 每日權益")
        flat = []
        for d in days:
            row = {"session": d.get("session"), "asof": d.get("asof")}
            eq = d.get("equity_hkd") or {}
            if isinstance(eq, dict):
                row.update({f"equity_{k}": v for k, v in eq.items()})
            flat.append(row)
        st.dataframe(pd.DataFrame(flat), hide_index=True, use_container_width=True)


def _render_card(card: dict) -> None:
    action = card["action"]
    color = {"做多": "green", "做空": "red", "觀望": "gray"}.get(action, "gray")
    conf = " · 高信心" if card.get("high_confidence") else ""
    st.markdown(f"### {card['ticker']}  :{color}[{action}]{conf}")
    st.caption(
        f"#{card.get('dvol_rank', '—')} 成交額　·　{card.get('sector') or '—'}　·　"
        f"{'High/Low 優於基準' if card.get('beats_range', card.get('beats_baseline')) else 'High/Low 未優於該股基準'}"
    )
    st.caption(
        f"數據來源：{card.get('data_source') or '未知'}　·　"
        f"{card.get('universe_source') or 'S&P 500'}"
    )
    asof = card.get("asof")
    last_bar = card.get("last_bar_date")
    stale = bool(last_bar and asof and str(last_bar) < str(asof))
    if last_bar:
        if stale:
            st.caption(f"最後有 bar：:orange[{last_bar}]　·　⚠️ 滯後於 asof {asof}")
        else:
            st.caption(f"最後有 bar：{last_bar}")
    else:
        st.caption("最後有 bar：—")
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
            f"入場：{card.get('entry')} {_fmt_px(card.get('entry_px'))}　·　止盈前收 {_fmt_px(card.get('tp'))}　·　止損 {_fmt_px(card.get('sl'))}"
        )
    err = card.get("recent_error") or {}
    st.caption(
        f"近期樣本外收市誤差（{err.get('n', 0)} 日）："
        f"{_fmt_px(err.get('mae_close_px'))}　（{_fmt_pct(err.get('mae_close_ret'))}）"
    )
    st.markdown("---")


if __name__ == "__main__":
    main()
