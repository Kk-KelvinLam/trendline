"""Post-US-close next-day dashboard (zh-HK) — three model families + scoreboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import streamlit as st

try:
    from st_keyup import st_keyup
except ImportError:  # pragma: no cover
    st_keyup = None


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
from components.tl_widgets import pins_bridge


st.set_page_config(page_title="Trendline · 美股翌日預測", page_icon="📈", layout="wide")

DISCLAIMER = "本頁為量化模型輸出，並非投資建議。過往回測不代表未來表現。Model output, not investment advice."

FAMILY_PAGES = {
    "共用模型": ("shared", CARDS_SHARED_PATH),
    "行業模型": ("sector", CARDS_SECTOR_PATH),
    "個股模型": ("stock", CARDS_STOCK_PATH),
    "釘選對照": ("pinned", None),
    "計分板": ("scoreboard", None),
    "流水": ("ledger", None),
}

FAMILY_LABELS = (("shared", "共用"), ("sector", "行業"), ("stock", "個股"))


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






def _ensure_pinned_state() -> list[str]:
    if "pinned_tickers" not in st.session_state:
        st.session_state["pinned_tickers"] = []
    return st.session_state["pinned_tickers"]


def _normalize_pins(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pins")
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        t = str(item or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _stash_pin_query() -> None:
    """Remember ?pin=TICKER across the localStorage hydrate rerun."""
    try:
        raw = st.query_params.get("pin")
    except Exception:
        return
    if raw is None or raw == "":
        return
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    ticker = str(raw).strip().upper()
    if ticker:
        st.session_state["_pending_pin"] = ticker


def _sync_pins_storage() -> None:
    """Read parent.localStorage. nonce forces a fresh read after a pin click."""
    hydrated = bool(st.session_state.get("_pins_hydrated"))
    current = list(_ensure_pinned_state())
    incoming = pins_bridge(pins=current, write=hydrated, key="tl_pins_bridge")
    if hydrated:
        return
    if incoming is None:
        return
    loaded = _normalize_pins(incoming)
    pending = st.session_state.pop("_pending_pin", None)
    if pending:
        if pending in loaded:
            loaded = [x for x in loaded if x != pending]
        else:
            loaded.append(pending)
    st.session_state["_pins_hydrated"] = True
    st.session_state["pinned_tickers"] = loaded
    if loaded != current:
        st.rerun()


def _pin_ticker(ticker: str) -> None:
    t = (ticker or "").strip().upper()
    if not t:
        return
    pinned = _ensure_pinned_state()
    if t not in pinned:
        pinned.append(t)


def _unpin_ticker(ticker: str) -> None:
    t = (ticker or "").strip().upper()
    pinned = _ensure_pinned_state()
    st.session_state["pinned_tickers"] = [x for x in pinned if x != t]


def _toggle_pin(ticker: str) -> None:
    t = (ticker or "").strip().upper()
    if not t:
        return
    pinned = _ensure_pinned_state()
    if t in pinned:
        st.session_state["pinned_tickers"] = [x for x in pinned if x != t]
    else:
        pinned.append(t)






def _clear_text_key(key: str) -> None:
    """on_click callback: safe to clear a keyed text widget before next render."""
    st.session_state[key] = ""


def _live_search(label: str, *, placeholder: str, key: str, debounce: int = 150) -> str:
    """Filter as the user types. Falls back to Enter/blur if streamlit-keyup is missing."""
    if st_keyup is not None:
        raw = st_keyup(label, placeholder=placeholder, key=key, debounce=debounce)
        return "" if raw is None else str(raw)
    raw = st.text_input(label, placeholder=placeholder, key=key)
    return "" if raw is None else str(raw)


def _filter_cards_by_query(cards: list[dict], query: str) -> list[dict]:
    q = (query or "").strip().upper()
    if not q:
        return cards
    return [c for c in cards if str(c.get("ticker", "")).upper().startswith(q)]


def _card_index_by_ticker(payload: dict | None) -> dict[str, dict]:
    if not payload:
        return {}
    out: dict[str, dict] = {}
    for c in payload.get("cards") or []:
        t = str(c.get("ticker", "")).upper()
        if t:
            out[t] = c
    return out


def _recent_error_summary(card: dict | None) -> str:
    if not card:
        return "—"
    err = card.get("recent_error") or {}
    if not err.get("n"):
        return "暫無"
    scope = err.get("scope") or "recent"
    scope_zh = {"walk_forward": "WF", "recent": "近期"}.get(scope, scope)
    return f"{scope_zh} {_fmt_px(err.get('mae_close_px'))}（{_fmt_pct(err.get('mae_close_ret'))}，n={err.get('n', 0)}）"


def _find_ticker_in_payloads(ticker: str, payloads: dict[str, dict | None]) -> bool:
    t = ticker.strip().upper()
    for payload in payloads.values():
        if payload and any(str(c.get("ticker", "")).upper() == t for c in payload.get("cards") or []):
            return True
    return False


def _inject_back_to_top(*, jump: bool) -> None:
    """Fixed ↑ control. Avoid remounting the JS iframe on every widget rerun (scroll jank)."""
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
/* Keep ticker + action + pin on one tight row (mobile + desktop) */
.tl-card-head {
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: 0.45rem;
  margin: 0 0 0.15rem 0;
  line-height: 1.2;
}
.tl-card-head .tl-t {
  font-size: 1.35rem;
  font-weight: 700;
  white-space: nowrap;
  color: inherit;
}
.tl-card-head .tl-a {
  font-size: 1.35rem;
  font-weight: 700;
  white-space: nowrap;
}
.tl-card-head .tl-a.green { color: #2ecc71; }
.tl-card-head .tl-a.red { color: #e74c3c; }
.tl-card-head .tl-a.gray { color: #9ca3af; }
.tl-card-head .tl-conf {
  font-size: 0.85rem;
  opacity: 0.75;
  white-space: nowrap;
}
/* Pin lives in the same HTML row as ticker + direction. */
.tl-card-head a.tl-pin {
  text-decoration: none !important;
  font-size: 1.2rem;
  line-height: 1;
  margin-left: 0.15rem;
  cursor: pointer;
}
#tl-pin-sync-mark { display: none; }
[data-testid="stElementContainer"]:has(#tl-pin-sync-mark) + [data-testid="stElementContainer"],
[data-testid="element-container"]:has(#tl-pin-sync-mark) + [data-testid="element-container"] {
  position: absolute !important;
  left: -9999px !important;
  height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
}
.element-container:has(.tl-card-head) + .element-container button {
  width: 2.1rem !important;
  height: 2.1rem !important;
  min-height: 2.1rem !important;
  padding: 0 !important;
  border-radius: 999px !important;
}
/* Exactly-2-column rows only (nth-child(2):last-child). Outer card grid has 3 cols — excluded. */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child):has(.tl-card-head) {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 0.35rem !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child):has(.tl-card-head)
  > div[data-testid="column"]:last-child {
  flex: 0 0 2.5rem !important;
  width: 2.5rem !important;
  min-width: 2.5rem !important;
  max-width: 2.5rem !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child):has(.tl-card-head) button {
  border-radius: 999px !important;
  width: 2.25rem !important;
  height: 2.25rem !important;
  min-height: 2.25rem !important;
  padding: 0 !important;
  line-height: 1 !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child):has(.tl-search-mark) {
  flex-wrap: nowrap !important;
  align-items: flex-end !important;
  gap: 0.5rem !important;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child):has(.tl-search-mark)
  > div[data-testid="column"]:last-child {
  flex: 0 0 4.5rem !important;
  width: 4.5rem !important;
  min-width: 4.5rem !important;
  max-width: 4.5rem !important;
}

</style>
<div id="tl-top"></div>
<a id="tl-arrow" href="#tl-top" aria-label="回到頁頂">↑</a>
""",
        unsafe_allow_html=True,
    )

    # Only mount the scroll helper iframe when needed (page change / first load).
    need_js = jump or not st.session_state.get("_tl_arrow_js_mounted_v3")
    if not need_js:
        return
    st.session_state["_tl_arrow_js_mounted_v3"] = True
    flag = "1" if jump else "0"
    components.html(
        f"""
<script>
(function() {{
  const win = window.parent;
  const doc = win.document;
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
    if (top && top.scrollIntoView) top.scrollIntoView({{block: "start"}});
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
  if (!win.__tlPinBound) {{
    win.__tlPinBound = true;
    doc.addEventListener("click", function (e) {{
      var a = e.target.closest("a.tl-pin");
      if (!a) return;
      e.preventDefault();
      e.stopPropagation();
      var t = (a.getAttribute("data-ticker") || "").toUpperCase();
      if (!t) return;
      var key = "trendline_pinned";
      var pins = [];
      try {{ pins = JSON.parse(win.localStorage.getItem(key) || "[]"); }} catch (err) {{ pins = []; }}
      if (!Array.isArray(pins)) pins = [];
      var i = pins.indexOf(t);
      if (i >= 0) pins.splice(i, 1); else pins.push(t);
      try {{ win.localStorage.setItem(key, JSON.stringify(pins)); }} catch (err) {{}}
      var btn = Array.from(doc.querySelectorAll("button")).find(function (b) {{
        return (b.textContent || "").trim() === "pin-sync";
      }});
      if (btn) btn.click();
    }}, true);
  }}
  if (!win.__tlArrowTimer) {{
    win.__tlArrowTimer = win.setInterval(sync, 400);
  }}
  if ("{flag}" === "1") {{
    hardTop();
    win.setTimeout(hardTop, 80);
  }}
  sync();
}})();
</script>
""",
        height=0,
    )



def main() -> None:
    _sync_pins_storage()
    _ensure_pinned_state()
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
    if key == "pinned":
        _render_pinned()
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
    if key == "shared":
        st.caption(
            "現有規則（共用）：淡區間入場，無 Close q50 方向閘。"
            "近期 Close MAE（20 日、至少 10 個樣本）高於 2.5% 則整張觀望；"
            "高於 1.8% 則唔標高信心。三套系統（模型＋規則＋執行）賽馬；以流水權益為準，勝率只供參考。"
        )
    elif key == "sector":
        st.caption(
            "現有規則（行業）：淡區間入場，另要 Close q50 ≥ +10bps 先做多、≤ −10bps 先做空。"
            "近期 Close MAE（20 日、至少 10 個樣本）高於 2.5% 則整張觀望；"
            "高於 1.8% 則唔標高信心。三套系統（模型＋規則＋執行）賽馬；以流水權益為準，勝率只供參考。"
        )
    elif key == "stock":
        st.caption(
            "現有規則（個股）：淡區間入場，另要 Close q50 ≥ +10bps 先做多、≤ −10bps 先做空。"
            "薄歷史票會 fallback 共用模型。近期 Close MAE 高於 2.5% 則整張觀望；"
            "高於 1.8% 則唔標高信心。三套系統（模型＋規則＋執行）賽馬；以流水權益為準，勝率只供參考。"
        )

    search_q = _live_search(
        "搜尋",
        placeholder="搜尋股票代號（例如 NVDA）",
        key=f"search_{key}",
        debounce=150,
    )
    filt = st.radio("篩選", ["全部", "做多", "做空", "高信心"], horizontal=True, key=f"filt_{key}")

    cards = cards_payload.get("cards", [])
    cards = _filter_cards_by_query(cards, search_q)
    if filt == "做多":
        cards = [c for c in cards if c["action"] == "做多"]
    elif filt == "做空":
        cards = [c for c in cards if c["action"] == "做空"]
    elif filt == "高信心":
        cards = [c for c in cards if c.get("high_confidence")]

    st.subheader("翌日交易卡")
    if not cards:
        st.info("此篩選沒有卡片。" if not (search_q or "").strip() else "搜尋／篩選沒有符合嘅卡片。")
    else:
        cols = st.columns(3)
        for i, card in enumerate(cards):
            with cols[i % 3]:
                _render_card(card, family=key)

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


def _fmt_hkd_delta(x) -> str | None:
    """Streamlit colors metric deltas by whether the string starts with '-'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    val = float(x)
    if val == 0:
        return None
    sign = "-" if val < 0 else ""
    return f"{sign}HK${abs(val):,.0f}"


def _render_ledger() -> None:
    st.subheader("流水 · 三戶口賽馬")
    st.warning("紙上模擬，未接券商。三個模型同 VOO 基準各 HK$500,000。入場當日一定平倉，未中止盈／止損就用當日收市價出場，唔留過夜。")
    st.caption(
        "對賬路徑：只認 America/New_York 常規時段 5 分鐘 bar；"
        "觸價要喺 12:30 ET 之前先入場，之後先到價當錯過。"
        "缺 5m 亦當錯過，不再回退日 K。Walk-forward OOS 仍用日 K。"
    )
    st.caption(
        "勝率／成交數按「有方向嘅訊號卡」計，未扣佣、亦包括未入書（倉位太細／觸及名義下限）嘅觸價。"
        "三套系統（模型＋規則＋執行）賽馬：共用無 Close 閘，行業／個股要 Close q50 ±10bps。"
        "以三本權益對照 VOO：2026-09-14 開市一把過買住，之後唔買賣、只按收市計市值（無佣）。勝率只供參考。"
    )
    view = ledger_view()
    ledger = view["ledger"]
    acct = ledger.get("account") or {}
    headlines = view.get("headlines") or {}
    labels = (("shared", "共用"), ("sector", "行業"), ("stock", "個股"))
    cols = st.columns(4)
    for col, (fam, label) in zip(cols, labels):
        h = headlines.get(fam) or {}
        with col:
            st.metric(f"{label}權益", _fmt_hkd(h.get("equity_hkd")), delta=_fmt_hkd_delta(h.get("pnl_hkd")))
    with cols[3]:
        vh = headlines.get("voo") or {}
        st.metric("VOO 基準", _fmt_hkd(vh.get("equity_hkd")), delta=_fmt_hkd_delta(vh.get("pnl_hkd")))
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
    vh = headlines.get("voo") or {}
    rows.append(
        {
            "模型": "VOO 基準",
            "權益HKD": vh.get("equity_hkd"),
            "損益HKD": vh.get("pnl_hkd"),
            "回報": vh.get("ret"),
            "信號": "—",
            "成交": vh.get("shares"),
            "錯過": "—",
            "勝率": "—",
            "High MAE$": "—",
            "Low MAE$": "—",
            "Close MAE$": "—",
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
        st.caption("按模型分頁。出入場時間為 America/New_York（5 分鐘 bar）。12:30 ET 或之後先到價、或缺 5m，唔入呢度。")

        def _fill_rows(items: list[dict]) -> list[dict]:
            rows = []
            for f in items:
                rows.append(
                    {
                        "session": f.get("session"),
                        "asof": f.get("asof"),
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
            return rows

        fill_tabs = st.tabs(["共用", "行業", "個股"])
        for tab, fam in zip(fill_tabs, ("shared", "sector", "stock")):
            with tab:
                fam_fills = [f for f in fills if f.get("family") == fam]
                fam_fills = list(reversed(fam_fills))
                if not fam_fills:
                    st.info("呢個模型未有成交。")
                else:
                    st.caption(f"{len(fam_fills)} 筆")
                    st.dataframe(pd.DataFrame(_fill_rows(fam_fills)), hide_index=True, use_container_width=True)

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


def _same_row(*builders) -> None:
    """Ticker + pin on one flex row. Falls back if Streamlit is too old for horizontal=."""
    row = None
    try:
        row = st.container(horizontal=True, vertical_alignment="center", gap="small", wrap=False)
    except TypeError:
        try:
            row = st.container(horizontal=True, vertical_alignment="center", gap="small")
        except TypeError:
            row = None
    if row is None:
        for build in builders:
            build()
        return
    with row:
        for build in builders:
            build()


def _render_card(card: dict, *, family: str = "shared") -> None:
    action = card["action"]
    color = {"做多": "green", "做空": "red", "觀望": "gray"}.get(action, "gray")
    conf = " · 高信心" if card.get("high_confidence") else ""
    ticker = str(card.get("ticker", "")).upper()
    pinned = _ensure_pinned_state()
    is_pinned = ticker in pinned
    conf_txt = conf.strip(" ·") if conf else ""
    pin_icon = "📍" if is_pinned else "📌"
    tip = "取消釘選" if is_pinned else "釘選對照"

    def _title() -> None:
        st.markdown(
            f'<div class="tl-card-head">'
            f'<span class="tl-t">{ticker}</span>'
            f'<span class="tl-a {color}">{action}</span>'
            f'<span class="tl-conf">{conf_txt}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    def _pin() -> None:
        st.button(
            pin_icon,
            key=f"pin_{family}_{ticker}",
            help=tip,
            on_click=_toggle_pin,
            args=(ticker,),
        )

    _same_row(_title, _pin)
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
    scope = err.get("scope") or ("recent" if err.get("n") else "none")
    if scope == "walk_forward":
        label = f"樣本外收市誤差（walk-forward 全段 {err.get('n', 0)} 日）"
    elif scope == "recent":
        label = f"近期樣本外收市誤差（{err.get('n', 0)} 日）"
    else:
        label = "樣本外收市誤差（暫無 OOS 檔）"
    st.caption(
        f"{label}："
        f"{_fmt_px(err.get('mae_close_px'))}　（{_fmt_pct(err.get('mae_close_ret'))}）"
    )
    st.markdown("---")


def _comparison_row(card: dict | None) -> dict:
    if card is None:
        return {
            "行動": "—",
            "方向": "—",
            "前收": "—",
            "入場價": "—",
            "止盈": "—",
            "止損": "—",
            "預測高 q50": "—",
            "預測低 q50": "—",
            "預測收 q50": "—",
            "最後 bar": "—",
            "近期誤差": "—",
        }
    p = card.get("pred") or {}
    return {
        "行動": card.get("action") or "—",
        "方向": card.get("side") or "—",
        "前收": _fmt_px(card.get("prior_close")),
        "入場價": _fmt_px(card.get("entry_px")),
        "止盈": _fmt_px(card.get("tp")),
        "止損": _fmt_px(card.get("sl")),
        "預測高 q50": _fmt_px((p.get("high") or {}).get("q50")),
        "預測低 q50": _fmt_px((p.get("low") or {}).get("q50")),
        "預測收 q50": _fmt_px((p.get("close") or {}).get("q50")),
        "最後 bar": card.get("last_bar_date") or "—",
        "近期誤差": _recent_error_summary(card),
    }


def _render_pinned() -> None:
    st.subheader("釘選對照 · 三族模型並排")
    st.caption("喺共用／行業／個股頁面 Pin 股票，或喺下面直接輸入代號加入。對照行動、入場、止盈止損同預測中位。")

    payloads = {
        "shared": _load_json(CARDS_SHARED_PATH) or _load_json(CARDS_PATH),
        "sector": _load_json(CARDS_SECTOR_PATH),
        "stock": _load_json(CARDS_STOCK_PATH),
    }
    indexes = {fam: _card_index_by_ticker(payloads.get(fam)) for fam, _ in FAMILY_LABELS}

    if st.session_state.pop("_clear_pinned_add", False):
        st.session_state["pinned_add_input"] = ""
    add_q = st.text_input(
        "手動釘選",
        placeholder="輸入股票代號（例如 AAPL）",
        key="pinned_add_input",
    )
    add_clicked = st.button("加入釘選", key="pinned_add_btn", use_container_width=True)
    if add_clicked:
        t = (add_q or "").strip().upper()
        if not t:
            st.warning("請輸入股票代號。")
        elif not _find_ticker_in_payloads(t, payloads):
            st.warning(f"三族卡片都搵唔到 {t}。")
        else:
            _pin_ticker(t)
            st.session_state["_clear_pinned_add"] = True
            st.rerun()

    pinned = list(_ensure_pinned_state())
    if not pinned:
        st.info("尚未釘選任何股票。請到共用／行業／個股頁面按 📌 Pin，或喺上面輸入代號加入。")
        return

    st.caption(f"已釘選 {len(pinned)} 隻")
    for ticker in pinned:
        def _title() -> None:
            st.markdown(
                f'<div class="tl-card-head"><span class="tl-t">{ticker}</span></div>',
                unsafe_allow_html=True,
            )

        def _pin() -> None:
            st.button(
                "📍",
                key=f"unpin_{ticker}",
                help="取消釘選",
                on_click=_toggle_pin,
                args=(ticker,),
            )

        _same_row(_title, _pin)

        cards_by_fam = {fam: indexes[fam].get(ticker) for fam, _ in FAMILY_LABELS}
        actions = {
            fam: (cards_by_fam[fam] or {}).get("action")
            for fam, _ in FAMILY_LABELS
            if cards_by_fam[fam] is not None
        }
        present_actions = {a for a in actions.values() if a}
        if len(present_actions) > 1:
            bits = "／".join(
                f"{label}：{(cards_by_fam[fam] or {}).get('action') or '—'}"
                for fam, label in FAMILY_LABELS
            )
            st.warning(f"⚠️ 三族行動不一致：{bits}")
        elif len(present_actions) == 1 and sum(1 for c in cards_by_fam.values() if c) < 3:
            st.caption("部分模型未有此股票卡片。")

        cols = st.columns(3)
        for col, (fam, label) in zip(cols, FAMILY_LABELS):
            with col:
                st.markdown(f"**{label}**")
                card = cards_by_fam[fam]
                if card is None:
                    st.info("無此卡片")
                    continue
                row = _comparison_row(card)
                st.dataframe(
                    pd.DataFrame([row]).T.rename(columns={0: "值"}),
                    use_container_width=True,
                    height=420,
                )
        st.markdown("---")


if __name__ == "__main__":
    main()
