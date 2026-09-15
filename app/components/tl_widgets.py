"""Lightweight HTML components (single-row card head + search)."""

from __future__ import annotations

import os

import streamlit.components.v1 as components

_DIR = os.path.dirname(__file__)

_card_head = components.declare_component(
    "tl_card_head",
    path=os.path.join(_DIR, "tl_card_head"),
)
_search_bar = components.declare_component(
    "tl_search_bar",
    path=os.path.join(_DIR, "tl_search_bar"),
)


def card_head(
    *,
    ticker: str,
    action: str,
    color: str,
    conf: str = "",
    pinned: bool = False,
    key: str | None = None,
) -> str | None:
    """Render ticker + action + circular pin on one nowrap row. Returns 'toggle' on pin click."""
    return _card_head(
        ticker=ticker,
        action=action,
        color=color,
        conf=conf,
        pinned=pinned,
        key=key,
        default=None,
    )


def search_bar(
    *,
    label: str = "搜尋",
    placeholder: str = "",
    value: str = "",
    debounce: int = 150,
    key: str | None = None,
) -> str:
    """Search input + clear on one nowrap row; updates as you type."""
    out = _search_bar(
        label=label,
        placeholder=placeholder,
        value=value,
        debounce=debounce,
        key=key,
        default=value or "",
    )
    return "" if out is None else str(out)
