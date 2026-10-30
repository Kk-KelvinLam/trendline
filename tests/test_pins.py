"""Pin toggle must change session state in one callback (no full reload)."""

from __future__ import annotations

import streamlit as st


def test_toggle_pin_adds_and_removes(monkeypatch):
    import streamlit_app as app

    state: dict = {}

    class _State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    fake = _State()
    monkeypatch.setattr(app.st, "session_state", fake)
    monkeypatch.setattr(st, "session_state", fake)

    app._toggle_pin("MU")
    assert fake["pinned_tickers"] == ["MU"]
    app._toggle_pin("NVDA")
    assert fake["pinned_tickers"] == ["MU", "NVDA"]
    app._toggle_pin("MU")
    assert fake["pinned_tickers"] == ["NVDA"]
