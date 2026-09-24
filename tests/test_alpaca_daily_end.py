"""Alpaca daily free-SIP end must be RFC3339, not bare YYYY-MM-DD."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from trendline.data.providers import (
    AlpacaProvider,
    _alpaca_daily_end_rfc3339,
    expected_equity_session,
)


def test_alpaca_daily_end_rfc3339_completed_session_close_plus_15m():
    """Mid-evening ET after close → end is cash close+15m in UTC, not date-only."""
    # 2026-09-23 is EDT (UTC-4): 16:15 ET → 20:15Z
    end_incl = pd.Timestamp("2026-09-23")
    # ~22:42 ET Sep 23 = 02:42 UTC Sep 24 (manual Nightly ~10:42 HKT Sep 24)
    now = datetime(2026, 9, 24, 2, 42, 0, tzinfo=timezone.utc)
    got = _alpaca_daily_end_rfc3339(end_incl, now_utc=now)
    assert got == "2026-09-23T20:15:00Z"
    assert "T" in got and got.endswith("Z")
    assert got.count("-") >= 2  # not bare YYYY-MM-DD only shape without time


def test_alpaca_daily_end_rfc3339_clamps_when_close_plus_15_too_recent():
    """Just after close+15m but within 16m of now → clamp to now-16m."""
    end_incl = pd.Timestamp("2026-09-23")
    # 16:20 ET = 20:20Z; desired 20:15Z is only 5m old → clamp
    now = datetime(2026, 9, 23, 20, 20, 0, tzinfo=timezone.utc)
    got = _alpaca_daily_end_rfc3339(end_incl, now_utc=now)
    assert got == "2026-09-23T20:04:00Z"


def test_alpaca_daily_end_rfc3339_skips_before_cash_close():
    end_incl = pd.Timestamp("2026-09-23")
    # 15:00 ET = 19:00Z — session still open
    now = datetime(2026, 9, 23, 19, 0, 0, tzinfo=timezone.utc)
    assert _alpaca_daily_end_rfc3339(end_incl, now_utc=now) is None


def test_download_passes_rfc3339_end_not_date_only(monkeypatch):
    """AlpacaProvider.download must send RFC3339 end (contains T and Z)."""
    captured: dict = {}

    def fake_fetch(self, symbols, *, start, end, sym_to_orig):
        captured["start"] = start
        captured["end"] = end
        captured["symbols"] = list(symbols)
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
            ]
        )

    monkeypatch.setattr(AlpacaProvider, "_fetch_chunk", fake_fetch)
    # Freeze "now" via the helper path used inside download
    fixed_now = datetime(2026, 9, 24, 2, 42, 0, tzinfo=timezone.utc)

    def fixed_end(end_incl, *, now_utc=None):
        return _alpaca_daily_end_rfc3339(end_incl, now_utc=fixed_now)

    monkeypatch.setattr(
        "trendline.data.providers._alpaca_daily_end_rfc3339",
        fixed_end,
    )

    prov = AlpacaProvider(key_id="test-key", secret="test-secret")
    # end exclusive → expected session 2026-09-23
    out = prov.download(["AAPL"], start="2026-09-20", end="2026-09-24")
    assert out.empty
    assert "end" in captured
    end = captured["end"]
    assert "T" in end
    assert end.endswith("Z") or "+00:00" in end
    assert end != "2026-09-23"
    assert not end.endswith("2026-09-23") or "T" in end
    # bare date would be exactly 10 chars YYYY-MM-DD
    assert len(end) > 10
    assert end == "2026-09-23T20:15:00Z"
    assert expected_equity_session("2026-09-24") == pd.Timestamp("2026-09-23")


def test_fetch_chunk_url_encodes_rfc3339_end(monkeypatch):
    """_fetch_chunk puts end into query string unchanged (RFC3339)."""
    seen_url: list[str] = []

    class _Resp:
        def read(self):
            return b'{"bars":{}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=90):
        seen_url.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(
        "trendline.data.providers.urllib.request.urlopen",
        fake_urlopen,
    )
    prov = AlpacaProvider(key_id="test-key", secret="test-secret")
    prov._fetch_chunk(
        ["AAPL"],
        start="2026-09-20",
        end="2026-09-23T20:15:00Z",
        sym_to_orig={"AAPL": "AAPL"},
    )
    assert seen_url
    qs = parse_qs(urlparse(seen_url[0]).query)
    assert qs["end"] == ["2026-09-23T20:15:00Z"]
    assert "T" in qs["end"][0]
