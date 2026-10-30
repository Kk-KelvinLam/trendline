from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nightly import _is_first_sunday  # noqa: E402
from update_universe import _yahoo_ticker  # noqa: E402


def test_yahoo_share_class():
    assert _yahoo_ticker("BRK.B") == "BRK-B"
    assert _yahoo_ticker("AAPL") == "AAPL"


def test_first_sunday():
    assert _is_first_sunday(date(2026, 9, 6)) is True
    assert _is_first_sunday(date(2026, 9, 13)) is False
    assert _is_first_sunday(date(2026, 9, 1)) is False  # Tuesday
