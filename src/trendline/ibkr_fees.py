"""IBKR Pro Fixed US stock costs (paper). Not a live broker.

Source: https://www.interactivebrokers.com/en/pricing/commissions-stocks.php
as of 2026-09-13. Lite is US-residents only; HK accounts use Pro.
Fixed still passes SEC / FINRA TAF / CAT. Exchange and NSCC are Tiered-only.
"""

from __future__ import annotations

# IBKR Pro Fixed
IBKR_PER_SHARE = 0.005
IBKR_MIN_ORDER = 1.00
IBKR_MAX_TRADE_FRAC = 0.01

# Regulatory (sell-side except CAT)
SEC_SALE_RATE = 0.0000206  # $20.60 / $1m, effective 2026-04-04
FINRA_TAF_PER_SHARE = 0.000195
FINRA_TAF_MAX = 9.79
FINRA_CAT_PER_SHARE = 0.000003

# Pass-through on IB commission (Fixed)
NYSE_PT_ON_COMM = 0.000175
FINRA_PT_ON_COMM = 0.00056


def _cents(x: float) -> float:
    return round(float(x) + 1e-12, 2)


def commission(shares: int, price: float) -> float:
    shares = int(shares)
    price = float(price)
    if shares <= 0 or price <= 0:
        return 0.0
    raw = shares * IBKR_PER_SHARE
    cap = shares * price * IBKR_MAX_TRADE_FRAC
    # IB: if the 1% cap is below the $1 min, the cap wins.
    return min(max(raw, IBKR_MIN_ORDER), cap)


def order_fees(shares: int, price: float, *, is_sell: bool) -> dict:
    """One fill: commission + regulatory + tiny pass-throughs."""
    comm = commission(shares, price)
    notional = int(shares) * float(price)
    sec = (SEC_SALE_RATE * notional) if is_sell else 0.0
    taf = min(FINRA_TAF_PER_SHARE * shares, FINRA_TAF_MAX) if is_sell else 0.0
    cat = FINRA_CAT_PER_SHARE * shares
    nyse_pt = comm * NYSE_PT_ON_COMM
    finra_pt = comm * FINRA_PT_ON_COMM
    total = comm + sec + taf + cat + nyse_pt + finra_pt
    return {
        "commission": _cents(comm),
        "sec": _cents(sec),
        "taf": _cents(taf),
        "cat": _cents(cat),
        "pass_through": _cents(nyse_pt + finra_pt),
        "total": _cents(total),
        "is_sell": bool(is_sell),
    }


def roundtrip_fees(side: int, shares: int, entry: float, exit: float) -> dict:
    """Long: buy then sell. Short: sell then buy. Fees in USD."""
    if side == 1:
        open_leg = order_fees(shares, entry, is_sell=False)
        close_leg = order_fees(shares, exit, is_sell=True)
    elif side == -1:
        open_leg = order_fees(shares, entry, is_sell=True)
        close_leg = order_fees(shares, exit, is_sell=False)
    else:
        return {
            "open": {},
            "close": {},
            "commission": 0.0,
            "regulatory": 0.0,
            "total": 0.0,
        }
    comm = open_leg["commission"] + close_leg["commission"]
    reg = (
        open_leg["sec"]
        + open_leg["taf"]
        + open_leg["cat"]
        + open_leg["pass_through"]
        + close_leg["sec"]
        + close_leg["taf"]
        + close_leg["cat"]
        + close_leg["pass_through"]
    )
    return {
        "open": open_leg,
        "close": close_leg,
        "commission": _cents(comm),
        "regulatory": _cents(reg),
        "total": _cents(open_leg["total"] + close_leg["total"]),
    }
