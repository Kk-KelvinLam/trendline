from trendline.ibkr_fees import commission, order_fees, roundtrip_fees


def test_ib_fixed_examples():
    # IB published: 100 @ $25 = $1.00; 1000 @ $25 = $5.00; 10 @ $0.20 = $0.02
    assert abs(commission(100, 25.0) - 1.00) < 1e-9
    assert abs(commission(1000, 25.0) - 5.00) < 1e-9
    assert abs(commission(10, 0.20) - 0.02) < 1e-9


def test_sec_and_taf_on_sell_only():
    buy = order_fees(200, 50.0, is_sell=False)
    sell = order_fees(200, 50.0, is_sell=True)
    assert buy["sec"] == 0.0
    assert buy["taf"] == 0.0
    assert sell["sec"] > 0
    assert sell["taf"] > 0
    assert sell["total"] > buy["total"]


def test_short_roundtrip_sell_then_buy():
    fees = roundtrip_fees(-1, 40, 102.0, 100.0)
    assert fees["total"] >= 2.0  # two $1 mins
    assert fees["open"]["is_sell"] is True
    assert fees["close"]["is_sell"] is False
    assert fees["open"]["sec"] > 0
    assert fees["close"]["sec"] == 0.0
