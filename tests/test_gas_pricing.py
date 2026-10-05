import numpy as np
import pandas as pd
import pytest

from quant_research.gas_pricing import GasPriceModel, backtest, load_gas_prices, price_storage_contract


@pytest.fixture(scope="module")
def prices():
    return load_gas_prices()


def test_data_shape(prices):
    assert len(prices) == 48 and prices.index.is_monotonic_increasing and prices.notna().all()


def test_recovers_planted_trend_and_season():
    idx = pd.date_range("2020-01-31", periods=48, freq="ME")
    t = (idx - idx[0]).days / 365.25
    s = pd.Series(10 + 0.5 * t + 1.0 * np.sin(2 * np.pi * t), index=idx)
    m = GasPriceModel(1).fit(s)
    assert m.coef_[1] == pytest.approx(0.5, abs=1e-6)
    assert m.coef_[2] == pytest.approx(1.0, abs=1e-6)


def test_interpolates_observed_and_rejects_far_future(prices):
    m = GasPriceModel().fit(prices)
    assert m.price(prices.index[3]) == pytest.approx(prices.iloc[3])
    with pytest.raises(ValueError):
        m.price("2027-01-01")
    with pytest.raises(ValueError):
        m.price("2019-01-01")


def test_seasonal_model_beats_baselines_on_holdout(prices):
    bt = backtest(prices).set_index("model")
    assert bt.loc["trend + annual seasonality", "RMSE"] < bt.loc["linear trend only", "RMSE"]


FLAT = lambda d: 10.0  # noqa: E731


def test_contract_arithmetic():
    cv = price_storage_contract(
        lambda d: 10.0 if pd.Timestamp(d) < pd.Timestamp("2024-02-01") else 12.0,
        injections={"2024-01-01": 100}, withdrawals={"2024-03-01": 100},
        max_volume=100, injection_rate=100, withdrawal_rate=100,
        storage_cost_per_unit_day=0.01, injection_fee_per_unit=0.1, withdrawal_fee_per_unit=0.2)
    days = (pd.Timestamp("2024-03-01") - pd.Timestamp("2024-01-01")).days  # 60
    assert cv.revenue == 1200 and cv.purchase_cost == 1000
    assert cv.storage_cost == pytest.approx(100 * days * 0.01)
    assert cv.fees == pytest.approx(10 + 20)
    assert cv.value == pytest.approx(1200 - 1000 - 60 - 30)


@pytest.mark.parametrize("inj,wd,kw", [
    ({"2024-01-01": 200}, {}, {}),                                  # over capacity
    ({"2024-01-01": 50}, {"2024-02-01": 80}, {}),                   # withdraw more than stored
    ({"2024-01-01": 60}, {}, {"injection_rate": 50}),               # rate cap
    ({"2024-01-01": 50}, {"2024-02-01": 50}, {"withdrawal_rate": 10}),
])
def test_infeasible_contracts_rejected(inj, wd, kw):
    args = dict(max_volume=100, injection_rate=100, withdrawal_rate=100) | kw
    with pytest.raises(ValueError):
        price_storage_contract(FLAT, inj, wd, **args)


def test_unwithdrawn_gas_warns():
    with pytest.warns(UserWarning):
        price_storage_contract(FLAT, {"2024-01-01": 10}, {}, max_volume=100, injection_rate=100, withdrawal_rate=100)
