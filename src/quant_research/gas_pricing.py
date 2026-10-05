"""Natural-gas price curve and storage-contract valuation.

Price model: month-end prices are explained by a linear trend plus one annual harmonic
(gas is seasonal: winter demand). Inside the observed window the curve interpolates the
observed prices; outside it, it extrapolates the fitted trend+seasonality for at most one year.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "nat_gas.csv"
MAX_EXTRAPOLATION_DAYS = 366


def load_gas_prices(path: Path | str = DATA) -> pd.Series:
    df = pd.read_csv(path)
    df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    return df.set_index("Dates")["Prices"].sort_index()


def _design(dates: pd.DatetimeIndex, origin: pd.Timestamp, harmonics: int = 1) -> np.ndarray:
    t_years = (dates - origin).days.to_numpy() / 365.25
    cols = [np.ones(len(dates)), t_years]
    for h in range(1, harmonics + 1):
        phase = 2 * np.pi * h * t_years
        cols += [np.sin(phase), np.cos(phase)]
    return np.column_stack(cols)


class GasPriceModel:
    """Trend + seasonality fitted by least squares; `price(date)` is the public interface."""

    def __init__(self, harmonics: int = 1):
        self.harmonics = harmonics

    def fit(self, prices: pd.Series) -> "GasPriceModel":
        self.observed_ = prices.sort_index()
        self.origin_ = self.observed_.index[0]
        X = _design(self.observed_.index, self.origin_, self.harmonics)
        self.coef_, *_ = np.linalg.lstsq(X, self.observed_.to_numpy(), rcond=None)
        return self

    def fitted(self, dates) -> np.ndarray:
        dates = pd.DatetimeIndex(dates)
        return _design(dates, self.origin_, self.harmonics) @ self.coef_

    def price(self, date) -> float:
        d = pd.Timestamp(date)
        lo, hi = self.observed_.index[0], self.observed_.index[-1]
        if lo <= d <= hi:  # interpolate observed month-end prices
            x = (self.observed_.index - self.origin_).days.to_numpy()  # unit-safe: days since first obs
            return float(np.interp((d - self.origin_).days, x, self.observed_.to_numpy()))
        if d > hi + pd.Timedelta(days=MAX_EXTRAPOLATION_DAYS) or d < lo:
            raise ValueError(f"{d.date()} is outside the supported window ({lo.date()} to {(hi + pd.Timedelta(days=MAX_EXTRAPOLATION_DAYS)).date()})")
        return float(self.fitted([d])[0])


def backtest(prices: pd.Series, holdout_months: int = 12) -> pd.DataFrame:
    """Fit on all but the last year, score on that year, versus honest baselines."""
    train, test = prices.iloc[:-holdout_months], prices.iloc[-holdout_months:]
    rows = []

    def add(name, pred):
        err = np.asarray(pred) - test.to_numpy()
        rows.append({"model": name, "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err**2).mean()),
                     "MAPE_%": 100 * np.mean(np.abs(err) / test.to_numpy())})

    add("last value carried forward", np.repeat(train.iloc[-1], len(test)))
    add("same month last year", train.iloc[-12:].to_numpy()[: len(test)])
    add("linear trend only", GasPriceModel(0).fit(train).fitted(test.index))
    add("trend + annual seasonality", GasPriceModel(1).fit(train).fitted(test.index))
    return pd.DataFrame(rows).round(3)


@dataclass
class ContractValue:
    value: float
    revenue: float
    purchase_cost: float
    storage_cost: float
    fees: float
    peak_inventory: float
    schedule: list = field(default_factory=list)


def price_storage_contract(
    price_fn,
    injections: dict,
    withdrawals: dict,
    max_volume: float,
    injection_rate: float,
    withdrawal_rate: float,
    storage_cost_per_unit_day: float = 0.0,
    injection_fee_per_unit: float = 0.0,
    withdrawal_fee_per_unit: float = 0.0,
) -> ContractValue:
    """Value of a storage contract = sales - purchases - storage rent - fees.

    ``injections`` / ``withdrawals`` map date -> volume. Raises ``ValueError`` for plans that
    break physical limits (negative inventory, over capacity, rate caps) instead of silently
    pricing an infeasible contract. Storage rent accrues on the inventory held between events.
    """
    events = sorted([(pd.Timestamp(d), +v) for d, v in injections.items()]
                    + [(pd.Timestamp(d), -v) for d, v in withdrawals.items()], key=lambda e: (e[0], e[1]))
    if any(v == 0 for _, v in events):
        raise ValueError("zero-volume events are not allowed")
    inv, prev, peak = 0.0, None, 0.0
    revenue = purchase = storage = fees = 0.0
    schedule = []
    for date, vol in events:
        if prev is not None:
            storage += inv * (date - prev).days * storage_cost_per_unit_day
        p = price_fn(date)
        if vol > 0:
            if vol > injection_rate:
                raise ValueError(f"{date.date()}: injection {vol} exceeds rate {injection_rate}")
            inv += vol
            if inv > max_volume + 1e-9:
                raise ValueError(f"{date.date()}: inventory {inv} exceeds capacity {max_volume}")
            purchase += vol * p
            fees += vol * injection_fee_per_unit
        else:
            out = -vol
            if out > withdrawal_rate:
                raise ValueError(f"{date.date()}: withdrawal {out} exceeds rate {withdrawal_rate}")
            if out > inv + 1e-9:
                raise ValueError(f"{date.date()}: withdrawal {out} exceeds inventory {inv}")
            inv -= out
            revenue += out * p
            fees += out * withdrawal_fee_per_unit
        peak = max(peak, inv)
        schedule.append({"date": date.date().isoformat(), "volume": vol, "price": round(p, 4), "inventory": inv})
        prev = date
    if inv > 1e-9:
        warnings.warn(f"{inv} units still in storage at contract end; they are not valued", stacklevel=2)
    return ContractValue(revenue - purchase - storage - fees, revenue, purchase, storage, fees, peak, schedule)
