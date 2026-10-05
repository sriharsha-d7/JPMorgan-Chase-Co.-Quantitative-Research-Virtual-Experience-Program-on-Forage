"""Quantitative research toolkit: credit-risk scoring and natural-gas storage pricing."""

from .fico_buckets import optimal_fico_buckets, assign_rating
from .credit_model import expected_loss, fit_default_model, evaluate_models, load_loans
from .gas_pricing import GasPriceModel, load_gas_prices, price_storage_contract

__all__ = [
    "optimal_fico_buckets",
    "assign_rating",
    "expected_loss",
    "fit_default_model",
    "evaluate_models",
    "load_loans",
    "GasPriceModel",
    "load_gas_prices",
    "price_storage_contract",
]
