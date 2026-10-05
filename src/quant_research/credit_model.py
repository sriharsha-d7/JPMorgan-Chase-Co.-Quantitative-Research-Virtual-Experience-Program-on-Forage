"""Probability of default and expected loss for a book of loans."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "loan_data.csv"
RAW_FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]
FEATURES = RAW_FEATURES + ["debt_to_income"]
TARGET = "default"
RECOVERY_RATE = 0.10  # standard assumption for this exercise


def load_loans(path: Path | str = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["debt_to_income"] = df["total_debt_outstanding"] / df["income"]
    return df


def _models(seed: int = 42) -> dict[str, Pipeline | HistGradientBoostingClassifier]:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed),
    }


def fit_default_model(df: pd.DataFrame, kind: str = "logistic", seed: int = 42):
    """Fit a PD model on the whole book and remember the training ranges for input checks."""
    model = _models(seed)[kind]
    model.fit(df[FEATURES], df[TARGET])
    model.feature_ranges_ = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}
    return model


def expected_loss(model, borrower: dict, recovery_rate: float = RECOVERY_RATE) -> tuple[float, float]:
    """Return ``(pd, expected_loss)``: EL = PD * (1 - recovery) * loan balance.

    Warns when an input lies outside the range the model was trained on, because the
    probability is then an extrapolation and should not be trusted.
    """
    row = dict(borrower)
    row["debt_to_income"] = row["total_debt_outstanding"] / row["income"]
    ranges = getattr(model, "feature_ranges_", {})
    out = [f for f, (lo, hi) in ranges.items() if f in row and not lo <= row[f] <= hi]
    if out:
        warnings.warn(f"inputs outside training range, PD is an extrapolation: {out}", stacklevel=2)
    pd_hat = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    return pd_hat, pd_hat * (1 - recovery_rate) * row["loan_amt_outstanding"]


def evaluate_models(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Held-out + 5-fold CV comparison against deliberately simple baselines."""
    train, test = train_test_split(df, test_size=0.25, random_state=seed, stratify=df[TARGET])
    rows = []

    def add(name, p_test, cv_auc=np.nan):
        rows.append(
            {
                "model": name,
                "test_auc": roc_auc_score(test[TARGET], p_test),
                "test_brier": brier_score_loss(test[TARGET], np.clip(p_test, 0, 1)),
                "cv_auc_mean": cv_auc,
            }
        )

    # Baseline 1: rank by one feature. How much of the problem is already solved?
    for feat in ("credit_lines_outstanding", "fico_score"):
        score = test[feat] if feat != "fico_score" else -test[feat]
        span = score.max() - score.min()
        add(f"single feature: {feat}", ((score - score.min()) / span).to_numpy())

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, model in _models(seed).items():
        scores = cross_validate(model, df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")["test_score"]
        model.fit(train[FEATURES], train[TARGET])
        add(name, model.predict_proba(test[FEATURES])[:, 1], scores.mean())
    return pd.DataFrame(rows).round(5)
