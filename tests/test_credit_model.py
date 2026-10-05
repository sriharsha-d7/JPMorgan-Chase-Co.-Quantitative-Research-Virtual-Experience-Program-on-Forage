import pytest

from quant_research.credit_model import evaluate_models, expected_loss, fit_default_model, load_loans


@pytest.fixture(scope="module")
def loans():
    return load_loans()


def test_data_integrity(loans):
    assert len(loans) == 10_000 and loans.customer_id.is_unique and loans.isna().sum().sum() == 0
    assert set(loans.default.unique()) == {0, 1}


def test_expected_loss_formula(loans):
    model = fit_default_model(loans)
    b = {"credit_lines_outstanding": 0, "loan_amt_outstanding": 5000.0, "total_debt_outstanding": 3000.0,
         "income": 80000.0, "years_employed": 6, "fico_score": 720}
    pd_hat, el = expected_loss(model, b, recovery_rate=0.25)
    assert 0 <= pd_hat <= 1
    assert el == pytest.approx(pd_hat * 0.75 * 5000.0)
    assert pd_hat < 0.05  # clearly low-risk profile


def test_out_of_range_input_warns(loans):
    model = fit_default_model(loans)
    b = {"credit_lines_outstanding": 2, "loan_amt_outstanding": 10000, "total_debt_outstanding": 30000,
         "income": 50000, "years_employed": 10, "fico_score": 700}  # debt/income 0.6 > any training row
    with pytest.warns(UserWarning, match="debt_to_income"):
        expected_loss(model, b)


def test_models_beat_weakest_baseline(loans):
    res = evaluate_models(loans).set_index("model")
    assert res.loc["logistic", "test_auc"] > res.loc["single feature: fico_score", "test_auc"]
    assert res.loc["gradient_boosting", "test_auc"] > 0.95
