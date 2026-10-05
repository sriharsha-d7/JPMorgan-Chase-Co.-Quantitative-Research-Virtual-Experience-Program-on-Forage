"""Regenerate and execute the two notebooks: `python scripts/build_notebooks.py`."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = "import sys, warnings\nsys.path.insert(0, '../src')\nimport numpy as np, pandas as pd\nfrom quant_research import *\nfrom quant_research import viz\npd.set_option('display.width', 120)"

credit = [
    md("# Credit risk: probability of default, expected loss and FICO ratings\n\n"
       "Three questions on a book of 10,000 loans (18.5% defaulted): **(1)** how predictable is default, "
       "**(2)** what is the expected loss on a new borrower, **(3)** how should FICO scores be mapped to a small "
       "number of ratings?"),
    code(SETUP + "\nloans = load_loans()\nloans.describe().T[['min','50%','max']]"),
    md("## 1. How predictable is default?\n\nBefore fitting anything, check how much a *single* feature explains."),
    code("viz.default_rate_by_credit_lines(loans, '../docs/default_by_credit_lines.png')\n"
         "from IPython.display import Image; Image('../docs/default_by_credit_lines.png')"),
    md("Nobody with 0-1 open credit lines defaulted and almost everyone with 5 did. So this data set is "
       "**near-deterministic**: ROC-AUC near 1.0 comes from the data, not from a clever model. "
       "The table puts that next to the models, with a single-feature baseline."),
    code("results = evaluate_models(loans)\nresults"),
    md("Reading it honestly: the single feature `credit_lines_outstanding` already reaches AUC 0.995; logistic "
       "regression adds the last half-point. FICO alone is weak here (0.71). A real lending book would not look "
       "like this, so treat the near-perfect scores as a property of the exercise data."),
    md("## 2. Expected loss\n\n`EL = PD x (1 - recovery) x loan balance`, with a 10% recovery rate. "
       "The function **warns when an input is outside the training range**, because the earlier version of this "
       "analysis reported a 98.8% PD for a borrower whose debt-to-income (0.6) was above anything in the data (max 0.41)."),
    code("model = fit_default_model(loans)\n"
         "typical = dict(credit_lines_outstanding=2, loan_amt_outstanding=7000, total_debt_outstanding=15000,\n"
         "               income=50000, years_employed=4, fico_score=610)\n"
         "p, el = expected_loss(model, typical)\nprint(f'PD {p:.1%}   expected loss ${el:,.0f}')\n"
         "outlier = dict(typical, total_debt_outstanding=30000, loan_amt_outstanding=10000, fico_score=700, years_employed=10)\n"
         "with warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter('always')\n"
         "    p2, el2 = expected_loss(model, outlier)\nprint(f'PD {p2:.1%}  <- {w[0].message}')"),
    md("## 3. FICO to rating by dynamic programming\n\n"
       "Choose bucket boundaries that maximise the log-likelihood of the observed defaults, "
       "`sum_b k_b ln(p_b) + (n_b - k_b) ln(1 - p_b)`. An exact DP over the ~440 distinct scores finds the global "
       "optimum (tested against brute force). Equal-frequency `qcut` bins, the usual shortcut, ignore defaults."),
    code("from quant_research.fico_buckets import _bucket_loglik\n"
         "rows=[]\nfor k in (3,5,8):\n    e,ll=optimal_fico_buckets(loans.fico_score, loans.default, k)\n"
         "    q=pd.qcut(loans.fico_score,k,labels=False)\n"
         "    qll=sum(_bucket_loglik(np.array([(q==i).sum()]),np.array([loans.default[q==i].sum()]))[0] for i in range(k))\n"
         "    rows.append(dict(buckets=k, dp_loglik=round(ll,1), qcut_loglik=round(qll,1), gain=round(ll-qll,1), dp_edges=e))\n"
         "pd.DataFrame(rows)"),
    code("dp_edges,_ = optimal_fico_buckets(loans.fico_score, loans.default, 5)\n"
         "q_edges = pd.qcut(loans.fico_score, 5, retbins=True)[1][1:-1].tolist()\n"
         "viz.bucket_pd_curves(loans, dp_edges, q_edges, '../docs/fico_buckets.png')\nImage('../docs/fico_buckets.png')"),
    code("loans['rating'] = assign_rating(loans.fico_score, dp_edges)   # 1 = best credit\n"
         "loans.groupby('rating').agg(borrowers=('default','size'), default_rate=('default','mean'),\n"
         "                            fico_min=('fico_score','min'), fico_max=('fico_score','max')).round(3)"),
    md("Ratings are monotone (rating 1 = best): default rate rises as the rating worsens, and the DP boundaries "
       "sit where default behaviour actually changes."),
]

gas = [
    md("# Natural-gas prices and storage-contract valuation\n\nFour years of month-end prices (Oct 2020 to Sep 2024). "
       "Goal: estimate the price on any date, extrapolate a year ahead, and value a storage contract."),
    code(SETUP + "\nprices = load_gas_prices()\nmodel = GasPriceModel(harmonics=1).fit(prices)\n"
         "viz.gas_curve(prices, model, '../docs/gas_curve.png')\nfrom IPython.display import Image; Image('../docs/gas_curve.png')"),
    md("## Does the seasonal model earn its complexity?\n\nHold out the last 12 months, fit on the first 36, compare with "
       "simple baselines. With only 48 observations this is indicative, not conclusive."),
    code("from quant_research.gas_pricing import backtest\nbt = backtest(prices)\n"
         "viz.backtest_bars(bt, '../docs/gas_backtest.png')\nbt"),
    code("Image('../docs/gas_backtest.png')"),
    md("The earlier version used a straight-line fit; it misses the winter/summer cycle, and trend alone is no "
       "better than carrying the last price forward. Adding one annual harmonic cuts the error by roughly two thirds."),
    md("## Pricing a storage contract\n\nBuy in summer, sell in winter. `price_storage_contract` values the plan as "
       "**sales - purchases - storage rent - fees**, rejects physically impossible plans (overfilling, selling gas "
       "you do not hold, breaching rate limits), and charges rent on the inventory actually held."),
    code("cv = price_storage_contract(\n    model.price,\n    injections={'2024-06-30': 1000, '2024-07-31': 1000},\n"
         "    withdrawals={'2024-12-31': 1000, '2025-01-31': 1000},\n    max_volume=2000, injection_rate=1000, withdrawal_rate=1000,\n"
         "    storage_cost_per_unit_day=0.002, injection_fee_per_unit=0.01, withdrawal_fee_per_unit=0.01)\n"
         "print(f'value ${cv.value:,.0f} = revenue ${cv.revenue:,.0f} - purchases ${cv.purchase_cost:,.0f} '\n"
         "      f'- storage ${cv.storage_cost:,.0f} - fees ${cv.fees:,.0f}')\npd.DataFrame(cv.schedule)"),
    code("# Same idea, wrong season: buying in winter and selling in summer destroys value.\n"
         "bad = price_storage_contract(model.price, {'2024-01-31': 1000}, {'2024-07-31': 1000}, max_volume=1000,\n"
         "    injection_rate=1000, withdrawal_rate=1000, storage_cost_per_unit_day=0.002)\nprint(f'value ${bad.value:,.0f}')"),
]

for name, cells in (("01_credit_risk", credit), ("02_gas_storage", gas)):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
    print("built", name)
