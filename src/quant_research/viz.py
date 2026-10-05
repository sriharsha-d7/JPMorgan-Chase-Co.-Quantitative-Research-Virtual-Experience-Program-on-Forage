"""Figures used by the notebooks and README (validated reference palette, light surface)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#9a9992"


def _style(ax, xlabel="", ylabel="", title=""):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=12)


def default_rate_by_credit_lines(loans: pd.DataFrame, path: str):
    g = loans.groupby("credit_lines_outstanding")["default"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(g.index, g["mean"] * 100, color=BLUE, width=0.6)
    for r in bars:
        r.set_linewidth(0)
    for x, (m, n) in g.iterrows():
        ax.text(x, m * 100 + 2, f"{m:.1%}\n(n={int(n):,})", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 118)
    ax.set_xticks(g.index)
    _style(ax, "Credit lines outstanding", "Default rate (%)",
           "One feature almost solves the problem: default rate by open credit lines")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bucket_pd_curves(loans: pd.DataFrame, dp_edges, q_edges, path: str):
    fico, d = loans["fico_score"].to_numpy(), loans["default"].to_numpy()

    def step(edges):
        b = np.searchsorted(np.asarray(edges), fico, side="left")
        pd_b = np.array([d[b == i].mean() for i in range(len(edges) + 1)])
        grid = np.arange(fico.min(), fico.max() + 1)
        return grid, pd_b[np.searchsorted(np.asarray(edges), grid, side="left")]

    bins = np.arange(400, 860, 20)
    mids, rate = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fico >= lo) & (fico < hi)
        if m.sum() >= 20:
            mids.append(lo + 10)
            rate.append(d[m].mean() * 100)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.scatter(mids, rate, s=22, color=NEUTRAL, zorder=2, label="Observed default rate (20-pt bins)")
    gx, gy = step(q_edges)
    ax.step(gx, gy * 100, where="post", color=ORANGE, lw=2, label="Equal-frequency buckets (qcut)")
    gx, gy = step(dp_edges)
    ax.step(gx, gy * 100, where="post", color=BLUE, lw=2, label="Likelihood-optimal buckets (DP)")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax, "FICO score", "Default rate (%)", "Bucketed PD against what the data shows")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gas_curve(observed: pd.Series, model, path: str):
    future = pd.date_range(observed.index[-1], periods=13, freq="ME")
    fit_idx = pd.date_range(observed.index[0], observed.index[-1], freq="D")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(observed.index, observed.values, s=18, color=NEUTRAL, zorder=3, label="Observed month-end price")
    ax.plot(fit_idx, model.fitted(fit_idx), color=BLUE, lw=2, label="Trend + seasonality (fit)")
    ax.plot(future, model.fitted(future), color=BLUE, lw=2, ls="--", label="12-month extrapolation")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    _style(ax, "", "Price", "Natural gas: seasonal pattern on a rising trend")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def backtest_bars(bt: pd.DataFrame, path: str):
    bt = bt.sort_values("MAPE_%")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [BLUE if m.startswith("trend + annual") else NEUTRAL for m in bt["model"]]
    bars = ax.barh(bt["model"], bt["MAPE_%"], color=colors, height=0.55)
    for r, v in zip(bars, bt["MAPE_%"]):
        r.set_linewidth(0)
        ax.text(v + 0.08, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    _style(ax, "Mean absolute % error, final 12 months held out", "", "Seasonality is what matters")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
