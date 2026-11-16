"""Figures from results/<dataset>_baselines.json — no re-running needed.

    python -m experiments.plots --dataset movielens

Two figures, both threshold-first:

  cost_to_target   For each method, mean expensive invocations on the runs that
                   REACHED the target, with the fraction reaching it annotated.
                   This is the paper's headline: the objective is to reach θ,
                   so cost is only comparable among runs that got there.

  trajectory       Quality against cumulative expensive invocations, averaged
                   over queries. Shows HOW methods approach θ, not just where
                   they stop — the shape is the argument for allocating
                   inference rather than spending it uniformly.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = ["PredictAtEnd", "static-naive", "static", "PredictEverything",
         "PTVD-noIncr", "PTVD"]
COLOR = {"PTVD": "#2a7", "static": "#c63", "PredictEverything": "#666",
         "static-naive": "#e9a", "PredictAtEnd": "#9bd",
         "PTVD-noIncr": "#adc"}


def load(dataset: str) -> pd.DataFrame:
    path = RESULTS / f"{dataset}_baselines.json"
    if not path.exists():
        sys.exit(f"no results at {path} — run experiments.run_baselines first")
    return pd.DataFrame(json.loads(path.read_text()))


def fig_cost_to_target(df: pd.DataFrame, dataset: str, theta: float) -> Path:
    d = df[df.semantics == "EVIDENCE"]
    labels = [c for c in ORDER if c in set(d.config)]
    cost, frac = [], []
    for c in labels:
        sub = d[d.config == c]
        met = sub[sub.met]
        cost.append(met.model_invocations.mean() if len(met) else np.nan)
        frac.append(sub.met.mean())

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(labels, cost, color=[COLOR.get(c, "#999") for c in labels],
                  width=0.6)
    for b, c, f in zip(bars, cost, frac):
        if np.isnan(c):
            continue
        ax.text(b.get_x() + b.get_width() / 2, c,
                f"{c:,.0f}\n{f*100:.0f}% met", ha="center", va="bottom",
                fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("expensive model invocations (log)")
    ax.set_title(f"{dataset}: cost to reach the target (θ_AEv={theta}); "
                 f"lower is better, % = runs reaching it")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = RESULTS / f"{dataset}_cost_to_target.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_trajectory(df: pd.DataFrame, dataset: str, theta: float) -> Path:
    """Quality vs cumulative expensive calls, interpolated onto a common grid."""
    d = df[df.semantics == "EVIDENCE"]
    # PredictAtEnd has no trajectory: it predicts once, after the loop,
    # so its in-loop AEv is structurally 0 and the curve is meaningless.
    labels = [c for c in ORDER if c in set(d.config) and c != "PredictAtEnd"]
    hi = max(
        (r["model_invocations"] for _, row in d.iterrows() for r in row["rounds"]),
        default=1)
    grid = np.geomspace(1, max(hi, 10), 120)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for c in labels:
        curves = []
        for _, row in d[d.config == c].iterrows():
            rounds = row["rounds"]
            if not rounds:
                continue
            x = np.array([0] + [r["model_invocations"] for r in rounds], float)
            y = np.array([0.0] + [r["AEv"] for r in rounds], float)
            keep = np.concatenate(([True], np.diff(x) > 0))
            curves.append(np.interp(grid, np.maximum(x[keep], 1), y[keep],
                                    left=0.0, right=y[keep][-1]))
        if curves:
            ax.plot(grid, np.mean(curves, axis=0), label=c,
                    color=COLOR.get(c, "#999"), lw=2)

    ax.axhline(theta, ls="--", lw=1, color="k", alpha=.6)
    ax.text(hi, theta, f"θ={theta} ", va="bottom", ha="right", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, hi)
    ax.set_xlabel("cumulative expensive model invocations (log)")
    ax.set_ylabel("AEv")
    ax.set_title(f"{dataset}: quality against inference spent")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out = RESULTS / f"{dataset}_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--theta", type=float, default=0.8)
    args = ap.parse_args()
    df = load(args.dataset)
    print(fig_cost_to_target(df, args.dataset, args.theta))
    print(fig_trajectory(df, args.dataset, args.theta))


if __name__ == "__main__":
    main()
