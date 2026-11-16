"""Calibrate θ_CEv against what CEv can actually reach.

A fixed θ_CEv = 0.6 is unreachable by construction on these lakes, and that is
a property of the measure, not of any method:

  * CEv averages over the query's attributes the fraction of PRESENT values
    that are unrequested. An attribute with one requested and one unrequested
    value therefore caps at 1/2, so any binary predicted attribute drags the
    average down however good the search is.
  * CIKM's own post-prune CEv on its three datasets was 0.55 / 0.37 / 0.43 —
    every value below 0.6.

So the honest procedure is to measure the ceiling per query — the CEv of the
table you get by retrieving everything the disjunctive query matches, labelling
all of it, and pruning — and set the target as a FRACTION of that ceiling.
Chasing a fixed constant by weakening CEprune would be fitting the algorithm to
an unreachable number.

    python -m experiments.calibrate_cev --dataset movielens --queries 20
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib

import numpy as np
import pandas as pd

from boxes.prune import CEprune
from core.measures import cev, vev
from experiments.queries import generate_queries
from experiments.run_baselines import DATASETS
from setup.models import build_models


def ceiling_for(Q, lake, models) -> tuple[float, float]:
    """(CEv, VEv) of the fully-retrieved, fully-labelled, pruned table."""
    mask = pd.Series(False, index=lake.index)
    for a, vals in Q.static.items():
        if a in lake.columns:
            mask |= lake[a].isin(vals)
    rows = lake.loc[mask]
    if len(rows) == 0:
        return 0.0, 0.0
    rows = rows.copy()
    for attr in Q.predicted:
        if attr in models:
            rows[attr] = models[attr].predict(rows)
    pruned = CEprune(rows, Q.all)
    return cev(Q.all, pruned), vev(Q.all, pruned)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="movielens")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mod, kw = DATASETS[args.dataset]
    ds = importlib.import_module(mod)
    df_train, sources, truth = ds.build_lake(**kw)
    lake = pd.concat(sources.values(), ignore_index=True)
    models = build_models(df_train, ds.SPECS)

    Qs = generate_queries(lake, truth, ds.STATIC_ATTRS, ds.PREDICTED_ATTRS,
                          n=args.queries, seed=args.seed, min_static_attrs=2)

    cevs, vevs = [], []
    for Q in Qs:
        c, v = ceiling_for(Q, lake, models)
        cevs.append(c)
        vevs.append(v)
    a = np.array(cevs)

    print(f"dataset={args.dataset}  {len(Qs)} queries  "
          f"({len(ds.PREDICTED_ATTRS)} predicted attr(s))\n")
    print(f"achievable CEv (predict-everything, then CEprune):")
    print(f"  min={a.min():.3f}  median={np.median(a):.3f}  "
          f"mean={a.mean():.3f}  max={a.max():.3f}")
    print(f"  VEv at that point: median={np.median(vevs):.3f}\n")

    print("fixed thresholds — fraction of queries where the target is even POSSIBLE:")
    for th in (0.6, 0.5, 0.4, 0.3):
        print(f"  θ_CEv={th:<4} reachable by {100*(a >= th).mean():>5.0f}% of queries")

    print("\ncalibrated thresholds — θ as a fraction of each query's own ceiling:")
    for frac in (0.9, 0.8, 0.7, 0.6):
        print(f"  θ = {frac:.0%} of ceiling  → median target "
              f"{np.median(a * frac):.3f}  (reachable by 100% by construction)")

    print("\nRecommendation: report CEv against the per-query ceiling, or set "
          "θ_CEv = 0.8 x ceiling.\nA fixed 0.6 tests whether the DATA can "
          "supply unrequested values, not whether\nthe METHOD found them.")


if __name__ == "__main__":
    main()
