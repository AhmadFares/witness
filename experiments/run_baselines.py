"""THE experiment. One command reproduces every reported number.

    python -m experiments.run_baselines --dataset adult --quick
    python -m experiments.run_baselines --dataset movielens --queries 40
    python -m experiments.run_baselines --dataset adult --semantics both

Headline metric (threshold-first, per the problem statement): the objective is
to REACH the target thresholds, not to maximise quality. So the leading
columns are `met` (fraction of runs reaching the target) and `cost|met` (mean
expensive invocations on the runs that reached it). Quality beyond θ is worth
nothing and is reported only as context.

Effective cost folds the cheap-model traffic in at the MEASURED speed ratio:
    eff = expensive + cheap / R
so no call is hidden (the cheap tier is ~13-25x faster, not free).

Writes results/<dataset>_baselines.json (one record per run, with per-round
trajectory) so plots can be produced without re-running.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib
import json

import numpy as np
import pandas as pd

from boxes import target_met
from core.query import COUNTER_EVIDENCE, EVIDENCE, Thresholds
from pipeline import run
from setup.models import build_cheap_models, build_models
from experiments.queries import generate_queries

RESULTS = Path(__file__).resolve().parents[1] / "results"

DATASETS = {
    # name -> (module, build_lake kwargs)
    "adult":          ("setup.data", {}),
    "movielens":      ("setup.movielens", {}),
    "movielens-full": ("setup.movielens",
                       dict(n_rows=None, partition=("occupation", "decade"))),
    "covertype":      ("setup.covertype", {}),
}

CONFIGS = [
    ("PredictAtEnd",      dict(mode="PredictAtEnd")),
    ("static-naive",      dict(mode="PredictEachSource", stop_on_zero_gain=True)),
    ("static",            dict(mode="PredictEachSource")),          # repaired
    ("PredictEverything", dict(mode="PredictEverything")),
    # ── ablations: strip one PTVD mechanism at a time ─────────────────────
    ("PTVD-noIncr",       dict(mode="PTVD", incremental=False)),
    ("PTVD",              dict(mode="PTVD")),
]
USES_CHEAP = {"PTVD", "PTVD-noIncr"}

SEMANTICS = {
    EVIDENCE:         dict(th=Thresholds(aev=0.8, cev_max=0.3), min_static_attrs=1),
    # CounterEvidence needs >= 2 static attributes: with one, disjunctive
    # retrieval returns only rows carrying the requested value and CEv is
    # pinned at a constant, so the semantics cannot be exercised.
    # theta_CEv CALIBRATED, not inherited. CEv averages over query attributes
    # the fraction of present values that are unrequested, so an attribute with
    # one requested and one unrequested value caps at 1/2 and any binary
    # predicted attribute drags the mean down regardless of the method. Measured
    # ceiling (predict everything, then CEprune) on Adult: median 0.50, and
    # theta=0.6 is reachable by only 5% of queries — which is why every method
    # scored 0%. CIKM's own post-prune CEv was 0.55/0.37/0.43, all below 0.6.
    # 0.40 is ~80% of the measured median ceiling; see experiments.calibrate_cev.
    COUNTER_EVIDENCE: dict(th=Thresholds(vev=1.0, cev_min=0.40), min_static_attrs=2),
}


def measure_speed_ratio(exp_models: dict, cheap_models: dict,
                        lake: pd.DataFrame, reps: int = 3) -> float:
    """Measured expensive:cheap wall-clock ratio on this lake's rows."""
    probe = lake.head(min(len(lake), 20000))

    def bench(models):
        for m in models.values():
            m.predict(probe.head(8))
        t = time.perf_counter()
        for _ in range(reps):
            for m in models.values():
                m.predict(probe)
        return (time.perf_counter() - t) / reps

    te, tc = bench(exp_models), bench(cheap_models)
    for m in (*exp_models.values(), *cheap_models.values()):
        m.reset()
    return te / tc if tc > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="adult")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--semantics", choices=["evidence", "both"], default="evidence")
    ap.add_argument("--quick", action="store_true", help="1 seed, evidence only")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="subset of config labels to run")
    args = ap.parse_args()

    seeds = args.seeds[:1] if args.quick else args.seeds
    semantics = ({EVIDENCE: SEMANTICS[EVIDENCE]}
                 if (args.quick or args.semantics == "evidence") else SEMANTICS)
    configs = ([c for c in CONFIGS if c[0] in set(args.configs)]
               if args.configs else CONFIGS)

    mod_name, build_kw = DATASETS[args.dataset]
    ds = importlib.import_module(mod_name)
    df_train, sources, truth = ds.build_lake(**build_kw)
    lake = pd.concat(sources.values(), ignore_index=True)

    # Train ONCE. Counters are reset per run — retraining gradient boosting for
    # every run costs more than the entire experiment it is measuring.
    exp_models = build_models(df_train, ds.SPECS)
    cheap_models = build_cheap_models(df_train, ds.SPECS)
    R = measure_speed_ratio(exp_models, cheap_models, lake)

    RESULTS.mkdir(exist_ok=True)
    print(f"dataset={args.dataset}: {len(sources)} sources, {len(lake)} lake rows, "
          f"{len(ds.PREDICTED_ATTRS)} predicted attr(s), cheap speedup R={R:.1f}x")
    print(f"{args.queries} queries x {len(seeds)} seed(s) x {len(configs)} configs "
          f"x {len(semantics)} semantics\n")

    records, rows = [], []
    t_all = time.perf_counter()
    for sem, cfg in semantics.items():
        th = cfg["th"]
        for seed in seeds:
            queries = generate_queries(
                lake, truth, ds.STATIC_ATTRS, ds.PREDICTED_ATTRS,
                n=args.queries, seed=seed,
                min_static_attrs=cfg["min_static_attrs"])
            for qi, Q in enumerate(queries, 1):
                for label, kw in configs:
                    for m in (*exp_models.values(), *cheap_models.values()):
                        m.reset()
                    res = run(Q, sources, exp_models,
                              cheap_models=(cheap_models
                                            if label in USES_CHEAP else None),
                              thresholds=th, semantics=sem, **kw)
                    s = res.summary()
                    s.update(config=label, semantics=sem, seed=seed,
                             query=f"Q{qi}", dataset=args.dataset,
                             met=bool(target_met(Q.all, res.T, th, sem)),
                             effective=res.model_invocations
                                       + res.cheap_invocations / R)
                    rows.append(s)
                    records.append({**s, "rounds": res.rounds})
            print(f"  done {sem} seed={seed}  [{time.perf_counter()-t_all:.0f}s]",
                  flush=True)

    df = pd.DataFrame(rows)
    order = [c for c, _ in configs]

    for sem in semantics:
        d = df[df.semantics == sem]
        th = semantics[sem]["th"]
        tgt = (f"VEv>={th.vev} & CEv>={th.cev_min}" if sem == COUNTER_EVIDENCE
               else f"AEv>={th.aev} & CEv<={th.cev_max}")
        print("\n" + "=" * 104)
        print(f"{sem}  target: {tgt}   ({args.queries} queries x {len(seeds)} seeds)"
              f"   [threshold-first: met and cost|met are the headline]")

        def agg_one(cdf):
            met = cdf[cdf.met]
            return pd.Series({
                "met": f"{100*cdf.met.mean():.0f}%",
                "cost|met": met.model_invocations.mean() if len(met) else np.nan,
                "eff|met": met.effective.mean() if len(met) else np.nan,
                "expensive": cdf.model_invocations.mean(),
                "cheap": cdf.cheap_invocations.mean(),
                "sql": cdf.sql_queries.mean(),
                "sources": cdf.sources.mean(),
                "AEv": cdf.AEv.mean(),
            })

        print(d.groupby("config").apply(agg_one, include_groups=False)
               .reindex(order).round(1).to_string())

    out = RESULTS / f"{args.dataset}_baselines.json"
    out.write_text(json.dumps(records, indent=2, default=str))
    print(f"\nwrote {out}  ({len(records)} runs, {time.perf_counter()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
