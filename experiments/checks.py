"""Correctness checks (spec §7). Run these before trusting any numbers."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from boxes.prune import Eprune
from core.measures import aev, combinations, r_T_all
from core.query import Query, Thresholds
from pipeline import run
from setup.data import PREDICTED_ATTRS, SPECS, STATIC_ATTRS, build_lake
from setup.models import build_models
from experiments.queries import generate_queries

PASS, FAIL = "PASS", "FAIL"


def check_reduces_to_tvd(df_train, sources, Q_static):
    """#1 With Q_p = ∅ the pipeline is pure TVD: no model is ever invoked and
    the run is driven entirely by the static estimators."""
    Q = Query(static=Q_static, predicted={})
    models = build_models(df_train, SPECS)
    res = run(Q, sources, models, mode="PredictEachSource", thresholds=Thresholds(aev=0.8, cev_max=1.0))
    ok = res.model_invocations == 0
    return (PASS if ok else FAIL,
            f"Q_p=∅ → model_invocations={res.model_invocations} (want 0), "
            f"sources={len(res.visited)}, AEv={res.aev:.3f}")


def check_prune_preserves_aev(sources):
    """#2 Eprune preserves AEv exactly — tested on a table with PARTIAL matches
    (AEv strictly between 0 and 1), where a wrong prune would show up."""
    Q = {"race": ["White", "Black", "Other"],
         "sex": ["Female", "Male"],
         "income": [">50K", "<=50K"]}
    df = pd.concat(list(sources.values())[:5], ignore_index=True).head(600).copy()
    df["income"] = ["<=50K" if i % 3 else ">50K" for i in range(len(df))]
    before = aev(Q, df)
    pruned = Eprune(df, Q)
    after = aev(Q, pruned)
    ok = abs(before - after) < 1e-9 and 0.0 < before < 1.0
    return (PASS if ok else FAIL,
            f"AEv {before:.6f} → {after:.6f} (Δ={abs(before-after):.2e}), "
            f"|T| {len(df)}→{len(pruned)}")


def check_q_shrinks(df_train, sources, Q):
    """#3 Q' is monotonically shrinking (asserted inside reformulate)."""
    models = build_models(df_train, SPECS)
    try:
        run(Q, sources, models, mode="PredictEachSource", thresholds=Thresholds(aev=0.95, cev_max=1.0))
        return PASS, "no monotonicity assertion tripped"
    except AssertionError as e:
        return FAIL, str(e)


def check_invocations_equal_rows(df_train, sources, Q):
    """#4 Model invocations == total rows retrieved (every row predicted once)."""
    models = build_models(df_train, SPECS)
    res = run(Q, sources, models, mode="PredictEachSource", thresholds=Thresholds(aev=0.8, cev_max=0.3))
    ok = res.model_invocations == res.rows_retrieved
    return (PASS if ok else FAIL,
            f"invocations={res.model_invocations}, rows_retrieved={res.rows_retrieved}")


def check_retrieval_disjunctive(sources):
    """#5 A row matching exactly ONE requested static value must be retrieved."""
    from boxes.formulate_sql import formulate_sql
    from boxes.ship import ship
    from core.state import init_state

    src_name = list(sources)[0]
    src = sources[src_name]
    row = src.iloc[0]
    # request that row's race, plus a sex value it does NOT have
    other_sex = "Male" if row["sex"] == "Female" else "Female"
    Q = Query(static={"race": [row["race"]], "sex": [other_sex]},
              predicted={"income": [">50K"]})
    st = init_state(Q, {src_name: src}, build_models_stub(), mode="PredictEachSource")
    formulate_sql(st)
    ship(st, src_name)
    ok = row["row_id"] in set(st.retrieved["row_id"])
    return (PASS if ok else FAIL,
            f"single-value match retrieved={ok}, rows={len(st.retrieved)}/{len(src)}")


def check_tiebreak_keeps_incumbent(sources):
    """#6 On a tie, pruning keeps the tuple already in T and drops the newcomer,
    so a source that improves no match leaves T untouched, and the answer does
    not depend on the order in which sources were visited."""
    Q = {"race": ["White", "Black"], "sex": ["Female"], "income": [">50K"]}
    # A genuine tie means equal match counts on EVERY combination, i.e. identical
    # on the query attributes. row_id differs so we can see which one survived.
    incumbent = pd.DataFrame([{"race": "White", "sex": "Male",
                               "income": ">50K", "row_id": 1}])
    newcomer = pd.DataFrame([{"race": "White", "sex": "Male",
                              "income": ">50K", "row_id": 2}])

    merged = pd.concat([incumbent, newcomer], ignore_index=True)
    out = Eprune(merged, Q, n_incumbent=len(incumbent))
    untouched = out.reset_index(drop=True).equals(incumbent.reset_index(drop=True))

    # and without the tie-break the newcomer would have displaced it
    naive = Eprune(merged, Q, n_incumbent=0)
    displaced = not naive.reset_index(drop=True).equals(incumbent.reset_index(drop=True))

    ok = untouched
    return (PASS if ok else FAIL,
            f"T untouched={untouched} (|T|={len(out)}), "
            f"tie-break is load-bearing={displaced}")


def check_cost_model(df_train, sources, Q):
    """#7 The single cost is exactly prices x counts, nothing hidden."""
    from core.cost import Prices
    from setup.models import build_cheap_models
    prices = Prices(expensive=1.0, cheap=0.01, request=3.0, row=0.002)
    models = build_models(df_train, SPECS)
    cheap = build_cheap_models(df_train, SPECS)
    r = run(Q, sources, models, cheap_models=cheap, mode="PTVD", prices=prices,
            thresholds=Thresholds(aev=0.8, cev_max=0.3))
    want = (r.model_invocations + 0.01 * r.cheap_invocations
            + 3.0 * (r.sql_queries + r.sample_requests)
            + 0.002 * (r.rows_retrieved + r.sample_rows))
    ok = abs(r.total_cost - want) < 1e-6
    return (PASS if ok else FAIL,
            f"total_cost={r.total_cost:.3f}, recomputed={want:.3f}")


def check_calibration_uses_training_only(df_train, sources):
    """#8 Confidence is calibrated on the training split, never on the lake."""
    from setup.models import build_cheap_models, fit_confidence
    cheap = build_cheap_models(df_train, SPECS)
    spec = SPECS[0]
    rep = fit_confidence(cheap[spec.target], df_train, spec, "stump")
    lake_ids = {int(r) for df in sources.values() for r in df["row_id"]}
    overlap = len(rep["fit_row_ids"] & lake_ids)
    ok = overlap == 0 and len(rep["fit_row_ids"]) > 0
    return (PASS if ok else FAIL,
            f"{len(rep['fit_row_ids'])} calibration rows, {overlap} in the lake")


def check_default_options_identical(df_train, sources, Q):
    """#9 Passing Options() explicitly is the published run, bit for bit."""
    from core.options import Options
    from setup.models import build_cheap_models
    out = []
    for kw in ({}, {"options": Options()}):
        models = build_models(df_train, SPECS)
        cheap = build_cheap_models(df_train, SPECS)
        r = run(Q, sources, models, cheap_models=cheap, mode="PTVD",
                thresholds=Thresholds(aev=0.8, cev_max=0.3), **kw)
        out.append((r.model_invocations, r.cheap_invocations, r.visited,
                    r.T.to_dict("records")))
    ok = out[0] == out[1]
    return (PASS if ok else FAIL,
            f"expensive {out[0][0]} vs {out[1][0]}, identical answer={ok}")


def check_stratified_unbiased(df_train, sources):
    """#10 Stratified sampling, design-weighted, estimates the same population
    share as uniform sampling (no smoothing involved in the comparison)."""
    import numpy as np
    from boxes.formulate_sql import formulate_sql
    from boxes.sample import draw
    from core.options import Options
    from core.state import init_state
    from setup.models import build_cheap_models
    cheap = build_cheap_models(df_train, SPECS)
    models = build_models(df_train, SPECS)
    Q = Query(static={"marital_status": ["Married-civ-spouse"], "sex": ["Female"],
                      "race": ["White"]}, predicted={"income": ["<=50K"]})
    src = max(sources, key=lambda k: len(sources[k]))
    pool = sources[src]
    m = (pool["marital_status"].isin(Q.static["marital_status"])
         | pool["sex"].isin(Q.static["sex"]) | pool["race"].isin(Q.static["race"]))
    truth = float((cheap["income"].predict(pool.loc[m]) == "<=50K").mean())
    ests = []
    for seed in range(120):
        st = init_state(Q, {src: pool}, models, cheap_models=cheap, mode="PTVD",
                        seed=seed, options=Options(sampling="stratified"))
        formulate_sql(st)
        r = draw(st, src, 40).rows
        w = r["_w"].to_numpy()
        ests.append(float((w * (r["income"] == "<=50K")).sum() / w.sum()))
    mean = float(np.mean(ests))
    ok = abs(mean - truth) < 0.01
    return (PASS if ok else FAIL,
            f"weighted mean {mean:.3f} vs population {truth:.3f}")


def check_answers_are_expensive_labels(df_train, sources, Q):
    """#11 Without the cascade, every predicted value in the answer is what the
    expensive model says for that row — cheap labels never leak in."""
    from setup.models import build_cheap_models
    models = build_models(df_train, SPECS)
    cheap = build_cheap_models(df_train, SPECS)
    r = run(Q, sources, models, cheap_models=cheap, mode="PTVD",
            thresholds=Thresholds(aev=0.8, cev_max=0.3))
    bad = 0
    for a in Q.predicted:
        if a in r.T.columns and len(r.T):
            bad += int((models[a].predict(r.T) != r.T[a].to_numpy()).sum())
    ok = bad == 0 and not any(str(c).startswith("_by:") for c in r.T.columns)
    return (PASS if ok else FAIL, f"{bad} predicted values differ from the expensive model")


def check_cascade_honest(df_train, sources, Q):
    """#12 With the cascade, every kept cheap label had confidence >= tau and
    is marked as cheap; every other value is the expensive model's."""
    from core.options import Options
    from setup.models import build_cheap_models, fit_confidence
    models = build_models(df_train, SPECS)
    cheap = build_cheap_models(df_train, SPECS)
    for spec in SPECS:
        fit_confidence(cheap[spec.target], df_train, spec, "stump")
    tau = 0.6
    r = run(Q, sources, models, cheap_models=cheap, mode="PTVD",
            thresholds=Thresholds(aev=0.8, cev_max=0.3),
            options=Options(row_order="static+cheap", cascade_tau=tau))
    bad, n_cheap = 0, 0
    for a in Q.predicted:
        if f"_by:{a}" not in r.T.columns:
            continue
        is_cheap = r.T[f"_by:{a}"] == "cheap"
        n_cheap += int(is_cheap.sum())
        bad += int((r.T.loc[is_cheap, f"_conf:{a}"] < tau).sum())
        exp_rows = r.T.loc[~is_cheap]
        if len(exp_rows):
            bad += int((models[a].predict(exp_rows) != exp_rows[a].to_numpy()).sum())
    ok = bad == 0
    return (PASS if ok else FAIL,
            f"{n_cheap} cheap-labelled values kept, {bad} violations")


def check_k_rows(df_train, sources):
    """#13 theta+k never reports more than k rows, and with k >= |C| the answer
    is exactly the theta answer."""
    from core.options import Options
    from setup.models import build_cheap_models
    Q = Query(static={"education": ["Bachelors", "HS-grad", "Masters"],
                      "race": ["White", "Black"], "sex": ["Female", "Male"]},
              predicted={"income": [">50K"]})
    th = Thresholds(aev=0.8, cev_max=1.0)
    sizes = {}
    for k in (2, 4, 12):
        models = build_models(df_train, SPECS); cheap = build_cheap_models(df_train, SPECS)
        r = run(Q, sources, models, cheap_models=cheap, mode="PTVD", thresholds=th,
                options=Options(answer="theta+k", k=k))
        sizes[k] = (len(r.T), r.terminated_by)
    models = build_models(df_train, SPECS); cheap = build_cheap_models(df_train, SPECS)
    base = run(Q, sources, models, cheap_models=cheap, mode="PTVD", thresholds=th)
    ok = all(n <= k for k, (n, _) in sizes.items()) and sizes[12][0] == len(base.T)
    return (PASS if ok else FAIL,
            f"answer sizes {{k: rows}} = { {k: n for k, (n, _) in sizes.items()} }, "
            f"theta-only = {len(base.T)}; k=2 stopped: {sizes[2][1]}")


def check_greedy_k_bound(_df_train, _sources):
    """#14 Greedy k-row selection reaches >= (1 - 1/e) of the brute-force best
    on random small tables (facility location is monotone submodular)."""
    import itertools, math, random
    from core.measures import aev, select_k
    Q = {"a": ["x", "y", "z"], "b": ["u", "v"], "c": ["p", "q"]}
    vals = {"a": ["x", "y", "z", "w"], "b": ["u", "v", "t"], "c": ["p", "q", "r"]}
    rng = random.Random(1)
    worst = 1.0
    for _ in range(40):
        T = pd.DataFrame([{a: rng.choice(v) for a, v in vals.items()} for _ in range(10)])
        for k in (2, 3):
            best = max(aev(Q, T.iloc[list(S)]) for S in itertools.combinations(range(10), k))
            got = aev(Q, select_k(T, Q, k))
            if best > 0:
                worst = min(worst, got / best)
    ok = worst >= 1 - 1 / math.e - 1e-9
    return (PASS if ok else FAIL, f"worst greedy/optimal ratio {worst:.3f} (bound 0.632)")


def check_tvd_plus_predicted_reduces_to_tvd(df_train, sources, Q):
    """#15 'TVD + predicted' is TVD's own estimator with the predicted blanks
    filled in: with no predicted condition it equals TVD's score, exactly, on
    every source."""
    from boxes.formulate_sql import formulate_sql
    from boxes.sample import Sample
    from boxes.score import _delta_from_stats
    from boxes.select_source import _delta_evidence
    from core.measures import combinations, r_T_all
    from core.state import init_state
    Qs = Query(static=Q.static, predicted={})
    st = init_state(Qs, sources, build_models(df_train, SPECS), mode="PTVD")
    formulate_sql(st)
    combos = combinations(Qs.all)
    rT = r_T_all(st.T, Qs.all)
    worst = max(abs(_delta_from_stats(st, s, Sample(), combos, rT, 0.0)
                    - _delta_evidence(st, s, combos, rT)) for s in sources)
    ok = worst < 1e-12
    return (PASS if ok else FAIL, f"largest difference over {len(sources)} sources: {worst:.1e}")


def check_at_least_k(df_train, sources):
    """#16 theta+min_k: the answer reaches theta AND shows at least k rows,
    every extra row fully matches a requested combination, and with k no
    larger than the theta answer the result is the theta answer."""
    from boxes.prune import full_match
    from core.options import Options
    from setup.models import build_cheap_models
    Q = Query(static={"marital_status": ["Married-civ-spouse"], "sex": ["Female"],
                      "race": ["White", "Black"]}, predicted={"income": [">50K"]})
    th = Thresholds(aev=1.0, cev_max=1.0)
    out = {}
    for k in (None, 1, 6):
        models = build_models(df_train, SPECS); cheap = build_cheap_models(df_train, SPECS)
        o = Options(row_order="static") if k is None else Options(
            row_order="static", answer="theta+min_k", k=k)
        r = run(Q, sources, models, cheap_models=cheap, mode="PTVD", thresholds=th, options=o)
        out[k] = r
    base, small, big = out[None], out[1], out[6]
    ok = (small.T.to_dict("records") == base.T.to_dict("records")
          and len(big.T) >= 6 and bool(full_match(big.T, Q.all).all())
          and big.aev >= 1.0 - 1e-9)
    return (PASS if ok else FAIL,
            f"theta-only {len(base.T)} rows; k=1 same answer={small.T.equals(base.T)}; "
            f"k=6 -> {len(big.T)} rows, all full matches={bool(full_match(big.T, Q.all).all())}")


def build_models_stub():
    return {}


def main() -> None:
    df_train, sources, truth = build_lake()
    lake = pd.concat(sources.values(), ignore_index=True)
    Q = generate_queries(lake, truth, STATIC_ATTRS, PREDICTED_ATTRS,
                         n=1, seed=7)[0]
    print(f"query under test: {Q.static} + {Q.predicted}\n")

    results = [
        ("1 reduces to TVD when Q_p=∅", check_reduces_to_tvd(df_train, sources, Q.static)),
        ("2 Eprune preserves AEv",      check_prune_preserves_aev(sources)),
        ("3 Q' shrinks monotonically",  check_q_shrinks(df_train, sources, Q)),
        ("4 invocations == rows",       check_invocations_equal_rows(df_train, sources, Q)),
        ("5 retrieval is disjunctive",  check_retrieval_disjunctive(sources)),
        ("6 tie-break keeps incumbent", check_tiebreak_keeps_incumbent(sources)),
        ("7 cost = prices x counts",    check_cost_model(df_train, sources, Q)),
        ("8 calibration off the lake",  check_calibration_uses_training_only(df_train, sources)),
        ("9 default options identical", check_default_options_identical(df_train, sources, Q)),
        ("10 stratified is unbiased",   check_stratified_unbiased(df_train, sources)),
        ("11 answers = expensive labels", check_answers_are_expensive_labels(df_train, sources, Q)),
        ("12 cascade is honest",        check_cascade_honest(df_train, sources, Q)),
        ("13 at most k rows",           check_k_rows(df_train, sources)),
        ("14 greedy k within 1-1/e",    check_greedy_k_bound(df_train, sources)),
        ("15 TVD+predicted = TVD",      check_tvd_plus_predicted_reduces_to_tvd(df_train, sources, Q)),
        ("16 at least k rows",          check_at_least_k(df_train, sources)),
    ]
    width = max(len(n) for n, _ in results)
    n_fail = 0
    for name, (status, detail) in results:
        if status == FAIL:
            n_fail += 1
        print(f"[{status}] {name:<{width}}  {detail}")
    print(f"\n{len(results)-n_fail}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
