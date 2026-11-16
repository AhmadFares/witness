"""The shared harness. Only box calls and loop control live here.

Three modes, all on the same code — that is the point: any difference in
results is attributable to WHERE prediction happens and HOW sources are
selected, nothing else.

  PredictEachSource   static selection; predicts every retrieved row of every
                      source it visits, as it goes
  PredictAtEnd        pure TVD: searches the static query only (blind to the
                      model), then predicts the final table once
  PredictEverything   visits every source and predicts everything — the ceiling

PTVD will be a fourth mode that changes select_source and nothing else.
"""

from __future__ import annotations

import pandas as pd

from boxes import (
    check_target, formulate_sql, predict, prune, reformulate, select_source,
    ship,
)
from boxes.check_target import plateau
from boxes.select_source import candidate_window
from core.measures import aev, cev, r_T_all, vev
from core.query import COUNTER_EVIDENCE, EVIDENCE, Query, Thresholds
from core.state import Result, RunState, init_state

CHECK = True        # runtime invariant assertions (cheap; disable for timing runs)


def pick_source(state: RunState):
    """Choose the next source, contacting as few as possible.

    Histograms are local, so ranking every source is free; only a top-κ window
    is contacted. If no source in the window can improve the answer, the
    window widens. Contacting all sources up front would be the AllSources
    pattern the design avoids.
    """
    if state.mode == "random":
        # TVD's Random baseline: any remaining source, in arbitrary order (not
        # only the statistics' top window).
        src = state.rng.choice(list(state.remaining))
        state.note("selected", source=src, among=["any remaining source"])
        return src
    offset = 0
    while offset < len(state.remaining):
        window = candidate_window(state, offset=offset)
        if not window:
            break
        state.note("window", offset=offset, contacted=list(window))
        src = select_source(state, window)
        if src is not None:
            state.note("selected", source=src, among=list(window))
            return src
        state.note("no_pick", among=list(window))
        offset += len(window)
    return None


def run(Q: Query, D: dict[str, pd.DataFrame], models: dict,
        mode: str = "PredictEachSource",
        incremental: bool | None = None,
        stop_on_zero_gain: bool = False,
        **cfg) -> Result:
    """PTVD's two mechanisms — sampled selection and incremental consumption —
    are independently switchable so each can be ablated. Left unset,
    incremental consumption is on for PTVD and off for the baselines."""
    if incremental is None:
        incremental = (mode == "PTVD")
    state = init_state(Q, D, models, mode=mode, **cfg)
    state.incremental = incremental
    state.stop_on_zero_gain = stop_on_zero_gain
    predict_each_round = (mode != "PredictAtEnd")

    from boxes.check_target import k_infeasible
    if k_infeasible(state):
        state.terminated_by = "infeasible_k"
        state.note("infeasible", k=state.options.k,
                   why="no answer of k rows can reach the threshold")
        state.remaining = []

    if mode == "Proxy" and state.remaining:
        return _run_proxy(state)

    while state.remaining:
        formulate_sql(state)
        src = pick_source(state)
        if src is None:
            state.terminated_by = "no_source"
            break

        ship(state, src)
        rows = predict(state) if predict_each_round else state.retrieved

        # Incumbents come first in `merged`; Eprune breaks ties in their favour,
        # so a source improving no match leaves T untouched (see prune.py).
        n_incumbent = len(state.T)
        # The tie-break invariant is an Eprune property, so it is asserted under
        # EVIDENCE only. CEprune is a greedy cover that legitimately reselects
        # rows each round.
        check_stable = (CHECK and state.semantics == EVIDENCE
                        and state.options.answer != "theta+min_k")
        T_before = state.T.copy() if check_stable else None
        r_before = r_T_all(state.T, state.Q.all) if check_stable else None

        T_prev = state.T
        merged = pd.concat([state.T, rows], ignore_index=True) if len(rows) else state.T
        before = len(merged)
        if state.scan_pool is not None:
            state.T, state.scan_pool = state.scan_pool, None
        else:
            state.T = prune(state, merged, n_incumbent=n_incumbent)

        if check_stable and r_T_all(state.T, state.Q.all) == r_before:
            # If no combination improved, T must be untouched — not merely
            # equal in measure, so the answer does not depend on the order in
            # which sources were visited.
            assert state.T.astype(object).equals(T_before.astype(object)), \
                "non-improving round modified T (tie-break violated)"

        state.log_round(src, before)

        if mode != "PredictEverything" and check_target(state):
            state.terminated_by = "target_met"
            break
        # DIVERSE: give up once a source raised neither TWR nor DWR by more
        # than ε (TVD's rule): the target is then unlikely to be reached.
        if mode != "PredictEverything" and state.semantics == COUNTER_EVIDENCE:
            flat = plateau(state.Q_full.all, state.answer_of(T_prev),
                           state.answer_of(state.T), state.thresholds)
            state.flat_rounds = state.flat_rounds + 1 if flat else 0
            limit = state.options.patience if state.targets_diverse_gap else 1
            if state.flat_rounds >= limit:
                state.terminated_by = "plateau"
                break
        if state.options.budget is not None and state.cost >= state.options.budget:
            state.terminated_by = "budget_spent"
            break
        # PTVD under DIVERSE keeps the full query: dropping a satisfied value
        # would hide the rows of a table that holds the diversity still missing.
        if (mode != "PredictEverything" and not state.needs_evidence()
                and not state.targets_diverse_gap):
            # (the ceiling never narrows its query; and while more evidence is
            # wanted, values already covered must stay in the query)
            reformulate(state)

    if mode == "PredictAtEnd":                      # predict the final answer, once
        state.Q = state.Q_full            # from here on the predicted attr exists
        state.T = predict(state, state.T)
        state.T = prune(state, state.T)

    return _finish(state)


def _run_proxy(state: RunState) -> Result:
    """Proxy ranking over the whole lake, the approach of proxy-based query
    processing (e.g. SUPG): ignore source boundaries, fetch every relevant row,
    rank all rows by stored-value match and then by the cheap model's score
    (row order "static+cheap"), and label them with the expensive model in
    that order, in chunks, until the target is reached. It reuses PTVD's own
    labelling and stopping code, so only the treatment of sources differs."""
    formulate_sql(state)
    parts = []
    for src in list(state.remaining):
        ship(state, src)
        parts.append(state.retrieved)
    state.retrieved = pd.concat(parts, ignore_index=True)
    rows = predict(state)
    merged = pd.concat([state.T, rows], ignore_index=True) if len(rows) else state.T
    before = len(merged)
    state.T = prune(state, merged, n_incumbent=len(state.T))
    state.log_round(f"all {len(state.visited)} sources", before)
    if check_target(state):
        state.terminated_by = "target_met"
    return _finish(state)


def _finish(state: RunState) -> Result:
    Q_all = state.Q_full.all       # always score against the query the user asked
    state.T = state.answer_of(state.T)
    return Result(
        T=state.T,
        visited=state.visited,
        model_invocations=state.model_invocations,
        cheap_invocations=state.cheap_invocations,
        sql_queries=state.sql_queries,
        rows_retrieved=state.rows_retrieved,
        aev=aev(Q_all, state.T),
        vev=vev(Q_all, state.T),
        cev=cev(Q_all, state.T),
        terminated_by=state.terminated_by,
        sample_rows=state.sample_rows,
        rows_skipped=state.rows_skipped,
        plateau_stops=state.plateau_stops,
        rounds=state.rounds,
        seconds_sql=state.t_sql,
        seconds_inference=state.t_inference,
        seconds_local=state.t_local,
        mode=state.mode,
        trace=state.trace or [],
        sample_requests=state.sample_requests,
        total_cost=state.cost,
        answer_rows=len(state.T),
        rows_labelled=state.rows_labelled,
    )


if __name__ == "__main__":
    from setup.data import build_lake
    from setup.models import build_models

    df_train, sources, truth = build_lake()

    Q = Query(
        static={"marital_status": ["Married-civ-spouse"], "sex": ["Female"]},
        predicted={"income": [">50K"]},
    )
    th = Thresholds(aev=0.8, cev_max=0.3)

    for mode in ("PredictEachSource", "PredictAtEnd", "PredictEverything"):
        models = build_models(df_train)          # fresh counters per run
        res = run(Q, sources, models, mode=mode, thresholds=th)
        print(res.summary())
