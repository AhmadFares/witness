"""Reformulate — produce Q' from Q and T (spec §2.5). Q' only ever shrinks.

  EVIDENCE:         drop requested value (a,v) once EVERY combination
                    containing it is fully matched in T (r_T(c) = k).
  COUNTER_EVIDENCE: drop (a,v) once v ∈ values(a,T) AND attribute a holds at
                    least q_a = ceil(r_a · θ_CEv / (1 − θ_CEv)) distinct
                    unrequested values, r_a = |values(a,Q)|.
"""

from __future__ import annotations

import math
import time

from core.measures import combinations, r_T_all
from core.query import COUNTER_EVIDENCE, Query
from core.state import RunState


def _reformulate_evidence(state: RunState) -> Query:
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    k = len(cols)
    rT = r_T_all(state.T, Q_all)

    # (a,v) is droppable iff every combination containing it is fully matched.
    satisfied: dict[tuple, bool] = {}
    for combo in combinations(Q_all):
        full = (rT[combo] == k)
        for a, v in zip(cols, combo):
            key = (a, v)
            satisfied[key] = satisfied.get(key, True) and full

    new = state.Q.copy()
    for part in (new.static, new.predicted):
        for a in list(part.keys()):
            part[a] = [v for v in part[a] if not satisfied.get((a, v), False)]
    return new


def _reformulate_counter_evidence(state: RunState) -> Query:
    th = state.thresholds
    Q_all = state.Q.all
    new = state.Q.copy()
    ratio = th.cev_min / (1.0 - th.cev_min) if th.cev_min < 1.0 else float("inf")

    for part in (new.static, new.predicted):
        for a in list(part.keys()):
            if a not in state.T.columns:
                continue
            present = set(state.T[a].dropna().unique())
            requested = set(Q_all.get(a, []))
            r_a = max(1, len(requested))
            q_a = math.ceil(r_a * ratio) if ratio != float("inf") else float("inf")
            n_unrequested = len(present - requested)
            if n_unrequested >= q_a:
                part[a] = [v for v in part[a] if v not in present]
    return new


def reformulate(state: RunState) -> None:
    t0 = time.perf_counter()
    before = state.Qp.n_values()

    if state.semantics == COUNTER_EVIDENCE:
        new = _reformulate_counter_evidence(state)
    else:
        new = _reformulate_evidence(state)

    # Q' is monotone: never re-add a value dropped in an earlier round.
    for part_new, part_old in ((new.static, state.Qp.static),
                               (new.predicted, state.Qp.predicted)):
        for a in list(part_new.keys()):
            part_new[a] = [v for v in part_new[a] if v in part_old.get(a, [])]

    after = new.n_values()
    assert after <= before, f"Q' grew: {before} -> {after}"   # spec check #3
    state.Qp = new
    state.t_local += time.perf_counter() - t0
