"""CheckTarget — evaluated on the PRUNED T, against the ORIGINAL Q (spec §2.4).

  EVIDENCE (ASSOCIATIVE):     AEv (AWR) ≥ θ_AEv, as in the TVD paper
  COUNTER_EVIDENCE (DIVERSE): VEv (TWR) ≥ θ_VEv and CEv (DWR) ≥ θ_CEv; the
                              run also gives up once a source raises neither
                              VEv nor CEv by more than ε (TVD's plateau rule),
                              as the target is then unlikely to be reached

`target_met` is the single definition of "done" — the pipeline and the
experiment reporting both call it, so a run counted as successful in a table
is exactly a run the pipeline would have stopped on.
"""

from __future__ import annotations

import pandas as pd

from core.measures import aev, aev_upper_bound_k, cev, vev, vev_upper_bound_k
from core.query import COUNTER_EVIDENCE, Thresholds
from core.state import RunState


def target_met(Q_all: dict[str, list], T: pd.DataFrame,
               th: Thresholds, semantics: str) -> bool:
    if T is None or len(T) == 0:
        return False
    if semantics == COUNTER_EVIDENCE:
        return vev(Q_all, T) >= th.vev and cev(Q_all, T) >= th.cev_min
    return aev(Q_all, T) >= th.aev


def plateau(Q_all: dict[str, list], before: pd.DataFrame | None,
            after: pd.DataFrame, th: Thresholds) -> bool:
    """DIVERSE give-up rule: going from `before` to `after` (one source)
    raised neither TWR nor DWR by more than ε."""
    return (vev(Q_all, after) - vev(Q_all, before) < th.eps
            and cev(Q_all, after) - cev(Q_all, before) < th.eps)


def goal_reached(state: RunState, T: pd.DataFrame) -> bool:
    """When a run (or a scan inside a source) may stop.

    theta      the published test, on the pool
    theta+k    the same test, on the best k-row answer
    theta+min_k the same test, and the answer shows at least k rows
    """
    A = state.answer_of(T)
    if not target_met(state.Q.all, A, state.thresholds, state.semantics):
        return False
    return not state.needs_evidence(A)          # theta+min_k: and k rows shown


def k_infeasible(state: RunState) -> bool:
    """theta+k: no answer of k rows can reach theta, whatever the data."""
    o = state.options
    if o.answer != "theta+k":
        return False
    Q_all = state.Q.all
    if state.semantics == COUNTER_EVIDENCE:
        return state.thresholds.vev > vev_upper_bound_k(Q_all, o.k) + 1e-9
    return state.thresholds.aev > aev_upper_bound_k(Q_all, o.k) + 1e-9


def check_target(state: RunState) -> bool:
    """In-loop stop, evaluated on the query actually being SEARCHED.

    For PredictAtEnd that is the static query — it is pure TVD and stops when
    TVD's own target is met, having never consulted the model. Reporting still
    scores every run against the full query via `target_met`, so a run can stop
    here and still be counted as not meeting the user's real target.
    """
    return goal_reached(state, state.T)
