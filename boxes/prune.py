"""Prune — Eprune (EVIDENCE) and CEprune (COUNTER_EVIDENCE), spec §2.2.

Both are destructive and applied every round to T ∪ S_M.
Eprune preserves AEv exactly; CEprune preserves VEv exactly. In test mode the
preservation is asserted — if it fails, the implementation is wrong.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from core.measures import aev, vev
from core.query import COUNTER_EVIDENCE
from core.state import RunState

CHECK = True          # assert measure preservation (spec §2.2, check #2)
TOL = 1e-9


def Eprune(T: pd.DataFrame, Q: dict[str, list], n_incumbent: int = 0) -> pd.DataFrame:
    """Keep a row only if it is UNIQUELY the best match for some combination.

    Ties are broken in favour of tuples already retained: the first
    `n_incumbent` rows of T are the incumbents (T from the previous round) and
    the rest are newly contributed. Removal candidates are considered
    newcomers-first, so when a newcomer merely ties an incumbent the newcomer
    is dropped and the incumbent becomes the sole witness.

    Consequence: a source that improves no match has nothing retained, so T
    is left untouched — neither AEv nor CEv can move — and the pruned table
    is independent of the order in which sources are visited.
    """
    if T is None or T.empty:
        return T
    attrs = list(Q.keys())
    if not attrs:
        return T

    W = T.reset_index(drop=True)
    n = len(W)
    combos = list(itertools.product(*[Q[a] for a in attrs]))
    if not combos:
        return W

    present = [a for a in attrs if a in W.columns]
    A = W[present].to_numpy(dtype=object)
    pos = [attrs.index(a) for a in present]

    best_sets = [set() for _ in combos]
    row_to_combos = [set() for _ in range(n)]
    for j, comb in enumerate(combos):
        target = np.asarray([comb[p] for p in pos], dtype=object)
        matches = (A == target).sum(axis=1)
        winners = np.flatnonzero(matches == matches.max())
        best_sets[j] = set(map(int, winners))
        for i in winners:
            row_to_combos[int(i)].add(j)

    # newcomers first, then incumbents — this IS the tie-break
    order = list(range(n_incumbent, n)) + list(range(n_incumbent))

    alive = np.ones(n, dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        for j in row_to_combos[i]:
            if len(best_sets[j]) == 1:      # sole witness — must keep
                break
        else:
            alive[i] = False
            for j in row_to_combos[i]:
                best_sets[j].discard(i)
    return W.loc[alive].reset_index(drop=True)


def CEprune(T: pd.DataFrame, Q: dict[str, list]) -> pd.DataFrame:
    """Greedy minimal cover of requested values, preferring rows that cover the
    FEWEST new requested values and carry the fewest already-seen values."""
    if T is None or T.empty:
        return T
    attrs = list(Q.keys())
    if not attrs:
        return T

    W = T.reset_index(drop=True)
    req_pairs = {(a, v) for a, vs in Q.items() for v in vs}

    def requested_of(i):
        return {(a, W.at[i, a]) for a in attrs
                if a in W.columns and (a, W.at[i, a]) in req_pairs}

    def values_of(i):
        return {(a, W.at[i, a]) for a in W.columns
                if not str(a).startswith("_") and not pd.isna(W.at[i, a])}

    row_req = {i: requested_of(i) for i in W.index}
    row_val = {i: values_of(i) for i in W.index}

    uncovered = {p for p in req_pairs if any(p in row_req[i] for i in W.index)}
    seen: set = set()
    selected: list[int] = []
    remaining = set(W.index)

    while uncovered and remaining:
        cands = [i for i in remaining if row_req[i] & uncovered]
        if not cands:
            break
        fewest = min(len(row_req[i] & uncovered) for i in cands)
        tier = [i for i in cands if len(row_req[i] & uncovered) == fewest]
        pick = min(tier, key=lambda i: len(row_val[i] & seen))
        selected.append(pick)
        uncovered -= row_req[pick]
        seen |= row_val[pick]
        remaining.discard(pick)

    return W.loc[selected].reset_index(drop=True)


def full_match(T: pd.DataFrame, Q: dict[str, list]) -> pd.Series:
    """Rows equal to some requested combination on EVERY attribute. Since the
    combinations are a Cartesian product, that is: every value is requested."""
    m = pd.Series(True, index=T.index)
    for a, vals in Q.items():
        m &= T[a].isin(vals) if a in T.columns else False
    return m


def evidence_pool(T: pd.DataFrame, Q: dict[str, list], k: int,
                  n_incumbent: int = 0) -> pd.DataFrame:
    """Eprune's witnesses, then extra rows that fully match a requested
    combination until there are k rows. Extras are taken in turn across
    combinations (one per combination, then a second each, ...), incumbents
    first, so k rows show k different pieces of evidence rather than k copies
    of the easiest one. Adding rows can never lower AEv."""
    W = Eprune(T, Q, n_incumbent=n_incumbent)
    need = k - (0 if W is None else len(W))
    if need <= 0 or T is None or T.empty:
        return W
    X = T.reset_index(drop=True)
    have = set(W["row_id"]) if W is not None and len(W) else set()
    extra = X.loc[full_match(X, Q) & ~X["row_id"].isin(have)]
    if extra.empty:
        return W
    turn = extra.groupby(list(Q.keys()), sort=False).cumcount()
    extra = (extra.assign(_turn=turn.to_numpy())
                  .sort_values("_turn", kind="stable")
                  .drop(columns="_turn").head(need))
    return pd.concat([W, extra], ignore_index=True)


def pool_of(state: RunState, T: pd.DataFrame, n_incumbent: int = 0) -> pd.DataFrame:
    """What the search keeps: Eprune's witnesses, or more of them when the
    answer must show at least k rows; under DIVERSE, CEprune's cover (Eprune
    would drop exactly the rows that carry unrequested values)."""
    if state.semantics == COUNTER_EVIDENCE:
        return CEprune(T, state.Q.all)
    if state.options.answer == "theta+min_k" and state.semantics != COUNTER_EVIDENCE:
        return evidence_pool(T, state.Q.all, state.options.k, n_incumbent)
    return Eprune(T, state.Q.all, n_incumbent=n_incumbent)


def prune(state: RunState, T: pd.DataFrame | None = None,
          n_incumbent: int = 0) -> pd.DataFrame:
    t0 = time.perf_counter()
    T = state.T if T is None else T
    Q_all = state.Q.all

    if state.semantics == COUNTER_EVIDENCE:
        before = vev(Q_all, T) if CHECK else None
        out = CEprune(T, Q_all)
        if CHECK and before is not None and len(T) > 0:
            assert abs(vev(Q_all, out) - before) < TOL, "CEprune changed VEv"
    else:
        before = aev(Q_all, T) if CHECK else None
        out = pool_of(state, T, n_incumbent=n_incumbent)
        if CHECK and before is not None and len(T) > 0:
            assert abs(aev(Q_all, out) - before) < TOL, "Eprune changed AEv"

    state.t_local += time.perf_counter() - t0
    return out
