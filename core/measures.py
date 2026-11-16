"""Evidence measures (TVD paper §2.1), plus r_T.

All operate on a table T whose rows already carry predicted values (or lack the
predicted column entirely, in which case that attribute simply never matches —
which is exactly the semantics we want for prediction-blind phases).

  C_Q  = Cartesian product of requested values over ALL attributes (static and
         predicted).  k = |A| = number of attributes in Q.

  r_T(c)      = max over t in T of #attributes of c matched by t
  AEv(Q, T)   = mean over c in C_Q of r_T(c) / k
  VEv(Q, T)   = mean over attributes of |covered requested| / |requested|
  CEv(Q, T)   = mean over attributes of |unrequested present| / |all present|
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def combinations(Q: dict[str, list]) -> list[tuple]:
    """C_Q — Cartesian product of requested values across all attributes."""
    cols = list(Q.keys())
    return list(itertools.product(*[Q[c] for c in cols]))


def r_T(T: pd.DataFrame, Q: dict[str, list], combo: tuple) -> int:
    """Max number of attributes of `combo` matched by any single row of T.

    Attributes whose column is absent from T (e.g. the predicted attribute
    before any prediction has run) can never match.
    """
    if T is None or len(T) == 0:
        return 0
    return _best_match(_columns(T, Q), len(T), Q, combo)


def _columns(T: pd.DataFrame, Q: dict[str, list]) -> dict:
    """The query attributes of T as arrays: r_T runs after every labelled chunk."""
    return {c: T[c].to_numpy(dtype=object) for c in Q if c in T.columns}


def _best_match(columns: dict, n: int, Q: dict[str, list], combo: tuple) -> int:
    matched = np.zeros(n, dtype=int)
    for c, v in zip(Q, combo):
        if c in columns:
            matched += columns[c] == v
    return int(matched.max())


def r_T_all(T: pd.DataFrame, Q: dict[str, list]) -> dict[tuple, int]:
    """r_T for every combination — one pass, used by prune/reformulate/select."""
    if T is None or len(T) == 0:
        return {c: 0 for c in combinations(Q)}
    columns = _columns(T, Q)
    return {c: _best_match(columns, len(T), Q, c) for c in combinations(Q)}


def aev(Q: dict[str, list], T: pd.DataFrame) -> float:
    if T is None or len(T) == 0:
        return 0.0
    k = len(Q)
    if k == 0:
        return 1.0
    combos = combinations(Q)
    columns = _columns(T, Q)
    return sum(_best_match(columns, len(T), Q, c) for c in combos) / (k * len(combos))


def vev(Q: dict[str, list], T: pd.DataFrame) -> float:
    if T is None or len(T) == 0:
        return 0.0
    if not Q:
        return 1.0
    per_attr = []
    for a, requested in Q.items():
        req = set(requested)
        if not req:
            per_attr.append(1.0)
            continue
        present = set(T[a].dropna().unique()) if a in T.columns else set()
        per_attr.append(len(present & req) / len(req))
    return sum(per_attr) / len(Q)


def cev(Q: dict[str, list], T: pd.DataFrame) -> float:
    if T is None or len(T) == 0:
        return 0.0
    if not Q:
        return 0.0
    per_attr = []
    for a, requested in Q.items():
        if a not in T.columns:
            continue
        present = set(T[a].dropna().unique())
        if not present:
            continue
        per_attr.append(len(present - set(requested)) / len(present))
    return sum(per_attr) / len(Q) if per_attr else 0.0


def summary(Q: dict[str, list], T: pd.DataFrame) -> dict[str, float]:
    return {"AEv": aev(Q, T), "VEv": vev(Q, T), "CEv": cev(Q, T), "|T|": len(T)}


if __name__ == "__main__":
    Q = {"illness": ["flu"], "symptom": ["cough", "fever"]}
    T = pd.DataFrame([
        {"illness": "flu", "symptom": "cough"},
        {"illness": "flu", "symptom": "fever"},
    ])
    print(summary(Q, T))                      # AEv 1.0, VEv 1.0, CEv 0.0
    T2 = pd.DataFrame([{"illness": "flu", "symptom": "rash"}])
    print(summary(Q, T2))                     # AEv .5, VEv .5, CEv .5


# ─── at most k witness rows ──────────────────────────────────────────────────

def select_k(T: pd.DataFrame, Q: dict[str, list], k: int) -> pd.DataFrame:
    """The best answer of at most k rows, chosen greedily.

    f(S) = Σ_c max_{t∈S} match(t, c) is a facility-location function —
    monotone submodular — so adding, k times, the row with the largest marginal
    gain is within (1 − 1/e) of the best k-row answer (Nemhauser et al.). The
    AEv of the result is f(S) / (k_attrs · |C|). Ties keep the earlier row,
    i.e. the incumbent.
    """
    if T is None or len(T) <= k:
        return T
    cols = list(Q.keys())
    W = T.reset_index(drop=True)
    combos = combinations(Q)
    M = np.zeros((len(W), len(combos)), dtype=float)
    for j, c in enumerate(combos):
        for a, v in zip(cols, c):
            if a in W.columns:
                M[:, j] += (W[a].to_numpy() == v)
    cur = np.zeros(len(combos))
    chosen: list[int] = []
    free = np.ones(len(W), dtype=bool)
    for _ in range(k):
        gains = np.where(free, np.maximum(M - cur, 0.0).sum(axis=1), -1.0)
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        chosen.append(i)
        free[i] = False
        cur = np.maximum(cur, M[i])
    return W.loc[chosen].reset_index(drop=True)


def aev_upper_bound_k(Q: dict[str, list], k: int) -> float:
    """Highest AEv any k-row answer could reach. A row equals one combination
    on every attribute at most, so with k < |C| at most k combinations are
    fully matched and every other one is matched on at most K − 1 attributes."""
    K, C = len(Q), len(combinations(Q))
    if K == 0 or C == 0 or k >= C:
        return 1.0
    return (k * K + (C - k) * (K - 1)) / (K * C)


def vev_upper_bound_k(Q: dict[str, list], k: int) -> float:
    """Highest VEv any k-row answer could reach: k rows show at most k values
    of each attribute."""
    if not Q:
        return 1.0
    return sum(min(1.0, k / max(1, len(v))) for v in Q.values()) / len(Q)
