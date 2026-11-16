"""Per-source statistics — the ONLY free information (spec §1, §3).

Precomputed once per source and stored locally; never re-queried:
    p_S(a, v)  fraction of tuples in S with value v for STATIC attribute a
    |S|        number of tuples

Static attributes only. Predicted attributes have no histogram — the selector
treats them as never matching (probability 0), which is precisely what makes
these baselines "static-blind".
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def build_stats(sources: dict[str, pd.DataFrame], static_attrs: list[str]) -> dict:
    """{source -> {attr -> {value -> fraction}}} plus sizes."""
    hist: dict[str, dict[str, dict]] = {}
    sizes: dict[str, int] = {}
    for name, df in sources.items():
        n = len(df)
        sizes[name] = n
        per_attr: dict[str, dict] = {}
        for a in static_attrs:
            if a in df.columns and n > 0:
                per_attr[a] = (df[a].value_counts() / n).to_dict()
            else:
                per_attr[a] = {}
        hist[name] = per_attr
    return {"hist": hist, "sizes": sizes, "names": list(sources.keys())}


def veil_stats(stats: dict, coverage: float = 1.0, seed: int = 0) -> dict:
    """Withhold the histograms of a random 1-coverage fraction of sources.

    Real lakes do not hand out complete statistics: a source may be newly
    registered, may refuse to publish its distribution, or may have histograms
    too stale to trust. `coverage` is the fraction of sources whose histogram
    is available; the rest keep only their row count, exactly as a system
    catalog would.

    What this does and does not disable matters for reading the demo. Histograms
    feed the FREE ranking that picks which sources get contacted at all
    (`candidate_window`, and the static selector's scores). They do not feed
    PTVD's gain estimate, which comes from the per-source summary G_S and the
    cheap sample. So veiling degrades the ordering, not PTVD's ability to score
    what it has contacted — while the static selector, which has nothing else,
    degenerates to an arbitrary order. coverage=0.0 is the pure-sampling regime.
    """
    if coverage >= 1.0:
        return stats
    names = list(stats["names"])
    k = max(0, min(len(names), int(round(len(names) * coverage))))
    kept = set(random.Random(seed).sample(names, k)) if k else set()
    out = dict(stats)
    out["hist"] = {n: (stats["hist"][n] if n in kept else {}) for n in names}
    out["veiled"] = [n for n in names if n not in kept]
    return out


def p_S(stats: dict, src: str, a: str, v) -> float:
    """p_S(a, v) — 0.0 for predicted attributes (no histogram exists)."""
    return float(stats["hist"].get(src, {}).get(a, {}).get(v, 0.0))


# ─── Poisson-binomial machinery (spec §3, EVIDENCE selection) ────────────────

def p_atleast(ps: list[float]) -> list[float]:
    """Given per-attribute match probabilities, return P[r] = P(a random tuple
    matches at least r attributes), r = 0..k. Exact Poisson-binomial via DP."""
    k = len(ps)
    exact = np.zeros(k + 1)
    exact[0] = 1.0
    for p in ps:
        nxt = np.zeros(k + 1)
        for j in range(k + 1):
            if exact[j] == 0.0:
                continue
            nxt[j] += exact[j] * (1.0 - p)
            if j + 1 <= k:
                nxt[j + 1] += exact[j] * p
        exact = nxt
    out = np.zeros(k + 2)
    for r in range(k, -1, -1):
        out[r] = out[r + 1] + exact[r]
    return list(out[: k + 1])


def expected_match(ps: list[float], m: int, r_T_c: int) -> float:
    """E[match(c)] = (1/k)·( r_T(c) + Σ_{r=r_T(c)+1}^{k} P(Rmax(c) ≥ r) )
    with P(Rmax(c) ≥ r) = 1 − (1 − P(R(c) ≥ r))^m."""
    k = len(ps)
    if k == 0:
        return 1.0
    if r_T_c >= k or m <= 0:
        return r_T_c / k
    atleast = p_atleast(ps)
    total = float(r_T_c)
    for r in range(r_T_c + 1, k + 1):
        p = atleast[r]
        if p <= 0.0:
            continue
        total += 1.0 - (1.0 - p) ** m
    return total / k


# ─── Common estimators (spec §3) ─────────────────────────────────────────────

def VEvTuples(stats: dict, src: str, Qs: dict[str, list]) -> float:
    """|S| · (1 − Π_{(a,v) ∈ Q'_s} (1 − p_S(a,v)))
    Expected #rows matching at least one requested static value — i.e. the
    expected size of the disjunctive retrieval. Lower is cheaper."""
    m = stats["sizes"].get(src, 0)
    prod = 1.0
    for a, vals in Qs.items():
        for v in vals:
            prod *= (1.0 - p_S(stats, src, a, v))
    return m * (1.0 - prod)


def CEvFree(stats: dict, src: str, Qs: dict[str, list]) -> float:
    """|S| · Π_{a ∈ A_s} P(a takes a requested value in S)
    Expected #rows matching EVERY requested static attribute. Higher is better."""
    m = stats["sizes"].get(src, 0)
    prod = 1.0
    for a, vals in Qs.items():
        p_col = sum(p_S(stats, src, a, v) for v in vals)
        if p_col <= 0.0:
            return 0.0
        prod *= min(1.0, p_col)
    return m * prod
