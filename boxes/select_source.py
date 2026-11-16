"""Source selection — the ONLY component that differs between variants.

Interface (spec §8):   select_source(state) -> source name | None

Static selection: TVD histogram estimators restricted to the current Q' and T.
No model is ever invoked for selection; predicted attributes have no histogram
and contribute probability 0 to the Poisson-binomial — that is exactly what
makes this baseline static-blind, and what PTVD will improve on by sampling.

EVIDENCE scoring (spec §3):
    for each combination c and source S of size m:
        P(R(c) ≥ r)     Poisson-binomial over p_S(a, c.a)
        P(Rmax(c) ≥ r)  = 1 − (1 − P(R(c) ≥ r))^m
        E[match(c)]     = (1/k)·( r_T(c) + Σ_{r>r_T(c)}^{k} P(Rmax(c) ≥ r) )
    δ(S) = mean_c E[match(c)] − AEv(T)            (expected AEv GAIN)
Tie-break: max δ(S), then max CEvFree, then min VEvTuples.
Return None when no remaining source has δ(S) > 0.
"""

from __future__ import annotations

from typing import Optional

from core.measures import combinations, r_T_all
from core.query import COUNTER_EVIDENCE
from core.stats import CEvFree, VEvTuples, expected_match, p_S

EPS = 1e-12
WINDOW = 5          # how many candidates get contacted per attempt


def candidate_window(state, offset: int = 0, k: int = WINDOW) -> list[str]:
    """The top-k remaining sources by HISTOGRAM score alone.

    Histograms are held locally, so ranking every source is free; only the
    sources in this window are then contacted for their aggregate summary.
    Summarising all of them would be the AllSources pattern the design
    explicitly avoids.
    """
    ranked = sorted(state.remaining,
                    key=lambda s: CEvFree(state.stats, s, state.Qp.static),
                    reverse=True)
    return ranked[offset:offset + k]


def _delta_evidence(state, src: str, combos, rT: dict) -> float:
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    k = len(cols)
    m = state.stats["sizes"].get(src, 0)

    total_expected = 0.0
    total_current = 0.0
    for c in combos:
        ps = [p_S(state.stats, src, a, v) for a, v in zip(cols, c)]
        total_expected += expected_match(ps, m, rT[c])
        total_current += rT[c] / k
    n = len(combos)
    return (total_expected - total_current) / n if n else 0.0


def _delta_counter_evidence(state, src: str, alpha=0.7, beta=0.3) -> float:
    """α·ΔVEv(S|T) + β·CEvScore(S). Known-values-of-S come from the histogram."""
    from core.measures import vev
    Qp_static = state.Qp.static
    covered_now = 0.0
    covered_after = 0.0
    attrs = list(state.Q.all.keys())
    for a in attrs:
        requested = set(state.Q.all.get(a, []))
        if not requested:
            continue
        present = set(state.T[a].dropna().unique()) if a in state.T.columns else set()
        addable = {v for v in requested if p_S(state.stats, src, a, v) > 0}
        covered_now += len(present & requested) / len(requested)
        covered_after += len((present | addable) & requested) / len(requested)
    dvev = (covered_after - covered_now) / max(1, len(attrs))

    vt = VEvTuples(state.stats, src, Qp_static)
    cf = CEvFree(state.stats, src, Qp_static)
    cev_score = (vt - cf) / vt if vt > 0 else 0.0
    return alpha * dvev + beta * cev_score


def _delta_counts(state, src: str, combos, rT: dict) -> float:
    """Exact gain of the stored part, read off the source's counts G_S."""
    from boxes.summarize import j_S, summarize
    summary = summarize(state, src)
    cols = list(state.Q.all.keys())
    k = len(cols)
    n = len(combos)
    return (sum(max(rT[c], j_S(summary, c, cols)) / k - rT[c] / k for c in combos)
            / n) if n else 0.0


def select_source(state, candidates=None) -> Optional[str]:
    """`candidates` restricts the decision to the sources that have actually
    been contacted (the top-κ window). Defaults to everything remaining."""
    cands = list(candidates) if candidates is not None else list(state.remaining)
    if not cands:
        return None

    if state.mode == "PTVD":
        from boxes.score import select_by_sampling
        return select_by_sampling(state, cands)
    if state.targets_diverse_gap:     # the same selection, without the cheap sample
        from boxes.score import _select_by_gap
        return _select_by_gap(state, use_sample=False)
    if state.mode == "PredictEverything":
        return cands[0]           # visit everything, order irrelevant
    if state.mode == "random":
        return state.rng.choice(cands)

    if state.semantics == COUNTER_EVIDENCE:
        scored = [(s, _delta_counter_evidence(state, s)) for s in cands]
        best = max(scored, key=lambda t: t[1])
        return best[0] if best[1] > EPS else None

    # EVIDENCE
    from boxes.score import uses_counts
    combos = combinations(state.Q.all)
    rT = r_T_all(state.T, state.Q.all)
    counts = uses_counts(state)
    if counts:
        # STVD: score the stored part from exact per-source counts (one count
        # request per candidate). Still blind to predicted attributes, like TVD.
        deltas = [(s, _delta_counts(state, s, combos, rT)) for s in cands]
    else:
        deltas = [(s, _delta_evidence(state, s, combos, rT)) for s in cands]
    top = max(d for _, d in deltas)
    if top <= EPS:
        # Estimated gain is zero everywhere. That is NOT evidence the search is
        # done: histograms are blind to predicted attributes, so once the static
        # part of every combination is matched the estimator caps at k-|A_p| and
        # reports zero for every source, however much they still differ in what
        # the model would say. Stopping here makes the baseline degenerate, so
        # by default it keeps going while sources remain, ordered by the static
        # tie-breaks. `stop_on_zero_gain=True` restores the naive behaviour.
        if state.stop_on_zero_gain:
            return None
        C = list(cands)
    else:
        C = [s for s, d in deltas if d >= top - EPS]

    if counts and len(C) > 1:
        # STVD tie-break: the candidate that ships the fewest rows.
        from boxes.summarize import summarize
        n = {s: summarize(state, s).n_S for s in C}
        C = [min(C, key=lambda s: n[s])]
    if len(C) > 1:                                    # tie-break 1: max CEvFree
        scored = [(s, CEvFree(state.stats, s, state.Qp.static)) for s in C]
        best = max(v for _, v in scored)
        C = [s for s, v in scored if v >= best - EPS]
    if len(C) > 1:                                    # tie-break 2: min VEvTuples
        scored = [(s, VEvTuples(state.stats, s, state.Qp.static)) for s in C]
        low = min(v for _, v in scored)
        C = [s for s, v in scored if v <= low + EPS]
    return C[0]
