"""Expected gain of a source, and the incremental selection that uses it.

    r̂_S(c) = max( r_T(c), j_S(c) + 1 − (1 − P̂(c.a_p | s_1..s_d))^{m_c} )
    δ̂_AEv(S) = (1/|C_Q|) Σ_c r̂_S(c)/k  −  AEv(Q,T)

j_S(c) and m_c come from G_S; m_c counts TUPLES, not groups, which is what
separates a source holding one candidate row from one holding five hundred.
The existence lift 1−(1−P̂)^{m_c} says AEv rewards the best tuple, not a
typical one: with many candidates, near-certainly some of them carries the
requested label.

Ties are resolved by bootstrapping the cached sample rather than propagating
intervals analytically through the product. When the leader's interval clears
the runner-up's the decision is made; otherwise only the contenders are
sampled further, up to the cap, after which the tie is accepted.
"""

from __future__ import annotations

import time

import numpy as np

from boxes.sample import (
    BATCH, INIT_SAMPLE, MAX_SAMPLE, Sample, draw, label_space, p_conditional,
)
from boxes.summarize import Summary, j_S, m_c, summarize
from core.measures import aev, cev, combinations, r_T_all, vev
from core.query import COUNTER_EVIDENCE
from core.stats import VEvTuples, expected_match, p_atleast, p_S
from core.state import RunState

N_BOOT = 12        # enough spread for a tie-break; 40 was pure cost


def _delta_from_sample(state: RunState, summary: Summary, smp: Sample,
                       combos, rT, cur_aev) -> float:
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    k = len(cols)
    static_cols = list(state.Q.static.keys())
    pred_attrs = list(state.Q.predicted.keys())

    total = 0.0
    for c in combos:
        level = j_S(summary, c, cols)
        count = m_c(summary, c, cols, level)
        conditions = {col: v for col, v in zip(cols, c) if col in static_cols}

        # Expected BEST match over the m_c candidate tuples. Each predicted
        # attribute matches with its own estimated probability, so the number a
        # single tuple matches is Poisson-binomial; lifting each tail to "some
        # tuple among m_c achieves it" gives
        #     E[best] = j + Σ_r P(Rmax >= j + r).
        # Summing the per-attribute lifts INDEPENDENTLY instead (the obvious
        # reading of "summing over their match levels") overestimates badly
        # once |A_p| > 1: it credits a source for matching genre in one tuple
        # and rating in another, when AEv requires ONE tuple to match both.
        ps = [p_conditional(smp, a_p, c[cols.index(a_p)],
                            label_space(state, a_p), conditions)
              for a_p in pred_attrs]
        gain = float(level)
        if count > 0 and ps:
            atleast = p_atleast(ps)
            for r in range(1, len(ps) + 1):
                pr = atleast[r]
                if pr > 0.0:
                    gain += 1.0 - (1.0 - pr) ** max(1, count)
        total += max(rT[c], min(k, gain)) / k

    return (total / len(combos) if combos else 0.0) - cur_aev


def uses_counts(state: RunState) -> bool:
    """Whether the STORED part of a source is scored from exact per-source
    counts (STVD) or from precomputed statistics (TVD). By default each mode
    keeps its published behaviour: PTVD counts, the static selector reads
    statistics."""
    ss = state.options.static_scoring
    if ss == "default":
        return state.mode == "PTVD"
    return ss == "counts"


def _delta_from_stats(state: RunState, src: str, smp: Sample, combos, rT, cur,
                      optimistic: bool = False) -> float:
    """TVD + predicted: TVD's own estimator, where a predicted attribute — which
    has no histogram, so TVD gives it probability 0 — gets its probability from
    the cheap sample instead. With no predicted attribute this IS TVD.
    optimistic=True sets every predicted probability to 1, which bounds the
    estimate from above (the estimator is monotone in each probability)."""
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    static_cols = set(state.Q.static.keys())
    size = state.stats["sizes"].get(src, 0)
    total = 0.0
    for c in combos:
        cond = {col: v for col, v in zip(cols, c) if col in static_cols}
        ps = [p_S(state.stats, src, col, v) if col in static_cols
              else (1.0 if optimistic else
                    p_conditional(smp, col, v, label_space(state, col), cond))
              for col, v in zip(cols, c)]
        total += expected_match(ps, size, rT[c])
    return (total / len(combos) if combos else 0.0) - cur


def _score_from_sample(state: RunState, src: str, summary, smp: Sample,
                       combos, rT, cur) -> float:
    """What ranks a source: its expected gain, from counts or statistics."""
    return (_delta_from_sample(state, summary, smp, combos, rT, cur)
            if summary is not None else
            _delta_from_stats(state, src, smp, combos, rT, cur))


def source_score(state: RunState, src_name: str, summary: Summary) -> float:
    t0 = time.perf_counter()
    combos = combinations(state.Q.all)
    rT = r_T_all(state.T, state.Q.all)
    cur = aev(state.Q.all, state.T) if len(state.T) else 0.0
    smp = state.samples.get(src_name, Sample())
    out = _score_from_sample(state, src_name, summary, smp, combos, rT, cur)
    state.t_local += time.perf_counter() - t0
    return out


def _bootstrap_spread(state: RunState, src_name: str, summary: Summary) -> float:
    """Half-width of the bootstrap interval on δ̂, from resampling the sample."""
    smp = state.samples.get(src_name)
    if smp is None or len(smp) < 4:
        return float("inf")                # too little evidence to order anything

    t0 = time.perf_counter()
    combos = combinations(state.Q.all)
    rT = r_T_all(state.T, state.Q.all)
    cur = aev(state.Q.all, state.T) if len(state.T) else 0.0
    rng = np.random.default_rng(state.rng.randrange(1 << 30))

    vals = []
    n = len(smp)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals.append(_score_from_sample(
            state, src_name, summary,
            Sample(rows=smp.rows.iloc[idx].reset_index(drop=True)),
            combos, rT, cur))
    state.t_local += time.perf_counter() - t0
    return float(np.std(vals))


def ceiling_delta(state: RunState, summary: Summary) -> float:
    """Optimistic gain from G_S alone: assume every predicted attribute matches.

    This upper-bounds δ̂ and costs no inference, so it can order candidates
    before any of them is labelled — and prove that a candidate is not worth
    labelling at all.
    """
    t0 = time.perf_counter()
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    k = len(cols)
    n_pred = len(state.Q.predicted)
    rT = r_T_all(state.T, Q_all)
    cur = aev(Q_all, state.T) if len(state.T) else 0.0

    combos = combinations(Q_all)
    total = sum(max(rT[c], min(k, j_S(summary, c, cols) + n_pred)) / k
                for c in combos)
    out = (total / len(combos) if combos else 0.0) - cur
    state.t_local += time.perf_counter() - t0
    return out


def _richest_in_evidence(state: RunState, cands, summaries) -> str | None:
    """The candidate holding the most rows whose stored values are all
    requested — exact from the counts, or expected from the statistics."""
    def n_full(s):
        g = summaries.get(s)
        if g is not None:
            return sum(n for vals, n in g.groups.items()
                       if all(v in state.Q.static.get(col, ())
                              for col, v in zip(g.cols, vals)
                              if col in state.Q.static))
        size = state.stats["sizes"].get(s, 0)
        frac = 1.0
        for a, vals in state.Q.static.items():
            frac *= sum(p_S(state.stats, s, a, v) for v in vals)
        return size * frac
    best = max(cands, key=n_full, default=None)
    return best if best is not None and n_full(best) > 0 else None


def _expected_labels(state: RunState, summary: Summary, smp: Sample, combos, rT) -> float:
    """Expensive labels this source should need to deliver its gain: for each
    combination it can still improve, about 1/p rows until one carries the
    requested predicted values, and never more than the rows it holds."""
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    total = 0.0
    for c in combos:
        level = j_S(summary, c, cols)
        count = m_c(summary, c, cols, level)
        if count <= 0 or level + len(state.Q.predicted) <= rT[c]:
            continue
        conditions = {a: v for a, v in zip(cols, c) if a in state.Q.static}
        p = 1.0
        for a in state.Q.predicted:
            p *= p_conditional(smp, a, c[cols.index(a)], label_space(state, a), conditions)
        total += min(count, 1.0 / max(p, 1e-6))
    return max(total, 1.0)


def _select_by_density(state: RunState, cands: list[str]) -> str | None:
    """Associative selection: the contacted source with the most expected gain
    per expected expensive label. Gain alone favours big sources, since in a
    big source some row will almost surely match whatever the sample says."""
    Q_all = state.Q.all
    combos = combinations(Q_all)
    rT = r_T_all(state.T, Q_all)
    cur = aev(Q_all, state.T) if len(state.T) else 0.0
    summaries = {s: summarize(state, s) for s in cands}
    state.note("ceilings", ceilings={s: round(ceiling_delta(state, summaries[s]), 4)
                                     for s in cands})
    scored = []
    for s in cands:
        smp = draw(state, s, INIT_SAMPLE)
        gain = _delta_from_sample(state, summaries[s], smp, combos, rT, cur)
        if gain > 1e-12:
            scored.append((gain / _expected_labels(state, summaries[s], smp, combos, rT),
                           gain, s))
    if not scored:
        if state.needs_evidence():
            src = _richest_in_evidence(state, cands, summaries)
            if src is not None:
                state.note("decided", source=src,
                           why="no gain left; most rows that can fully match")
                return src
        state.note("no_improvement", among=list(cands))
        return None
    scored.sort(reverse=True)
    state.note("measured", deltas=[(s, round(g, 4)) for _, g, s in scored],
               sample_rows=state.sample_rows)
    state.note("decided", source=scored[0][2], why="most gain per expected label")
    return scored[0][2]


def _select_by_gap(state: RunState, use_sample: bool = True) -> str | None:
    """DIVERSE selection: go where the target's gap can be closed. First the
    sources that show requested values T lacks (TWR), then those that add
    unrequested values (DWR), the smallest first. Histograms are local, so every
    remaining source is ranked; the cheap sample is drawn only when a requested
    PREDICTED value is still missing."""
    th = state.thresholds
    Q, T, stored = state.Q_full.all, state.T, state.Q_full.static
    k = max(1, len(Q))
    shown = {a: (set(T[a].dropna().unique()) if len(T) and a in T.columns else set())
             for a in Q}
    need_twr = (vev(Q, T) if len(T) else 0.0) < th.vev
    need_dwr = (cev(Q, T) if len(T) else 0.0) < th.cev_min
    missing = [(a, v) for a, req in state.Q_full.predicted.items() for v in req
               if v not in shown[a]]
    scored = []
    for s in state.remaining:
        hist = state.stats["hist"].get(s, {})
        d_twr = d_dwr = 0.0
        for a, req in stored.items():
            req = set(req)
            after = shown[a] | {v for v, p in hist.get(a, {}).items() if p > 0}
            d_twr += (len(after & req) - len(shown[a] & req)) / len(req)
            before = len(shown[a] - req) / len(shown[a]) if shown[a] else 0.0
            d_dwr += max(0.0, (len(after - req) / len(after) if after else 0.0) - before)
        scored.append((round(d_twr / k, 6) if need_twr else 0.0,
                       round(d_dwr / k, 6) if need_dwr else 0.0,
                       -VEvTuples(state.stats, s, state.Qp.static), s))
    if not scored:
        return None
    scored.sort(reverse=True)
    top = scored[0]
    if top[0] <= 0 and top[1] <= 0 and not missing:
        state.note("no_improvement", among=[])
        return None
    if not missing or not use_sample:
        state.note("decided", source=top[3], why="closes the target's largest gap")
        return top[3]
    ties = [x for x in scored if x[:2] == top[:2]][:5]
    best, best_p = ties[0][3], -1.0
    for x in ties:
        smp = draw(state, x[3], INIT_SAMPLE)
        p = sum(p_conditional(smp, a, v, label_space(state, a), {}) for a, v in missing)
        if p > best_p:
            best, best_p = x[3], p
    state.note("decided", source=best, why="most likely to show the missing predicted value")
    return best


def select_by_sampling(state: RunState, candidates=None) -> str | None:
    """PTVD selection: rank on free evidence, then buy labels only where they
    could still change the answer.

    Sources are ordered by their G_S ceiling — an upper bound on what any of
    them could gain. Walking down that order, once a candidate's ceiling
    cannot beat the best gain already measured, neither it nor anything after
    it can win, so the walk stops and those sources are never sampled. Only
    then, if the leaders remain within each other's bootstrap intervals, are
    their samples extended.
    """
    cands = list(candidates) if candidates is not None else list(state.remaining)
    if not cands:
        return None

    if state.targets_diverse_gap:
        # The stored statistics already say which table closes the gap: over the
        # benchmark, sampling under DIVERSE cost requests and saved no labels.
        return _select_by_gap(state, use_sample=False)
    if (state.options.source_score == "density" and uses_counts(state)
            and state.semantics != COUNTER_EVIDENCE):
        return _select_by_density(state, cands)

    if uses_counts(state):
        summaries = {s: summarize(state, s) for s in cands}
        ceilings = {s: ceiling_delta(state, summaries[s]) for s in cands}
    else:
        # TVD + predicted: no count requests; the bound is TVD's estimate with
        # every predicted value assumed to come back as requested.
        summaries = {s: None for s in cands}
        combos = combinations(state.Q.all)
        rT = r_T_all(state.T, state.Q.all)
        cur = aev(state.Q.all, state.T) if len(state.T) else 0.0
        ceilings = {s: _delta_from_stats(state, s, Sample(), combos, rT, cur,
                                         optimistic=True) for s in cands}
    order = sorted(cands, key=lambda s: ceilings[s], reverse=True)
    state.note("ceilings", ceilings={s: round(ceilings[s], 4) for s in order})

    measured: list[tuple[float, str]] = []
    best = 0.0
    for i, s in enumerate(order):
        if ceilings[s] <= max(best, 1e-12):
            # Nothing from here on can win, so it is never labelled at all.
            state.note("unsampled", sources=order[i:],
                       reason="free G_S ceiling cannot beat the best measured gain",
                       best_measured=round(best, 4))
            break
        draw(state, s, INIT_SAMPLE)
        d = source_score(state, s, summaries[s])
        measured.append((d, s))
        best = max(best, d)

    if not measured:
        if state.needs_evidence():
            src = _richest_in_evidence(state, cands, summaries)
            if src is not None:
                state.note("decided", source=src,
                           why="no gain left; most rows that can fully match")
                return src
        state.note("no_improvement", among=list(cands))
        return None                  # no source can improve
    measured.sort(reverse=True)
    state.note("measured", deltas=[(s, round(d, 4)) for d, s in measured],
               sample_rows=state.sample_rows)
    if len(measured) == 1:
        state.note("decided", source=measured[0][1], why="only contender")
        return measured[0][1]

    # Only the contenders get more evidence, and only while it could flip them.
    size = INIT_SAMPLE
    while size < MAX_SAMPLE:
        (d0, s0), (d1, s1) = measured[0], measured[1]
        e0 = _bootstrap_spread(state, s0, summaries[s0])
        e1 = _bootstrap_spread(state, s1, summaries[s1])
        state.note("race", leader=s0, challenger=s1,
                   delta=[round(d0, 4), round(d1, 4)],
                   half_width=[round(e0, 4), round(e1, 4)],
                   separated=bool(d0 - d1 > e0 + e1), sample_size=size)
        if d0 - d1 > e0 + e1:
            state.note("decided", source=s0, why="intervals separated")
            return s0
        size += BATCH
        # Stop only when neither contender has rows left to draw. Checked on
        # `sample_rows`, which `draw` increments whichever model labels — an
        # earlier version checked `model_invocations`, which sampling with the
        # cheap tier never moves, so the race always ended after one extension
        # and decided without looking at the samples it had just bought.
        before = state.sample_rows
        draw(state, s0, size)
        draw(state, s1, size)
        if state.sample_rows == before:
            state.note("decided", source=s0,
                       why="both contenders exhausted; tie accepted")
            return s0                                # nothing left to draw
        measured = sorted(
            [(source_score(state, s, summaries[s]), s) for _, s in measured],
            reverse=True)
    state.note("decided", source=measured[0][1], why="sample cap reached; tie accepted")
    return measured[0][1]
