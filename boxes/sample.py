"""Sampling and the conditional estimator — buying information about A_p.

G_S describes what a source stores. It says nothing about what the model will
predict there, and that is what decides whether a tuple is useful. So for the
candidate sources we buy a little information: draw a random
sample of the tuples the disjunctive query would retrieve, label them, and
estimate the rate at which each predicted value occurs.

Costs are real and counted. Sampled rows count as retrieval (a small shipment)
and sampled labels count as model invocations against the same headline
number, so PTVD only wins if the sample costs less than the inference it
saves. The sample is drawn incrementally and cached: a candidate whose
standing is already clear is not sampled further, and no row is ever labelled
twice.

Estimator (paper, "Estimating predicted values"). Conditioning on a full
combination is unreliable — few sampled tuples carry every condition at once.
We condition one static value at a time and combine:

    P̂(x | s_1..s_d)  ∝  P̂(x|S) · Π_i [ P̂(x|s_i) / P̂(x|S) ]

normalized over the label space L. Laplace smoothing keeps an unseen label at
a small positive rate rather than a hard zero, which would otherwise zero out
a gain merely because the sample was small.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.state import RunState
from core.store import store_of

ALPHA = 1.0                 # Laplace pseudo-count
INIT_SAMPLE = 20
BATCH = 20
MAX_SAMPLE = 100


@dataclass
class Sample:
    """Labelled rows drawn from one source, plus the row ids already used.

    `tables` memoises the per-label rates derived from these rows. Without it
    every call to p_conditional recomputes the same pandas group filters, and
    the estimator is called once per combination per bootstrap resample —
    thousands of times per selection round.
    """
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawn: set = field(default_factory=set)
    tables: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def invalidate(self) -> None:
        self.tables.clear()


def _retrievable(state: RunState, src_name: str) -> pd.DataFrame:
    """The tuples the current disjunctive Q' would retrieve — the population
    the estimates must describe."""
    src = state.D[src_name]
    if len(src) == 0 or not state.sql:
        return src.iloc[0:0]
    return store_of(state.D).rows(src_name, state.sql)      # the same query, in DuckDB


def draw(state: RunState, src_name: str, n_target: int) -> Sample:
    """Extend this source's sample up to `n_target` labelled rows (capped).

    Every newly drawn row is labelled by the model and counted: model
    invocations against the headline, and rows against a separate sample-rows
    counter so the shipment is visible.
    """
    t0 = time.perf_counter()
    smp = state.samples.setdefault(src_name, Sample())
    n_target = min(n_target, MAX_SAMPLE)
    if len(smp) >= n_target:
        state.t_local += time.perf_counter() - t0
        return smp

    pool = _retrievable(state, src_name)
    pool = pool.loc[~pool["row_id"].isin(smp.drawn)] if len(pool) else pool
    need = n_target - len(smp)
    if len(pool) == 0 or need <= 0:
        state.t_local += time.perf_counter() - t0
        return smp

    if state.options.sampling == "stratified":
        take = _stratified_take(state, src_name, smp, pool, n_target)
    else:
        take = pool.sample(n=min(need, len(pool)), random_state=state.rng.randrange(1 << 30))
    state.t_local += time.perf_counter() - t0

    # Sampling uses the CHEAP model when one is supplied; its labels inform
    # selection and never enter T. Falls back to the expensive model so the
    # single-model configuration still runs (and is comparable in one currency).
    labelling = state.cheap_models or state.models
    labelled = take.copy()
    t1 = time.perf_counter()
    # When an option will reuse the cheap model's scores (confidence ordering
    # or the cascade), keep them in the run's cache so no row is ever scored
    # twice. Same single counted call; the label is the raw argmax, exactly
    # what predict() returns.
    keep = state.options.needs_cheap and bool(state.cheap_models)
    # Rates from calibrated probabilities: counting the top label says nothing
    # about a value the cheap model never ranks first.
    soft = state.options.sample_rates == "probabilities" and bool(state.cheap_models)
    for attr in state.Q_full.predicted:
        if attr not in labelling:
            continue
        model = labelling[attr]
        if keep or soft:
            raw = model.predict_proba(take, calibrated=False)
            labelled[attr] = raw.idxmax(axis=1).to_numpy()
            if soft:
                cal = model.calibrator.transform(raw) if model.calibrator is not None else raw
                for x in cal.columns:
                    labelled[f"_p:{attr}:{x}"] = cal[x].to_numpy()
        if keep:
            raw = raw.set_axis(take["row_id"].to_numpy(), axis=0)
            prev = state.cheap_raw.get(attr)
            state.cheap_raw[attr] = raw if prev is None else pd.concat([prev, raw])
        else:
            labelled[attr] = model.predict(take)
    state.t_inference += time.perf_counter() - t1

    smp.rows = pd.concat([smp.rows, labelled], ignore_index=True) if len(smp) else labelled
    if state.options.sampling == "stratified":
        _reweight(state, src_name, smp)
    smp.invalidate()
    smp.drawn.update(take["row_id"].tolist())
    state.sample_rows += len(take)
    state.sample_requests += 1        # one draw = one request to the source
    return smp


def _level(state: RunState, rows: pd.DataFrame) -> pd.Series:
    """Number of requested STORED values a row carries (its stratum)."""
    lvl = pd.Series(0, index=rows.index, dtype=int)
    for a, vals in state.Q.static.items():
        if a in rows.columns:
            lvl = lvl + rows[a].isin(vals).astype(int)
    return lvl


def _allocation(sizes: dict[int, int], n: int) -> dict[int, int]:
    """How many of n draws each stratum gets: half go to the top stratum (the
    rows that can become full witnesses), the rest in proportion to size.
    Capped by stratum size; leftovers go top-down."""
    if not sizes:
        return {}
    top = max(sizes)
    alloc = {h: 0 for h in sizes}
    alloc[top] = min(sizes[top], (n + 1) // 2)
    rest = n - alloc[top]
    total = sum(sizes.values())
    for h in sorted(sizes, reverse=True):
        alloc[h] += min(sizes[h] - alloc[h], int(rest * sizes[h] / total))
    left = n - sum(alloc.values())
    for h in sorted(sizes, reverse=True):
        if left <= 0:
            break
        extra = min(left, sizes[h] - alloc[h])
        alloc[h] += extra
        left -= extra
    return alloc


def _stratified_take(state: RunState, src_name: str, smp: Sample,
                     pool: pd.DataFrame, n_target: int) -> pd.DataFrame:
    full = _retrievable(state, src_name)
    sizes = _level(state, full).value_counts().to_dict()
    alloc = _allocation({int(h): int(n) for h, n in sizes.items()},
                        min(n_target, len(full)))
    have = (smp.rows["_lvl"].value_counts().to_dict()
            if len(smp) and "_lvl" in smp.rows.columns else {})
    lvl = _level(state, pool)
    parts = []
    for h, want in alloc.items():
        need = want - have.get(h, 0)
        cand = pool.loc[lvl == h]
        if need > 0 and len(cand):
            parts.append(cand.sample(n=min(need, len(cand)),
                                     random_state=state.rng.randrange(1 << 30)))
    take = pd.concat(parts) if parts else pool.iloc[0:0]
    return take.assign(_lvl=_level(state, take))


def _reweight(state: RunState, src_name: str, smp: Sample) -> None:
    """Design weights N_h / n_h, normalised to mean 1, so every rate estimated
    from the sample describes the whole retrievable population again."""
    full = _retrievable(state, src_name)
    N = _level(state, full).value_counts().to_dict()
    if "_lvl" not in smp.rows.columns:
        smp.rows["_lvl"] = _level(state, smp.rows)
    n = smp.rows["_lvl"].value_counts().to_dict()
    w = smp.rows["_lvl"].map(lambda h: N.get(h, 0) / max(1, n.get(h, 1)))
    smp.rows["_w"] = w * (len(w) / w.sum()) if w.sum() > 0 else 1.0


# ─── estimation ──────────────────────────────────────────────────────────────

def _rate(rows: pd.DataFrame, attr: str, value, labels: list) -> float:
    """Laplace-smoothed P̂(attr = value) over `rows` (design-weighted when the
    sample was stratified)."""
    if "_w" in rows.columns and len(rows):
        w = rows["_w"].to_numpy(dtype=float)
        hits = float(w[(rows[attr] == value).to_numpy()].sum())
        n = float(w.sum())
    else:
        n = len(rows)
        hits = int((rows[attr] == value).sum()) if n else 0
    return (hits + ALPHA) / (n + ALPHA * max(1, len(labels)))


def _rate_tables(smp: Sample, attr: str, labels: tuple) -> tuple[dict, dict]:
    """(base rates, {(col,val) -> conditional rates}) for this sample, cached."""
    key = (attr, labels, len(smp.rows))
    hit = smp.tables.get(key)
    if hit is not None:
        return hit

    if f"_p:{attr}:{labels[0]}" in smp.rows.columns and "_w" not in smp.rows.columns:
        def mean_of(rows):
            return {x: float(rows[f"_p:{attr}:{x}"].mean()) for x in labels}
        cond = {}
        for c in smp.rows.columns:
            series = smp.rows[c]
            if c == attr or str(c).startswith("_") or series.dtype.kind not in "OUSb":
                continue
            for v, idx in series.groupby(series, observed=True).groups.items():
                cond[(c, v)] = mean_of(smp.rows.loc[idx])
        smp.tables[key] = (mean_of(smp.rows), cond)
        return smp.tables[key]

    base = {x: _rate(smp.rows, attr, x, list(labels)) for x in labels}
    cond: dict = {}
    if len(smp.rows):
        col_vals = {c: smp.rows[c] for c in smp.rows.columns}
        att = smp.rows[attr]
        for c, series in col_vals.items():
            if c == attr or series.dtype.kind not in "OUSb":
                continue
            weights = (smp.rows["_w"] if "_w" in smp.rows.columns else None)
            for v, sub_idx in series.groupby(series, observed=True).groups.items():
                sub = att.loc[sub_idx]
                if weights is None:
                    n = len(sub)
                    cond[(c, v)] = {
                        x: (int((sub == x).sum()) + ALPHA) / (n + ALPHA * len(labels))
                        for x in labels
                    }
                else:
                    w = weights.loc[sub_idx].to_numpy(dtype=float)
                    n = float(w.sum())
                    cond[(c, v)] = {
                        x: (float(w[(sub == x).to_numpy()].sum()) + ALPHA)
                           / (n + ALPHA * len(labels))
                        for x in labels
                    }
    smp.tables[key] = (base, cond)
    return base, cond


def p_conditional(smp: Sample, attr: str, value, labels: list,
                  conditions: dict[str, object]) -> float:
    """P̂(x | s_1..s_d) by the one-condition-at-a-time combination.

    Each factor shifts the source-wide rate by the effect of a single static
    condition, and is estimated from the sampled tuples satisfying that
    condition — far more numerous than those satisfying all of them.
    """
    if len(smp) == 0:
        return 1.0 / max(1, len(labels))

    base, cond = _rate_tables(smp, attr, tuple(labels))
    score = dict(base)

    for col, v in conditions.items():
        table = cond.get((col, v))
        if table is None:
            continue                                  # condition unseen: no shift
        for x in labels:
            score[x] *= table[x] / base[x] if base[x] > 0 else 1.0

    total = sum(score.values())
    if total <= 0:
        return 1.0 / max(1, len(labels))
    return score[value] / total


def label_space(state: RunState, attr: str) -> list:
    """The label space L for a predicted attribute: requested values plus
    whatever the model has actually produced in samples so far."""
    labels = list(dict.fromkeys(state.Q_full.predicted.get(attr, [])))
    m = state.models.get(attr)
    if m is not None:
        for v in m.labels:
            if v not in labels:
                labels.append(v)
    return labels or [None]
