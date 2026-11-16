"""Dataset-agnostic query generation.

A query draws 1..3 static attributes with 1..3 values each, plus one requested
value for EVERY predicted attribute of the lake — so on MovieLens a query asks
about both `rating_band` and `genre`, exercising the multi-attribute path that
Adult (one binary attribute) cannot reach.

Queries are filtered to be realizable: at least one lake row must satisfy the
static part AND carry the requested predicted values. That check uses ground
truth, which is legitimate at GENERATION time — it only avoids spending the
experiment on questions nothing can answer. The pipeline never sees it.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

from core.query import Query

MAX_COMBOS = 12


def _truth_of(truth: dict, row_id, attr: str):
    rec = truth[row_id]
    return rec[attr] if isinstance(rec, dict) else rec


def _realizable(lake: pd.DataFrame, truth: dict, Q: Query) -> bool:
    mask = pd.Series(True, index=lake.index)
    for a, vals in Q.static.items():
        mask &= lake[a].isin(vals)
    if not mask.any():
        return False
    sub = lake.loc[mask, "row_id"]
    for a, vals in Q.predicted.items():
        got = sub.map(lambda r, _a=a: _truth_of(truth, r, _a))
        if not got.isin(vals).any():
            return False
        sub = sub[got.isin(vals).values]
        if sub.empty:
            return False
    return True


def _signature(Q: Query) -> tuple:
    key = lambda d: tuple(sorted((a, tuple(sorted(map(str, v)))) for a, v in d.items()))
    return (key(Q.static), key(Q.predicted))


def generate_queries(
    lake: pd.DataFrame,
    truth: dict,
    static_attrs: list[str],
    predicted_attrs: list[str],
    n: int = 10,
    seed: int = 0,
    min_static_attrs: int = 1,
    max_static_attrs: int = 3,
    n_predicted: int | None = None,
) -> list[Query]:
    """`n_predicted` predicted attributes get a condition (None = all of them)."""
    rng = random.Random(seed)
    label_pool = {
        a: sorted({str(_truth_of(truth, r, a)) for r in list(truth)[:5000]})
        for a in predicted_attrs
    }
    seen: set[tuple] = set()
    out: list[Query] = []
    attempts = 0

    while len(out) < n and attempts < 8000:
        attempts += 1
        k = rng.randint(max(1, min_static_attrs), max_static_attrs)
        attrs = rng.sample(static_attrs, min(k, len(static_attrs)))

        static, n_combos = {}, 1
        for a in attrs:
            distinct = lake[a].dropna().unique().tolist()
            nv = rng.randint(1, min(3, len(distinct)))
            static[a] = sorted(map(str, rng.sample(distinct, nv)))
            n_combos *= nv
        if n_combos > MAX_COMBOS:
            continue

        chosen = (predicted_attrs if n_predicted is None else
                  rng.sample(predicted_attrs, min(n_predicted, len(predicted_attrs))))
        predicted = {a: [rng.choice(label_pool[a])] for a in chosen}
        Q = Query(static=static, predicted=predicted)

        sig = _signature(Q)
        if sig in seen or not _realizable(lake, truth, Q):
            continue
        seen.add(sig)
        out.append(Q)
    return out


# ─── Queries that need several sources ───────────────────────────────────────

def _low_to_medium(lake: pd.DataFrame, attr: str, keep: float = 0.7) -> list[str]:
    """The attribute's values without its most frequent ones, as in the TVD
    paper's workload, and without values too rare to be queried at all."""
    freq = lake[attr].value_counts()
    freq = freq[freq >= 50].sort_values()
    return [str(v) for v in freq.index[:max(2, round(len(freq) * keep))]]


def needs_several_sources(lake: pd.DataFrame, Q: Query, semantics: str, th,
                          cache: dict | None = None) -> bool:
    """True when the whole lake can reach the target and no single source can.

    `lake` carries a `_src` column and the predicted attributes as the expensive
    model labels them, because those labels are what every method will see.
    Generation time only, like `_realizable`: the search never sees this.
    """
    from itertools import product
    from core.query import COUNTER_EVIDENCE
    Q_all = Q.all
    attrs = list(Q_all)
    cache = {} if cache is None else cache
    if "codes" not in cache:
        cache["codes"], cache["names"] = pd.factorize(lake["_src"])
    codes, names = cache["codes"], cache["names"]
    if semantics == COUNTER_EVIDENCE:
        per_attr = []
        for a, req in Q_all.items():
            req = set(req)
            if ("present", a) not in cache:
                cache[("present", a)] = {
                    src: set(vals) for src, vals in
                    lake.dropna(subset=[a]).groupby("_src", observed=True)[a].unique().items()}
            present = cache[("present", a)]
            every = set().union(*present.values())
            per_attr.append((req, present, every))
        def reached(sets):
            twr = sum(len(s & r) / len(r) for r, s in sets) / len(sets)
            dwr = sum(len(s - r) / len(s) for r, s in sets if s) / len(sets)
            return twr >= th.vev and dwr >= th.cev_min
        if not reached([(r, e) for r, _, e in per_attr]):
            return False
        return not any(reached([(r, p.get(src, set())) for r, p, _ in per_attr])
                       for src in names)
    equal = {}
    for a in attrs:
        for v in Q_all[a]:
            if (a, v) not in cache:
                cache[(a, v)] = (lake[a].to_numpy() == v).astype(np.int8)
            equal[(a, v)] = cache[(a, v)]
    best = np.zeros(len(names))
    union = 0.0
    combos = list(product(*[Q_all[a] for a in attrs]))
    for c in combos:
        level = sum(equal[(a, v)] for a, v in zip(attrs, c))
        top = np.zeros(len(names), dtype=np.int8)
        np.maximum.at(top, codes, level)
        best += top
        union += top.max()
    k = len(attrs) * len(combos)
    return union / k >= th.aev and (best / k).max() < th.aev


def multisource_queries(lake: pd.DataFrame, static_attrs: list[str],
                        predicted_attrs: list[str], semantics: str, th,
                        n: int = 40, seed: int = 0, max_combos: int = 16,
                        max_seconds: float = 300.0) -> list[Query]:
    """Deep (few attributes, many values) and Shallow (many attributes, few
    values) queries on low-to-medium-frequency values, as in the TVD paper,
    kept only if no single source can answer them."""
    rng = random.Random(seed)
    pool = {a: _low_to_medium(lake, a) for a in static_attrs}
    labels = {a: sorted(map(str, lake[a].dropna().unique())) for a in predicted_attrs}
    seen: set[tuple] = set()
    out: list[Query] = []
    cache: dict = {}
    start = time.perf_counter()
    for attempt in range(6000):
        if len(out) >= n or time.perf_counter() - start > max_seconds:
            break
        deep = attempt % 2 == 0
        k = rng.randint(2, 2) if deep else rng.randint(3, min(5, len(static_attrs)))
        attrs = rng.sample(static_attrs, min(k, len(static_attrs)))
        static, combos = {}, 1
        for a in attrs:
            nv = min(len(pool[a]), rng.randint(3, 4) if deep else rng.randint(1, 2))
            static[a] = sorted(rng.sample(pool[a], nv))
            combos *= nv
        if combos > max_combos:
            continue
        Q = Query(static=static, predicted={a: [rng.choice(labels[a])]
                                            for a in predicted_attrs})
        sig = _signature(Q)
        if sig in seen:
            continue
        seen.add(sig)
        if needs_several_sources(lake, Q, semantics, th, cache):
            out.append(Q)
    return out


def describe(Q: Query) -> str:
    parts = []
    for a, vs in Q.static.items():
        parts.append(f"{a}={vs[0]!r}" if len(vs) == 1
                     else f"{a}∈{{{', '.join(map(repr, vs))}}}")
    for a, vs in Q.predicted.items():
        parts.append(f"[pred] {a}={vs[0]!r}")
    n = 1
    for vs in Q.static.values():
        n *= len(vs)
    return f"({n} combos) " + "  ∧  ".join(parts)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from setup.movielens import PREDICTED_ATTRS, STATIC_ATTRS, build_lake

    _, sources, truth = build_lake()
    lake = pd.concat(sources.values(), ignore_index=True)
    for i, Q in enumerate(generate_queries(lake, truth, STATIC_ATTRS,
                                           PREDICTED_ATTRS, n=8, seed=42), 1):
        print(f"Q{i}: {describe(Q)}")
