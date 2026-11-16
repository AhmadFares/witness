"""Summary G_S — the cheap aggregate that precedes any inference.

Per source, one aggregate query over the static attributes appearing in Q':

    SELECT   A_s', COUNT(*)
    FROM     S
    WHERE    ⋁_{a ∈ A_s'} a ∈ values(a, Q')
    GROUP BY A_s'

It returns one group per distinct combination of static values occurring in
the tuples that match at least one requested value, with its multiplicity.
No tuples are shipped and no model is invoked.

What it buys over histograms: histograms give marginals independently
(“White is 64%”, “Female is 39%”) and cannot say whether any single tuple is
both. G_S answers co-occurrence exactly, and its counts give m_c — the number
of tuples at the best static match level — which is what separates a source
holding one candidate row from one holding five hundred.

Since Q' only shrinks, a summary computed once is reused, RESTRICTED to the
groups still matching Q'. n_S must be recomputed from the restricted groups;
carrying it over would overstate the retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from core.state import RunState
from core.store import store_of


@dataclass
class Summary:
    """G_S for one source: {tuple of static values -> count}, plus the columns
    those values are positioned by."""
    cols: list[str]                      # the A_s' the summary was grouped by
    groups: dict[tuple, int] = field(default_factory=dict)

    @property
    def n_S(self) -> int:
        return sum(self.groups.values())

    def restrict(self, sql: dict[str, list]) -> "Summary":
        """Filter to the groups still satisfying the (narrowed) disjunctive Q'.

        The cached summary may be FINER than the current query needs — under
        CounterEvidence an attribute can leave A_s' entirely. Filtering is what
        correctness requires; merging the now-redundant groups is only an
        efficiency matter, and j_S(c) / m_c compute correctly either way.
        """
        keep: dict[tuple, int] = {}
        for g, n in self.groups.items():
            for col, v in zip(self.cols, g):
                if col in sql and v in sql[col]:
                    keep[g] = n
                    break
        return Summary(cols=self.cols, groups=keep)


def summarize(state: RunState, src_name: str) -> Summary:
    """Compute (or reuse) G_S for one source. Counted as a SQL query, never as
    a model invocation."""
    t0 = time.perf_counter()
    cached = state.summaries.get(src_name)
    if cached is not None:
        out = cached.restrict(state.sql)
        state.t_sql += time.perf_counter() - t0
        return out

    src = state.D[src_name]
    cols = [a for a in state.sql if a in src.columns]
    state.sql_queries += 1

    if len(src) == 0 or not cols:
        summary = Summary(cols=cols)
    else:
        # SELECT A', COUNT(*) ... GROUP BY A', run in DuckDB
        groups = store_of(state.D).counts(src_name, cols, {a: state.sql[a] for a in cols})
        summary = Summary(cols=cols, groups=groups)

    state.summaries[src_name] = summary
    state.t_sql += time.perf_counter() - t0
    return summary


# ─── quantities read off the summary ─────────────────────────────────────────

def j_S(summary: Summary, combo: tuple, combo_cols: list[str]) -> int:
    """Best static match attainable in S for this combination:
       j_S(c) = max_{g ∈ G_S} |{a ∈ A_s' : g.a = c.a}|"""
    want = {c: v for c, v in zip(combo_cols, combo)}
    best = 0
    for g in summary.groups:
        hit = sum(1 for col, v in zip(summary.cols, g) if want.get(col) == v)
        if hit > best:
            best = hit
    return best


def m_c(summary: Summary, combo: tuple, combo_cols: list[str], level: int) -> int:
    """Number of TUPLES (not groups) in the groups attaining `level`."""
    want = {c: v for c, v in zip(combo_cols, combo)}
    total = 0
    for g, n in summary.groups.items():
        hit = sum(1 for col, v in zip(summary.cols, g) if want.get(col) == v)
        if hit == level:
            total += n
    return total
