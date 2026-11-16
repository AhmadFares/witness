"""Ship — execute the disjunctive predicate at the chosen source.

Counts SQL queries issued and rows retrieved (the other half of cost C).
"""

from __future__ import annotations

import time

import pandas as pd

from core.state import RunState
from core.store import store_of


def ship(state: RunState, src_name: str) -> None:
    t0 = time.perf_counter()
    src = state.D[src_name]
    state.visited.append(src_name)
    state.remaining = [n for n in state.remaining if n != src_name]
    state.sql_queries += 1

    if len(src) == 0 or not state.sql:
        state.retrieved = src.iloc[0:0]
        state.t_sql += time.perf_counter() - t0
        return

    # Disjunction: keep a row if ANY requested static value matches. The query
    # runs in DuckDB, where the tables of the lake are stored.
    state.retrieved = store_of(state.D).rows(src_name, state.sql).reset_index(drop=True)
    state.rows_retrieved += len(state.retrieved)
    state.t_sql += time.perf_counter() - t0
