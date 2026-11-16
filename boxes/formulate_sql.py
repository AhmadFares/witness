"""FormulateSQL — compile Q'_s into a DISJUNCTIVE predicate (spec §2.3).

    SELECT * FROM S WHERE a1 IN (...) OR a2 IN (...) OR ...

Never conjunctive: a row matching even one requested static value is retrieved.
Rows matching no requested static value are not retrieved. Rows whose only
requested value is predicted are unreachable — expected, not a bug.
"""

from __future__ import annotations

from core.state import RunState


def formulate_sql(state: RunState) -> None:
    state.sql = {a: list(vs) for a, vs in state.Qp.static.items() if vs}


def sql_text(sql: dict, table: str = "S") -> str:
    """Human-readable form of the predicate — for logging/debugging."""
    if not sql:
        return f"SELECT * FROM {table} WHERE FALSE"
    clauses = [
        f"{a} IN ({', '.join(repr(v) for v in vs)})" for a, vs in sql.items()
    ]
    return f"SELECT * FROM {table} WHERE " + " OR ".join(clauses)
