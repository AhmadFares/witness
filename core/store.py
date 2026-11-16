"""The data lake in DuckDB.

Every table of a lake is loaded once into an in-memory DuckDB database, and
every access the search makes to a table is a SQL query against it:

    scan         SELECT * FROM S WHERE a1 IN (...) OR a2 IN (...) ...
    count query  SELECT A', COUNT(*) FROM S WHERE ... GROUP BY A'
    sample pool  the same filter as the scan, then rows are drawn from it

Rows come back in their stored order (ORDER BY the row's position), with the
column types they had, so the search sees exactly what it saw before. The free
per-table statistics (histograms) stay outside: they are the lake's catalog,
not an access to a table.
"""

from __future__ import annotations

import threading

import duckdb
import pandas as pd

_POS = "_pos"
_LOCK = threading.Lock()
_STORES: dict[int, tuple] = {}          # id(lake tables) -> (tables, Store)


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class Store:
    """One DuckDB database holding the tables of one lake."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.con = duckdb.connect(":memory:")
        self.table: dict[str, str] = {}       # lake name -> DuckDB table
        self.index: dict[str, pd.Index] = {}  # the row labels, to give rows back as stored
        self.dtypes: dict[str, pd.Series] = {}
        self.columns: dict[str, list[str]] = {}
        for i, (name, df) in enumerate(tables.items()):
            t = f"t{i}"
            frame = df.reset_index(drop=True)
            # categories are stored as text, so that any requested value can be compared
            frame = frame.assign(**{c: frame[c].astype(object) for c in frame.columns
                                    if isinstance(frame[c].dtype, pd.CategoricalDtype)})
            frame[_POS] = range(len(frame))
            self.con.register("_incoming", frame)
            self.con.execute(f"CREATE TABLE {t} AS SELECT * FROM _incoming")
            self.con.unregister("_incoming")
            self.table[name], self.index[name] = t, df.index
            self.dtypes[name], self.columns[name] = df.dtypes, list(df.columns)

    def _where(self, name: str, conditions: dict[str, list]) -> tuple[str, list]:
        """A row qualifies if it carries ANY requested value (the disjunctive query)."""
        cols = self.columns[name]
        parts, params = [], []
        for a, vals in conditions.items():
            if a in cols and vals:
                parts.append(f"{_ident(a)} IN ({', '.join('?' for _ in vals)})")
                params.extend(vals)
        return (" OR ".join(parts) if parts else "FALSE"), params

    def rows(self, name: str, conditions: dict[str, list]) -> pd.DataFrame:
        """The rows of table `name` matching at least one requested value, in
        stored order, labelled as in the lake."""
        where, params = self._where(name, conditions)
        with _LOCK:
            got = self.con.cursor().execute(
                f"SELECT * FROM {self.table[name]} WHERE {where} ORDER BY {_POS}", params).df()
        pos = got.pop(_POS).to_numpy()
        got.index = self.index[name][pos]
        for c, dt in self.dtypes[name].items():
            if got[c].dtype != dt:
                got[c] = got[c].astype(dt)
        return got[self.columns[name]]

    def counts(self, name: str, columns: list[str], conditions: dict[str, list]) -> dict:
        """{tuple of values of `columns` -> number of rows}, over the rows that
        match at least one requested value."""
        where, params = self._where(name, conditions)
        cols = ", ".join(_ident(c) for c in columns)
        with _LOCK:
            got = self.con.cursor().execute(
                f"SELECT {cols}, COUNT(*) AS n FROM {self.table[name]} WHERE {where} GROUP BY {cols}",
                params).fetchall()
        return {tuple(r[:-1]): int(r[-1]) for r in got}


def store_of(tables: dict[str, pd.DataFrame]) -> Store:
    """The DuckDB store of a lake, built on first use and kept for later runs."""
    key = id(tables)
    with _LOCK:
        hit = _STORES.get(key)
        if hit is not None and hit[0] is tables:
            return hit[1]
    store = Store(tables)
    with _LOCK:
        _STORES[key] = (tables, store)
    return store
