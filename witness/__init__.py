"""Witness as a Python library: load a data lake, load its models, run a query.

    import witness
    lake = witness.load_datalake("ACSIncome", split="state x education")
    M, M_c = witness.load_model(lake), witness.load_cheap_model(lake)
    T, stats = witness.query(lake, {"sex": ["Female"], "income": [">50K"]},
                             semantics="associative", target_threshold=0.8,
                             model=M, cheap_model=M_c)

A thin layer over the search in `pipeline.run`: it changes nothing in how the
search works. A query is a plain dictionary {attribute: [requested values]};
which attributes are stored and which must be predicted is read from the lake.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.measures import aev, cev, vev
from core.options import Options
from core.query import COUNTER_EVIDENCE, EVIDENCE, Query, Thresholds

__all__ = ["load_datalake", "load_model", "load_cheap_model", "query", "Stats"]

_NAMES = {"acsincome": "acs-income", "acs-income": "acs-income",
          "movielens-imdb": "movielens-25m-imdb", "movielens": "movielens-25m-imdb",
          "movielens-25m-imdb": "movielens-25m-imdb"}
_SEMANTICS = {"associative": EVIDENCE, "diverse": COUNTER_EVIDENCE}
_EXPENSIVE, _CHEAP = "gb", "stump"      # the models of the demonstration


def load_datalake(name: str, split: str | None = None):
    """A data lake of horizontally partitioned tables: "ACSIncome" or
    "MovieLens-IMDb", cut into tables by `split` (default: the dataset's own)."""
    from demo.catalog import load_lake
    key = _NAMES.get(name.lower().replace(" ", ""), name)
    return load_lake(key, split)


def _tiers(lake):
    from demo.catalog import attach_saved_predictions, load_or_build_tiers
    cache = lake.__dict__.setdefault("_witness_tiers", {})
    if "tiers" not in cache:
        expensive, cheap, _, _ = load_or_build_tiers(lake, _EXPENSIVE, _CHEAP)
        attach_saved_predictions(lake.name, _EXPENSIVE, _CHEAP, expensive)
        cache["tiers"] = (expensive, cheap)
    return cache["tiers"]


def load_model(lake) -> dict:
    """The expensive model M, one per predicted attribute."""
    return _tiers(lake)[0]


def load_cheap_model(lake) -> dict:
    """The cheap model M_c, used to sample candidate tables and rank rows."""
    return _tiers(lake)[1]


@dataclass
class Stats:
    """What a run cost and how good its answer is."""
    m_calls: int            # expensive predictions
    cheap_calls: int        # cheap predictions
    tables_scanned: int
    requests: int           # every contact with a table: counts, samples, scans
    reached: bool           # the target was met
    awr: float
    twr: float
    dwr: float
    tables: list            # the tables scanned, in order


def query(lake, q: dict, semantics: str = "associative", target_threshold: float = 0.8,
          twr_threshold: float = 1.0, dwr_threshold: float = 0.4, epsilon: float = 0.05,
          model: dict | None = None, cheap_model: dict | None = None,
          min_confidence: float = 0.8, seed: int = 42) -> tuple[pd.DataFrame, Stats]:
    """Run one query with Witness and return its witness table and its Stats.

    Associative stops once AWR >= target_threshold; Diverse once TWR >=
    twr_threshold and DWR >= dwr_threshold, giving up after tables that raise
    neither by more than epsilon. A prediction is accepted only at
    `min_confidence`; the seed fixes which rows are sampled."""
    from pipeline import run
    sem = _SEMANTICS[semantics.lower()]
    model = model if model is not None else load_model(lake)
    cheap_model = cheap_model if cheap_model is not None else load_cheap_model(lake)
    predicted_attrs = set(lake.predicted_attrs)
    Q = Query(static={a: list(v) for a, v in q.items() if a not in predicted_attrs},
              predicted={a: list(v) for a, v in q.items() if a in predicted_attrs})
    th = Thresholds(aev=target_threshold, vev=twr_threshold, cev_min=dwr_threshold, eps=epsilon)
    # as in the demonstration: ranking rows with the cheap model helps Associative
    # queries on ACSIncome; elsewhere rows keep their stored-match order
    row_order = "static+cheap" if sem == EVIDENCE and lake.name.startswith("acs") else "static"
    for m in (*model.values(), *cheap_model.values()):
        m.reset()
    res = run(Q, lake.sources, model, cheap_models=cheap_model, mode="PTVD", semantics=sem,
              thresholds=th, seed=seed,
              options=Options(min_confidence=min_confidence, row_order=row_order))
    from boxes.check_target import target_met
    Q_all = Q.all
    T = res.T.reset_index(drop=True)
    stats = Stats(m_calls=res.model_invocations, cheap_calls=res.cheap_invocations,
                  tables_scanned=len(res.visited), requests=res.sql_queries + res.sample_requests,
                  reached=bool(target_met(Q_all, T, th, sem)),
                  awr=round(aev(Q_all, T), 4), twr=round(vev(Q_all, T), 4), dwr=round(cev(Q_all, T), 4),
                  tables=list(res.visited))
    keep = [c for c in T.columns if not str(c).startswith("_")]
    return T[keep], stats
