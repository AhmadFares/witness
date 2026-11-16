"""RunState — the shared contract between pipeline boxes, plus instrumentation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import pandas as pd

from core.measures import aev, cev, select_k, vev
from core.cost import Prices, run_cost
from core.options import Options
from core.query import COUNTER_EVIDENCE, EVIDENCE, Query, Thresholds
from core.stats import build_stats, veil_stats


@dataclass
class Result:
    T: pd.DataFrame
    visited: list[str]
    model_invocations: int          # THE headline metric (expensive model)
    cheap_invocations: int          # cheap-model calls used for selection only
    sql_queries: int
    rows_retrieved: int
    aev: float
    vev: float
    cev: float
    terminated_by: str              # target_met | plateau | no_source | budget_spent | exhausted
    sample_rows: int                # rows shipped for sampling
    rows_skipped: int               # retrieved rows never labelled (row ceiling)
    plateau_stops: int              # sources abandoned once they stopped contributing
    rounds: list[dict]              # per-round trajectory
    seconds_sql: float
    seconds_inference: float
    seconds_local: float
    mode: str
    trace: list = field(default_factory=list)   # decision log, empty unless traced
    sample_requests: int = 0        # sample draws sent to sources
    total_cost: float = 0.0         # single cost model, in expensive-prediction units
    answer_rows: int = 0            # rows in the reported answer
    rows_labelled: int = 0          # rows sent to a predictor (one row may cost
                                    # one prediction per predicted attribute)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "sources": len(self.visited),
            "model_invocations": self.model_invocations,
            "cheap_invocations": self.cheap_invocations,
            "sql_queries": self.sql_queries,
            "rows_retrieved": self.rows_retrieved,
            "rows_labelled": self.rows_labelled,
            "|T|": len(self.T),
            "AEv": round(self.aev, 4),
            "VEv": round(self.vev, 4),
            "CEv": round(self.cev, 4),
            "sample_rows": self.sample_rows,
            "rows_skipped": self.rows_skipped,
            "plateau_stops": self.plateau_stops,
            "terminated_by": self.terminated_by,
            "sec_sql": round(self.seconds_sql, 3),
            "sec_inference": round(self.seconds_inference, 3),
            "sec_local": round(self.seconds_local, 3),
            "sample_requests": self.sample_requests,
            "total_cost": round(self.total_cost, 3),
        }


@dataclass
class RunState:
    # problem
    Q: Query                                   # SEARCH query (PredictAtEnd: statics only)
    Q_full: Query                              # ORIGINAL query — ALL measurement uses this
    Qp: Query                                  # working query Q', shrinks each round
    D: dict[str, pd.DataFrame]
    models: dict                               # predicted attr -> expensive model
    stats: dict
    thresholds: Thresholds
    cheap_models: dict = field(default_factory=dict)   # sampling-only models
    semantics: str = EVIDENCE
    mode: str = "PredictEachSource"
    incremental: bool = False                  # consume selected source in chunks
    stop_on_zero_gain: bool = False            # naive static stop (degenerate)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    options: Options = field(default_factory=Options)
    prices: Prices = field(default_factory=Prices)
    # evolving
    T: pd.DataFrame = field(default_factory=pd.DataFrame)
    remaining: list[str] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)
    sql: dict = field(default_factory=dict)                       # current predicate
    summaries: dict = field(default_factory=dict)                 # source -> G_S (cached)
    samples: dict = field(default_factory=dict)                   # source -> Sample (cached)
    scan_pool: pd.DataFrame | None = None     # DIVERSE: the pool the last scan stopped on
    seen: dict = field(default_factory=dict)  # DIVERSE: stored values labelled in this source
    seen_source: str | None = None            # the source `seen` describes
    flat_rounds: int = 0                      # DIVERSE: tables in a row that added nothing
    cheap_raw: dict = field(default_factory=dict)                 # attr -> raw cheap scores by row_id
    sample_rows: int = 0                                          # rows drawn for sampling
    sample_requests: int = 0                                      # sample draws sent
    rows_skipped: int = 0                                         # rows never labelled (ceiling)
    rows_labelled: int = 0                                        # rows sent to a predictor
    plateau_stops: int = 0                                        # sources abandoned mid-scan
    retrieved: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows from last Ship
    # counters
    sql_queries: int = 0
    rows_retrieved: int = 0
    terminated_by: str = "exhausted"
    rounds: list[dict] = field(default_factory=list)
    trace: list | None = None          # None = tracing off (no cost in experiments)
    t_sql: float = 0.0
    t_inference: float = 0.0
    t_local: float = 0.0

    @property
    def model_invocations(self) -> int:
        """EXPENSIVE invocations — the headline metric."""
        return sum(m.calls for m in self.models.values())

    @property
    def cheap_invocations(self) -> int:
        return sum(m.calls for m in (self.cheap_models or {}).values())

    @property
    def cost(self) -> float:
        """Total cost so far under this run's prices (core.cost)."""
        return run_cost(self.prices, self)

    @property
    def targets_diverse_gap(self) -> bool:
        """DIVERSE as PTVD runs it: tables chosen by what the target still lacks,
        the full query kept, and no row labelled that shows nothing new."""
        chunked_statistics = self.mode == "PredictEachSource" and self.incremental
        return ((self.mode == "PTVD" or chunked_statistics)
                and self.semantics == COUNTER_EVIDENCE
                and self.options.source_score == "density")

    def answer_of(self, T: pd.DataFrame) -> pd.DataFrame:
        """What is reported from a pool of witnesses: the pool itself, or at
        most k of its rows when the answer shape asks for it. The search keeps
        the whole pool; only the answer and the stopping test use k."""
        o = self.options
        if o.answer in ("theta", "theta+min_k") or not o.k or T is None or len(T) <= o.k:
            return T
        if self.semantics == COUNTER_EVIDENCE:
            return T.head(o.k).reset_index(drop=True)   # CEprune's greedy order
        return select_k(T, self.Q.all, o.k)

    def needs_evidence(self, T: pd.DataFrame | None = None) -> bool:
        """theta+min_k: the answer still shows fewer than k rows, so the search
        keeps going after theta is reached. Associative semantics only."""
        o = self.options
        T = self.T if T is None else T
        return (o.answer == "theta+min_k" and self.semantics == EVIDENCE
                and (T is None or len(T) < o.k))

    def note(self, event: str, **fields) -> None:
        """Record one decision point. A no-op unless the caller asked to trace.

        The boxes call this at the places where a choice is made — which
        sources entered the window, which candidates
        were never sampled, why a scan stopped. It only observes; nothing in
        the pipeline reads the trace back, so turning it on cannot change a
        result.
        """
        if self.trace is not None:
            self.trace.append(
                {"round": len(self.rounds) + 1, "event": event, **fields})

    def log_round(self, source: str, t_before_prune: int) -> None:
        Q_all = self.Q_full.all          # trajectory is always against the real query
        A = self.answer_of(self.T)
        self.rounds.append({
            "round": len(self.rounds) + 1,
            "source": source,
            "model_invocations": self.model_invocations,
            "cheap_invocations": self.cheap_invocations,
            "sql_queries": self.sql_queries,
            "rows_retrieved": self.rows_retrieved,
            "rows_labelled": self.rows_labelled,
            "sample_rows": self.sample_rows,
            "sample_requests": self.sample_requests,
            "cost": round(self.cost, 4),
            "T_before_prune": t_before_prune,
            "T_after_prune": len(self.T),
            "AEv": round(aev(Q_all, A), 4),
            "VEv": round(vev(Q_all, A), 4),
            "CEv": round(cev(Q_all, A), 4),
        })


def init_state(
    Q: Query,
    D: dict[str, pd.DataFrame],
    models: dict,
    cheap_models: dict | None = None,
    thresholds: Thresholds | None = None,
    semantics: str = EVIDENCE,
    mode: str = "PredictEachSource",
    seed: int = 0,
    stats_coverage: float = 1.0,
    trace: bool = False,
    options: Options | None = None,
    prices: Prices | None = None,
) -> RunState:
    t0 = time.perf_counter()
    stats = veil_stats(build_stats(D, list(Q.static.keys())),
                       coverage=stats_coverage, seed=seed)

    # PredictAtEnd is pure TVD: it has no notion of predicted attributes, so it
    # searches the STATIC query only and predicts once, at the end. Every other
    # mode searches the full query.
    Q_search = Query(static=Q.static, predicted={}) if mode == "PredictAtEnd" else Q

    state = RunState(
        Q=Q_search, Q_full=Q, Qp=Q_search.copy(),
        D=D, models=models, cheap_models=cheap_models or {}, stats=stats,
        thresholds=thresholds or Thresholds(),
        semantics=semantics, mode=mode, rng=random.Random(seed),
        remaining=list(D.keys()),
        trace=[] if trace else None,
        options=options or Options(),
        prices=prices or Prices(),
    )
    state.t_local += time.perf_counter() - t0
    return state
