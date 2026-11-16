"""The single entry point the interface calls.

Everything the UI can do goes through `execute`. It owns exactly three
responsibilities the pipeline should not know about: resetting the model
counters between runs, routing a static-only query to TVD, and packaging the
run into something a page can render. No search logic lives here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import pandas as pd

from boxes import target_met
from core.cost import Prices
from core.options import Options
from core.query import COUNTER_EVIDENCE, EVIDENCE, Query, Thresholds
from demo.catalog import VARIANTS, Lake, needs_cheap, options_for
from pipeline import run

TVD_VARIANT = "Static selection (histogram-only)"
# Variants that are meaningful without any predicted condition.
STATIC_SAFE = {TVD_VARIANT, "PTVD — no cheap sampling", "Random source order",
               "Proxy ranking (whole lake)",
               "Exhaustive (predict everything)"}


@dataclass
class RunSpec:
    variant: str
    static: dict[str, list]
    predicted: dict[str, list]
    semantics: str = EVIDENCE
    theta_aev: float = 0.8
    eps: float = 0.05                    # DIVERSE: give up on a plateau
    cev_max: float = 0.3                 # no longer part of the stopping rule
    theta_vev: float = 1.0
    cev_min: float = 0.40
    stats_coverage: float = 1.0
    seed: int = 42
    regime: str = "Full statistics"      # label only; stats_coverage is the knob
    options: Options = field(default_factory=Options)
    prices: Prices = field(default_factory=Prices)
    trace: bool = True

    def thresholds(self) -> Thresholds:
        return Thresholds(aev=self.theta_aev, eps=self.eps, cev_max=self.cev_max,
                          vev=self.theta_vev, cev_min=self.cev_min)

    def target_text(self) -> str:
        if self.semantics == COUNTER_EVIDENCE:
            return (f"TWR ≥ {self.theta_vev:g} and "
                    f"DWR ≥ {self.cev_min:g}, giving up once a table raises neither by "
                    f"more than ε = {self.eps:g}")
        return f"AWR ≥ {self.theta_aev:g}"


@dataclass
class RunRecord:
    spec: RunSpec
    query: Query
    variant_run: str                 # may differ from spec.variant (TVD routing)
    routing_note: str | None
    T: pd.DataFrame
    summary: dict
    rounds: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    met: bool = False
    seconds: float = 0.0

    @property
    def engine(self) -> str:
        return "TVD" if not self.query.predicted else "PTVD"

    @property
    def shown_as(self) -> str:
        """The name shown in tables and pickers. A variant routed to TVD (a
        query with no predicted condition) says so, so two runs never share a
        name and the run picker can tell them apart."""
        name = (f"{self.spec.variant} → TVD" if self.routing_note
                else self.variant_run)
        return name.replace("PTVD", "Witness")     # the system's name on screen

    @property
    def label(self) -> str:
        if self.spec.stats_coverage == 1.0:
            return self.shown_as
        return f"{self.shown_as} · {self.spec.regime}"

    def row(self) -> dict:
        """One line of the comparison table.

        No wall-clock column: with a local predictor, seconds measure this
        prototype rather than the cost the method targets.
        """
        s = self.summary
        shipped = s["rows_retrieved"]
        # Rows, not predictions: a query on two predicted attributes pays two
        # expensive predictions per labelled row.
        labelled = s.get("rows_labelled", 0)
        return {
            "variant": self.shown_as,
            "statistics": self.spec.regime,
            "met": self.met,
            "total cost": round(s["total_cost"], 1),
            "expensive": s["model_invocations"],
            "cheap": s["cheap_invocations"],
            # Of everything SQL shipped, what fraction did it actually pay to
            # label? This is incremental consumption made visible.
            "labelled %": (round(100 * labelled / shipped, 1)
                           if shipped else 0.0),
            # Everything SQL shipped that never reached a predictor. Derived
            # rather than read from `rows_skipped`, which counts ONLY the
            # ceiling-based skip and misses rows left over when the scan stops
            # on reaching theta -- so the counter understates the saving.
            "unlabelled": max(0, shipped - labelled),
            "skipped by ceiling": s["rows_skipped"],
            "rows shipped": shipped,
            "sources": s["sources"],
            "sql": s["sql_queries"],
            "requests": s["sql_queries"] + s["sample_requests"],
            "|T|": s["|T|"],
            "AEv": s["AEv"], "VEv": s["VEv"], "CEv": s["CEv"],
            "correct %": s.get("correct %"),
            "kept cheap": s.get("kept cheap", 0),
            "stopped": s["terminated_by"],
        }

    @property
    def overhead_seconds(self) -> float:
        """Everything but model inference: set-up, SQL, selection, pruning,
        measures. This part stays the same whatever predictor is used."""
        return max(0.0, self.seconds - self.summary["sec_inference"])


def answer_quality(T: pd.DataFrame, query: Query, truth: dict) -> dict:
    """EVALUATION ONLY: how many predicted values in the answer are correct.

    Reads the lake's ground truth after the run has finished; nothing in the
    search ever sees it. Split by who produced the value, so the cost of
    keeping confident cheap labels shows up as an error rate, not a guess.
    """
    out = {"correct %": None, "kept cheap": 0, "cheap correct %": None}
    if T is None or len(T) == 0 or not query.predicted:
        return out
    single = list(query.predicted)[0] if len(query.predicted) == 1 else None
    tot = ok = n_cheap = ok_cheap = 0
    for _, row in T.iterrows():
        t = truth.get(int(row["row_id"]), {})
        if not isinstance(t, dict):          # some lakes store the bare value
            t = {single: t} if single is not None else {}
        for a in query.predicted:
            if a not in T.columns:
                continue
            is_cheap = row.get(f"_by:{a}") == "cheap"
            n_cheap += int(is_cheap)
            if a not in t or pd.isna(row[a]):   # unknown: below min_confidence
                continue
            hit = row[a] == t[a]
            tot += 1
            ok += int(hit)
            ok_cheap += int(hit and is_cheap)
    if tot:
        out["correct %"] = round(100 * ok / tot, 1)
    out["kept cheap"] = n_cheap
    if n_cheap:
        out["cheap correct %"] = round(100 * ok_cheap / n_cheap, 1)
    return out


def execute(lake: Lake, expensive: dict, cheap: dict, spec: RunSpec) -> RunRecord:
    """Run one query and return everything needed to explain what happened."""
    Q = Query(static=dict(spec.static), predicted=dict(spec.predicted))
    if Q.is_empty():
        raise ValueError("the query has no conditions — pick at least one value")

    # Static-only routes to TVD: with Q_p = empty the pipeline provably invokes
    # no model, so a PTVD variant would only be TVD wearing a different label.
    variant, note = spec.variant, None
    if not Q.predicted and variant not in STATIC_SAFE:
        note = (f"No predicted conditions, so {variant!r} was routed to "
                f"{TVD_VARIANT!r}. With Q_p empty this is plain TVD: the "
                f"pipeline provably invokes no model at all.")
        variant = TVD_VARIANT

    kw = dict(VARIANTS[variant])
    th = spec.thresholds()
    opts = options_for(variant, spec.options)
    use_cheap = needs_cheap(variant, spec.options) and bool(Q.predicted)

    # Counters are per-run; the model objects are reused across runs.
    for m in (*expensive.values(), *cheap.values()):
        m.reset()

    t0 = time.perf_counter()
    res = run(Q, lake.sources, expensive,
              cheap_models=cheap if use_cheap else None,
              thresholds=th, semantics=spec.semantics, seed=spec.seed,
              stats_coverage=spec.stats_coverage, trace=spec.trace,
              options=opts, prices=spec.prices, **kw)
    elapsed = time.perf_counter() - t0

    summary = res.summary()
    summary.update(answer_quality(res.T, Q, lake.truth))
    summary["visited"] = list(res.visited)
    summary["sources_in_lake"] = len(lake.sources)

    return RunRecord(
        spec=spec, query=Q, variant_run=variant, routing_note=note,
        T=res.T, summary=summary, rounds=res.rounds, trace=res.trace,
        met=bool(target_met(Q.all, res.T, th, spec.semantics)),
        seconds=elapsed)


def execute_sweep(lake: Lake, expensive: dict, cheap: dict, base: RunSpec,
                  variants: list[str],
                  regimes: list[tuple[str, float]] = (("Full statistics", 1.0),),
                  progress=None) -> list[RunRecord]:
    """Run ONE query across every (variant, statistics regime) pair.

    The point of the demo is the comparison, so the query, thresholds, seed and
    both model tiers are held fixed and only the search strategy varies — which
    is the same discipline the experiments use, and the reason any difference
    is attributable to selection rather than to setup.
    """
    out: list[RunRecord] = []
    total = max(1, len(variants) * len(regimes))
    for regime_name, coverage in regimes:
        for variant in variants:
            if progress is not None:
                progress(len(out) / total, f"{variant} · {regime_name}")
            out.append(execute(lake, expensive, cheap,
                               replace(base, variant=variant,
                                       stats_coverage=coverage,
                                       regime=regime_name)))
    if progress is not None:
        progress(1.0, "done")
    return out


def comparison(records: list[RunRecord]) -> pd.DataFrame:
    """Threshold-first ranking: reaching the target first, then expensive
    predictions, the cost the demo shows. Total cost breaks ties."""
    from demo.catalog import ROLE
    df = pd.DataFrame([r.row() for r in records])
    if df.empty:
        return df
    df.insert(1, "role", [ROLE.get(r.spec.variant, "") for r in records])
    df = df.sort_values(["met", "expensive", "total cost"],
                        ascending=[False, True, True])
    return df.reset_index(drop=True)


# How benchmark queries split their conditions between stored and predicted
# attributes: (min stored, max stored, predicted conditions; None = all).
QUERY_MIXES = {
    "Balanced": (1, 3, None),
    "Mostly stored": (2, 3, 1),
    "Mostly predicted": (1, 1, None),
}


SEVERAL_SOURCES = "Needs several sources"


def benchmark(lake: Lake, expensive: dict, cheap: dict, base: RunSpec,
              variants: list[str], n_queries: int = 20,
              query_seeds: tuple = (42, 7), progress=None,
              static_only: bool = False, mix: str = "Balanced") -> pd.DataFrame:
    """Average every variant over many generated queries.

    Same query generator, same seeds and same settings for every variant, so a
    difference in the averages is a difference between variants. Every column
    but `reached θ %` is averaged over the runs that reached the target
    (threshold-first: a cheap run that never got there saved nothing).
    """
    from experiments.queries import generate_queries, multisource_queries
    full = pd.concat([s.assign(_src=n) for n, s in lake.sources.items()],
                     ignore_index=True)
    queries = []
    if mix == SEVERAL_SOURCES:
        for a in lake.predicted_attrs:
            full[a] = expensive[a].predict(full)
        queries = multisource_queries(full, lake.static_attrs, lake.predicted_attrs,
                                      base.semantics, base.thresholds(),
                                      n=n_queries * len(query_seeds),
                                      seed=query_seeds[0], max_seconds=120)
    else:
        lo, hi, n_pred = QUERY_MIXES[mix]
        if base.semantics == COUNTER_EVIDENCE:  # diversity needs 2 stored attributes
            lo, hi = max(lo, 2), max(hi, 2)
        for qs in query_seeds:
            queries += generate_queries(full, lake.truth, lake.static_attrs,
                                        lake.predicted_attrs, n=n_queries, seed=qs,
                                        min_static_attrs=lo, max_static_attrs=hi,
                                        n_predicted=n_pred)
    rows, total, done = [], max(1, len(queries) * len(variants)), 0
    for Q in queries:
        for v in variants:
            if progress is not None:
                progress(done / total, v)
            r = execute(lake, expensive, cheap,
                        replace(base, variant=v, static=Q.static,
                                predicted={} if static_only else Q.predicted,
                                trace=False))
            rows.append({**r.row(), "variant": v})
            done += 1
    if progress is not None:
        progress(1.0, "done")
    df = pd.DataFrame(rows)

    def agg(g: pd.DataFrame) -> pd.Series:
        met = g[g.met]
        out = {"queries": len(g), "reached θ %": round(100 * g.met.mean(), 1)}
        for col in ("total cost", "expensive", "cheap", "requests",
                    "rows shipped", "sources", "|T|", "kept cheap",
                    "AEv", "VEv", "CEv"):
            out[col] = (round(met[col].mean(), 3 if col in ("AEv", "VEv", "CEv") else 1)
                        if len(met) else None)
        c = pd.to_numeric(met["correct %"], errors="coerce")
        out["correct % (eval)"] = round(c.mean(), 1) if c.notna().any() else None
        return pd.Series(out)

    out = df.groupby("variant", sort=False).apply(agg, include_groups=False)
    return out.reindex([v for v in variants if v in out.index])
