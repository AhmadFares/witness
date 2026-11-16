"""Explore an answer: the witness table, what each tuple witnesses, and where
it came from.

Pure pandas, no Streamlit, so the same views are usable from a notebook. Every
notion here is the one the measures use (core.measures): a tuple's match level
for a combination c is the number of query attributes on which it equals c,
r_T(c) is the best level any tuple reaches, AEv averages r_T(c)/k over the
combinations, VEv is requested-value coverage and CEv the share of present
values that were not requested. Nothing is re-defined for display.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from core.measures import combinations
from core.query import COUNTER_EVIDENCE

PRED_SUFFIX = " (predicted)"
UNSURE = "unsure"       # the model's confidence was below the limit: no value is used


def effort_by_source(trace: list[dict], row_source: dict[int, str]) -> dict[str, int]:
    """Expensive predictions spent on each source, from the run's trace."""
    ids = [e["row_ids"] for e in trace if e["event"] == "expensive"]
    if not ids:
        return {}
    return (pd.Series(np.concatenate(ids)).map(row_source)
            .value_counts().to_dict())


def _label(c: tuple) -> str:
    return " · ".join(map(str, c))


def _levels(T: pd.DataFrame, cols: list[str], c: tuple) -> pd.Series:
    """Per-tuple match level for combination c (same rule as r_T)."""
    lvl = pd.Series(0, index=T.index, dtype=int)
    for a, v in zip(cols, c):
        if a in T.columns:
            lvl = lvl + (T[a] == v).astype(int)
    return lvl


def witness_table(T: pd.DataFrame, query, visited: list[str],
                  row_source: dict[int, str]) -> pd.DataFrame:
    """The answer, one row per tuple, with its provenance and what it witnesses.

    `best witness for` lists the combinations this tuple reaches the best match
    level on — under Associative semantics, Eprune keeps a tuple only if it is
    the best witness for some combination, so this is the reason it is in T.
    `adds unrequested` lists the query attributes on which it shows a value that
    was NOT requested — the counter-evidence a Diverse answer is built from.
    """
    if T is None or len(T) == 0:
        return pd.DataFrame()
    Q = query.all
    cols = list(Q)
    k = len(cols)
    T = T.reset_index(drop=True)

    best_for = {i: [] for i in T.index}
    for c in combinations(Q):
        lvl = _levels(T, cols, c)
        top = int(lvl.max())
        if top > 0:
            for i in lvl.index[lvl == top]:
                best_for[i].append(f"{_label(c)} ({top}/{k})")

    order = {s: i + 1 for i, s in enumerate(visited)}
    src = T["row_id"].map(lambda r: row_source.get(int(r), "?"))
    unrequested = [
        ", ".join(f"{a}={row[a]}" for a in cols
                  if a in T.columns and pd.notna(row[a]) and row[a] not in Q[a])
        for _, row in T.iterrows()]

    out = pd.DataFrame({
        "tuple": [f"t{i + 1}" for i in T.index],
        "from source": src,
        "round": src.map(order),
    })
    # The query attributes right after provenance, so the requested values are
    # visible without scrolling; the long explanations follow them.
    pred = set(query.predicted)
    for a in cols:
        shown = T[a] if a in T.columns else pd.Series(pd.NA, index=T.index)
        if a in pred:
            shown = shown.astype(object).where(shown.notna(), UNSURE)
        out[a + (PRED_SUFFIX if a in pred else "")] = shown
    by_cols = [a for a in query.predicted if f"_by:{a}" in T.columns]
    if by_cols:
        out["labelled by"] = [
            ", ".join(f"{a}: {T.at[i, f'_by:{a}']} ({T.at[i, f'_conf:{a}']:.2f})"
                      for a in by_cols) for i in T.index]
    out["best witness for"] = ["; ".join(best_for[i]) or "—" for i in T.index]
    out["adds unrequested"] = [u or "—" for u in unrequested]
    rest = [c for c in T.columns
            if c not in cols and c != "row_id" and not str(c).startswith("_")]
    for c in rest:
        out[c] = T[c]
    out["row_id"] = T["row_id"]
    return out


def value_coverage(T: pd.DataFrame, query,
                   witnesses: pd.DataFrame) -> pd.DataFrame:
    """Every value that matters, per attribute: requested ones (covered or
    not — VEv) and unrequested ones present in the answer (CEv)."""
    Q = query.all
    T = (T if T is not None else pd.DataFrame()).reset_index(drop=True)
    pred = set(query.predicted)
    rows = []
    for a, req in Q.items():
        present = (set(T[a].dropna().unique())
                   if len(T) and a in T.columns else set())
        for v in list(req) + sorted(present - set(req), key=str):
            hit = T.index[T[a] == v] if len(T) and a in T.columns else []
            rows.append({
                "attribute": a + (PRED_SUFFIX if a in pred else ""),
                "value": v,
                "requested": v in req,
                "in answer": len(hit) > 0,
                "tuples": ", ".join(witnesses.loc[i, "tuple"] for i in hit) or "—",
                "sources": ", ".join(sorted({witnesses.loc[i, "from source"]
                                             for i in hit})) or "—",
            })
    return pd.DataFrame(rows)


def coverage_line(vals: pd.DataFrame) -> str:
    """The value coverage in one sentence: requested values covered (VEv),
    the ones missing, and how many unrequested values the answer shows (CEv)."""
    req = vals[vals["requested"]]
    missing = req[~req["in answer"]]
    extra = vals[~vals["requested"] & vals["in answer"]]
    line = f"Requested values shown: **{int(req['in answer'].sum())} of {len(req)}**"
    if len(missing):
        line += " · missing: " + ", ".join(
            f"{a.replace(PRED_SUFFIX, '')} = {v}"
            for a, v in zip(missing["attribute"], missing["value"]))
    if len(extra):
        per = extra.groupby("attribute", sort=False).size()
        line += (f". Values shown that were not requested: **{len(extra)}** ("
                 + ", ".join(f"{a.replace(PRED_SUFFIX, '')} {n}" for a, n in per.items())
                 + ")")
    return line + "."


# ─── the plan of a run ───────────────────────────────────────────────────────

STOPS = {
    "target_met":   "target reached",
    "plateau":      "the last table raised neither TWR nor DWR by more than ε, so the target is unlikely",
    "budget_spent": "budget spent",
    "infeasible_k": "no answer of k rows can reach the target",
    "no_source":    "no table left that can improve the answer",
    "exhausted":    "every table scanned",
}
PICKS = {"PredictEachSource": "Statistics pick", "random": "Random pick",
         "PredictEverything": "Next source", "Proxy": "Proxy ranking"}


def _candidates(events: list[dict], chosen: str) -> list[dict]:
    """PTVD's shortlist for one choice: every candidate with its best possible
    gain (ceiling), the cheap-sample estimate if it was sampled, and whether it
    was picked, sampled or never sampled. Other strategies log no shortlist."""
    ceilings = [e["ceilings"] for e in events if e["event"] == "ceilings"]
    if not ceilings:
        return []
    ceil = ceilings[-1]                  # the window the choice was made in
    est: dict[str, float] = {}
    for e in events:
        if e["event"] == "measured":
            est.update({s: float(d) for s, d in e["deltas"]})
        elif e["event"] == "race":
            est[e["leader"]], est[e["challenger"]] = (float(d) for d in e["delta"])
    return [dict(name=s, ceiling=float(c), estimate=est.get(s),
                 state="picked" if s == chosen else
                       "sampled" if s in est else "skipped")
            for s, c in ceil.items()]


def run_plan(rec) -> dict:
    """What a run did, source by source, for the plan view: how each source was
    chosen, how many rows it shipped, how many of them were labelled, and the
    answer's quality after it. Built only from the run's recorded rounds and
    trace: nothing is re-run."""
    from demo.catalog import VARIANTS
    by_round = defaultdict(list)
    for e in rec.trace:
        by_round[e["round"]].append(e)
    keys = ("model_invocations", "cheap_invocations", "sql_queries",
            "sample_requests", "rows_retrieved", "rows_labelled", "sample_rows")
    prev = dict.fromkeys(keys, 0)
    steps = []
    for rd in rec.rounds:
        ev = by_round.get(rd["round"], [])
        d = {k: rd.get(k, 0) - prev[k] for k in keys}
        prev = {k: rd.get(k, 0) for k in keys}
        steps.append(dict(
            source=rd["source"],
            candidates=_candidates(ev, rd["source"]),
            shipped=d["rows_retrieved"], labelled=d["rows_labelled"],
            expensive=d["model_invocations"], cheap=d["cheap_invocations"],
            sampled=d["sample_rows"],
            requests=d["sql_queries"] + d["sample_requests"],
            ruled_out=sum(e.get("skipped", 0) for e in ev if e["event"] in
                          ("row_skip", "scan_stop")),
            ordered=any(e["event"] == "row_order" for e in ev),
            aev=rd.get("AEv"), vev=rd.get("VEv"), cev=rd.get("CEv")))
    s = rec.summary
    return dict(steps=steps, met=rec.met, chunk=rec.spec.options.chunk,
                stop=STOPS.get(s["terminated_by"], s["terminated_by"]),
                pick=PICKS.get(VARIANTS.get(rec.variant_run, {}).get("mode"),
                               "Pick"),
                expensive=s["model_invocations"], cheap=s["cheap_invocations"],
                requests=s["sql_queries"] + s.get("sample_requests", 0),
                diverse=rec.spec.semantics == COUNTER_EVIDENCE,
                answer_rows=s["|T|"], aev=s["AEv"], vev=s["VEv"], cev=s["CEv"],
                theta_aev=rec.spec.theta_aev, theta_vev=rec.spec.theta_vev,
                cev_min=rec.spec.cev_min, gave_up=s["terminated_by"] == "plateau")


def sampled_tables(trace: list[dict]) -> set:
    """The tables Witness asked for a count and a small sample before choosing
    where to go: every candidate of every shortlist it scored."""
    out: set = set()
    for e in trace:
        if e["event"] == "ceilings":
            out.update(e["ceilings"])
    return out


def run_steps(rec, label_of=lambda s: s, shown: int = 5) -> str:
    """A run as a few lines of text, table by table: what it picked, how many
    rows it scanned and labelled, and the quality after it. The counterpart of
    the SQL text on the comparison page."""
    plan = run_plan(rec)
    diverse = plan["diverse"]
    goal = (f"TWR ≥ {plan['theta_vev']:g} and DWR ≥ {plan['cev_min']:g}" if diverse
            else f"AWR ≥ {plan['theta_aev']:g}")
    out = [f"# Witness: one table at a time, stop at {goal}"]
    steps = plan["steps"]
    for i, st in enumerate(steps[:shown]):
        out.append(f"step {i + 1}:")
        if st["candidates"]:
            tested = sum(c["state"] != "skipped" for c in st["candidates"])
            out.append(f"  sample:  {tested} tables, {st['sampled']} rows   # a count and a few rows each")
        out.append(f"  pick:    {label_of(st['source'])}")
        out.append(f"  scan:    {st['shipped']:,} rows")
        if st["expensive"]:
            out.append(f"  predict: {st['labelled']:,} rows   # expensive model, {plan['chunk']} at a time")
        quality = (f"  TWR:     {st['vev']:.2f}\n  DWR:     {st['cev']:.2f}" if diverse
                   else f"  AWR:     {st['aev']:.2f}")
        last = i == len(steps) - 1
        out.append(quality + ("   # target reached" if last and plan["met"] else ""))
    if len(steps) > shown:
        rest = steps[shown:]
        out.append(f"steps {shown + 1} to {len(steps)}:")
        out.append(f"  scan:    {len(rest)} more tables, {sum(r['shipped'] for r in rest):,} rows")
        end = rest[-1]
        out.append((f"  TWR:     {end['vev']:.2f}\n  DWR:     {end['cev']:.2f}" if diverse
                    else f"  AWR:     {end['aev']:.2f}") + ("   # target reached" if plan["met"] else ""))
    if not steps:
        out.append("# no table could improve the answer")
    n, m = plan["answer_rows"], len(steps)
    verdict = "" if plan["met"] else f"   # stopped: {plan['stop']}"
    out.append(f"answer: {n} row{'s' if n != 1 else ''} from {m} table{'s' if m != 1 else ''}{verdict}")
    return "\n".join(out)
