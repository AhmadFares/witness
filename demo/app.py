"""Witness demo — Streamlit front end. One tool: ask, run, read the results.

    streamlit run demo/app.py

Layout and widgets only: every computation goes through demo.runner, and every
appearance decision through demo.style, so nothing here can change what the
system does.

One query is run by PTVD and by each selected competitor and ablation. They
share the query, thresholds, seed and models, so only the method differs.
"""

from __future__ import annotations

import gc
import hashlib
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from core.cost import Prices
from boxes.check_target import target_met
from core.measures import aev, cev, r_T_all, vev
from core.options import ROW_ORDERS, Options
from core.query import COUNTER_EVIDENCE, EVIDENCE
from demo import explore, scenarios, style
from demo.catalog import (
    ABLATIONS, COMPETITORS, COST_UNITS, DATASETS, NO_CHUNKS, PROXY, SEMANTICS, TVD_STYLE, VARIANT_HELP, label_space, load_lake, load_or_build_tiers,
    attach_saved_predictions, source_axes, source_label, split_axes, split_options,
)
from demo.runner import RunSpec, comparison, execute

st.set_page_config(page_title="Witness demo", layout="wide", page_icon="◱",
                   initial_sidebar_state="expanded")
st.markdown(style.CSS, unsafe_allow_html=True)

EXPENSIVE_KIND, CHEAP_KIND = "gb", "stump"
EXHAUSTIVE = "Exhaustive (predict everything)"
SAVED_RUNS = Path(__file__).resolve().parent / "saved_runs"
SAVED_BENCHMARK = Path(__file__).resolve().parent / "saved_benchmark.csv"
PROBLEM_HELP = ["One row per requested combination, until AWR reaches θ",
                "Every requested value, next to other values"]
NOT_FOUND = "not found in the tables: needs to be predicted"
VIEWS = ["Comparison", "Where the predictions went", "How each run went", "Vary the target"]


# Only the selected dataset and its models stay in memory (max_entries=1), for
# hosting. Switching back reloads it; the models come from demo/saved_models.
@st.cache_resource(show_spinner="Loading the dataset…", max_entries=1)
def get_lake(name: str, split: str | None = None):
    return load_lake(name, split)


@st.cache_resource
def _in_memory() -> dict:
    """Which dataset this server holds, shared by every session."""
    return {"name": None, "split": None}


def free_other_datasets(name: str, split: str | None = None) -> None:
    """Drop the dataset in memory BEFORE loading another. max_entries alone
    evicts only after the new one is built, so both would sit in memory at
    once. The models do not depend on the split, so they stay."""
    held = _in_memory()
    if held["name"] not in (None, name):
        get_lake.clear()
        get_tiers.clear()
        gc.collect()
    elif held["split"] != split and held["name"] is not None:
        get_lake.clear()
        gc.collect()
    held["name"], held["split"] = name, split


@st.cache_resource(show_spinner="Loading the predictors (training them if "
                                "no saved copy matches)…", max_entries=1)
def get_tiers(name: str, expensive_kind: str, cheap_kind: str, _lake):
    tiers = load_or_build_tiers(_lake, expensive_kind, cheap_kind)
    attach_saved_predictions(name, expensive_kind, cheap_kind, tiers[0])
    return tiers


def render_answer(rec, lake, dataset: str, coverage: bool = True) -> None:
    """The answer itself: each tuple, its values on the query attributes, and
    the source it came from."""
    w = explore.witness_table(rec.T, rec.query, rec.summary["visited"],
                              lake.row_source)
    if w.empty:
        st.info("The answer is empty — no tuple was kept.")
        return
    w["from table"] = w["from source"].map(lambda s: source_label(dataset, s))
    q = rec.query
    cols = (["tuple"]
            + [a + (explore.PRED_SUFFIX if a in q.predicted else "") for a in q.all]
            + ["from table"] + [c for c in ("predicted by",) if c in w.columns])   # provenance first
    st.dataframe(style.style_witness(w[cols], q), width="stretch",
                 hide_index=True, height=min(420, 38 + 35 * len(w)))
    st.caption("Teal: a requested value. Amber: a value that was not requested. "
               "Unsure: the expensive model's confidence was below the limit, so no "
               "value is used.")
    if coverage:
        st.markdown(explore.coverage_line(explore.value_coverage(rec.T, q, w)))


@st.cache_resource
def _run_store() -> dict:
    """Runs already made, shared by every visitor. The lock is there because
    the models count their calls: two runs at once would mix their counts."""
    return {"lock": threading.Lock(), "runs": {}}


def run_once(lake, expensive, cheap, spec: RunSpec):
    """`execute`, remembering the result: the same run is not made twice."""
    store = _run_store()
    key = (lake.name, lake.split, repr(spec))
    with store["lock"]:
        if key not in store["runs"]:
            if len(store["runs"]) >= 100:
                store["runs"].clear()
            store["runs"][key] = _execute_or_load(lake, expensive, cheap, spec, key)
        return store["runs"][key]


def _execute_or_load(lake, expensive, cheap, spec: RunSpec, key: tuple):
    """Slow runs are kept on disk and survive a restart of the server. Exhaustive
    does not depend on how Witness searches; any other run is tied to the
    current search code, so a change to the algorithm cannot show stale results."""
    exhaustive = spec.variant == EXHAUSTIVE
    name = repr(key) if exhaustive else repr((key, _code_version()))
    path = SAVED_RUNS / (hashlib.sha1(name.encode()).hexdigest()[:20] + ".joblib")
    if path.exists():
        return joblib.load(path)
    record = execute(lake, expensive, cheap, spec)
    if exhaustive or record.seconds >= 5:
        SAVED_RUNS.mkdir(exist_ok=True)
        joblib.dump(record, path)
    return record


@st.cache_resource
def _code_version() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("boxes/*.py"), *root.glob("core/*.py"), root / "pipeline.py"])
    return hashlib.sha1(b"".join(f.read_bytes() for f in files)).hexdigest()[:10]


@st.cache_resource(show_spinner="Preparing five small tables…", max_entries=16)
def get_small(_lake, _expensive, key: str, _claim):
    return scenarios.small_instance(_lake, _expensive, _claim)


def row_order_for(choice: str, semantics: str, dataset: str) -> str:
    """Ranking rows with the cheap model inside a table saved expensive calls on
    the Associative benchmark of ACSIncome and cost more on the Diverse one."""
    if choice != "auto":
        return choice
    return "static+cheap" if semantics == EVIDENCE and dataset == "acs-income" else "static"


def sql_run(claim, inst, stored: bool, q: dict, expensive, cheap):
    """Witness on the five small tables, for the query as it stands now."""
    return run_once(
        inst.stored if stored else inst.predicted, expensive, cheap,
        RunSpec(variant="PTVD",
                static={**claim.static, **(claim.predicted if stored else {})},
                predicted={} if stored else claim.predicted,
                semantics=q["semantics"], theta_aev=q["theta_aev"], theta_vev=q["theta_vev"],
                cev_min=q["cev_min"], eps=q["eps"],
                options=Options(min_confidence=scenarios.CONFIDENCE, row_order=q["row_order"])))


def sql_explain(claim, stored: bool, diverse: bool, one_per_case: bool, sql: dict, rec) -> str:
    """What this query and these controls show, with their own numbers."""
    s = rec.summary
    n, m = s["|T|"], s["sources"]
    tables = f"{m} table" + ("s" if m != 1 else "")
    rows = sql["rows"]
    paid = (f" after {sql['predictions']} expensive predictions"
            if not stored and sql["predictions"] else "")
    if diverse:
        q_all = {**claim.static, **claim.predicted}
        sql_dwr, target = cev(q_all, rows), rec.spec.cev_min
        paid = (f", after {sql['predictions']} expensive predictions, one per row of the five tables"
                if not stored else "")
        text = (f"The Diverse SQL query loosens the conditions to any of them and keeps one row per "
                f"combination: {len(rows)} rows from all {sql['tables']} tables{paid}. Its size is set "
                f"by the data, not by a target. Witness shows the requested values next to other "
                f"values with {n} row{'s' if n != 1 else ''} from {tables}"
                + (f" and {s['model_invocations']} expensive predictions." if not stored else "."))
        if sql_dwr < target - 1e-9:
            text += (f" The SQL answer shows every value these five tables hold, and its DWR is "
                     f"{sql_dwr:.2f}: no answer on these tables can reach the DWR target of {target:g}.")
        return text
    if len(rows) == 0:
        return (f"SQL returns nothing{paid}: on these five tables, no row satisfies every "
                "condition at once. Our semantics is more permissive: the witness table still "
                f"shows the closest rows, and AWR ({s['AEv']:.2f}) says how close they are.")
    if one_per_case:
        cost = (f" and needs {sql['predictions']} expensive predictions, one per matching row"
                if not stored else "")
        return ("This second SQL query returns one row per requested combination, so its answer "
                f"is no longer redundant. It is still not relaxed: it scans all {sql['tables']} "
                f"tables{cost}, and it returns nothing when no row satisfies every condition. "
                f"Witness scanned {tables} and made {s['model_invocations']} expensive predictions.")
    if not stored:
        return (f"A single SQL query has to call the model on all {sql['predictions']} rows that "
                f"match the stored conditions. Witness made {s['model_invocations']} expensive "
                f"predictions in {tables}, and returns {n} row{'s' if n != 1 else ''}, each prediction "
                "shown with the model's confidence.")
    combos = int(rows.groupby(list(claim.static)).ngroups)
    ours = (f"Witness reaches the target with {n} row{'s' if n != 1 else ''} from {tables}"
            + (", enough to answer the question." if claim.text else ".")
            if rec.met else
            f"Witness returns {n} row{'s' if n != 1 else ''} from {tables}, short of the target.")
    return (f"A single SQL query scans all {sql['tables']} tables and returns {len(rows)} rows for "
            f"{combos} requested combination{'s' if combos != 1 else ''}: its answer is redundant. "
            + ours)


def _measure_tiles(values: list[tuple[str, float]], met: bool) -> None:
    """The measures of an answer and whether they reach the target, as tiles."""
    tiles = st.columns(3)
    for tile, (name, value) in zip(tiles, values):
        tile.metric(name, f"{value:g}")
    tiles[len(values)].metric("Target", "reached" if met else "not reached")


def render_sql_section(q: dict, lake, expensive, cheap, dataset: str) -> None:
    """The query above, on five tables small enough to read. Editing the query
    keeps the same five tables, so a stricter question can come back empty.
    SQL on the left and Witness on the right show the same four blocks: what
    was run, what it cost, what came back, and how good that answer is."""
    state = st.session_state
    heading, back = st.columns([6, 1.5], vertical_alignment="center")
    heading.markdown(style.step("2", "Compare with SQL"), unsafe_allow_html=True)
    if state.get("records"):
        back.button("← Back to results", on_click=scenarios.close_sql, args=(state,), width="stretch")
    static, predicted = q["static"], q["predicted"]
    target = next(iter(predicted), None)
    if dataset != "acs-income":
        st.info("The comparison with SQL is available on ACSIncome.")
        return
    if not static or target is None:
        st.info("Choose at least one stored and one predicted condition above.")
        return
    claim = scenarios.claim_from(state.get("example"), static, {target: predicted[target][:1]})
    base = state.get("sql_base")
    if base is None or base.predicted != claim.predicted:
        base = claim
    inst = get_small(lake, expensive, f"{lake.split}|{base.key()}", base)
    label = claim.label
    if inst is None:
        state["sql_base"] = None
        st.warning("The lake holds no row with these conditions that the model "
                   f"labels {label} with confidence at least {scenarios.CONFIDENCE:g}. "
                   "Try other conditions.")
        return
    state["sql_base"] = base
    opening = (f"<b>{claim.text}</b> " if claim.text else "")
    if not inst.exact:
        st.caption(f"No person in the lake meets every condition with a {label} prediction at "
                   f"confidence ≥ {scenarios.CONFIDENCE:g}, so these five tables hold the people "
                   "closest to the query.")
    st.markdown(f'<p class="lede">{opening}Your query on five tables of twenty people, '
                'small enough to check by eye. Change the conditions above and both sides '
                'update on these same tables.</p>', unsafe_allow_html=True)
    c = st.columns([1.7, 2, 1.1], vertical_alignment="center")
    stored = c[0].radio(f"{target.capitalize()} is", ["stored in the tables", NOT_FOUND],
                        key="sql_stored") == "stored in the tables"
    diverse = q["semantics"] == COUNTER_EVIDENCE
    one_per_case = c[1].checkbox("Second SQL query: one row per requested combination",
                                 key="sql_better", disabled=diverse,
                                 help="For the Associative answer. Under Diverse the SQL side "
                                      "shows the Diverse SQL query.") and not diverse

    sql = (scenarios.diverse_sql(inst, claim, stored) if diverse
           else scenarios.plain_sql(inst, claim, stored, one_per_case))
    rec = sql_run(claim, inst, stored, q, expensive, cheap)
    s = rec.summary
    name_of = lambda t: source_label(dataset, t)
    steps = explore.run_steps(rec, label_of=name_of)
    box = min(780, 60 + 23 * (max(sql["sql"].count("\n"), steps.count("\n")) + 1))
    order = [a for a in lake.static_attrs if a in claim.static]     # same columns, same order
    q_all = {**claim.static, **claim.predicted}
    th = rec.spec.thresholds()
    table_h = 250
    left, right = st.columns(2)
    with left, st.container(border=True, key="card_sql"):
        which = ("Diverse query: any condition, one row per combination" if diverse else
                 "second query, one row per requested combination" if one_per_case else "single query")
        st.markdown(f'<div class="sub">SQL<span>{which}</span></div>', unsafe_allow_html=True)
        st.code(sql["sql"], language="sql", wrap_lines=True, height=box)
        m = st.columns(3)
        m[0].metric("Tables scanned", sql["tables"])
        m[1].metric("Rows returned", len(sql["rows"]))
        m[2].metric("Expensive predictions", sql["predictions"])
        shown = sql["rows"].copy()
        shown["from table"] = shown["table"].map(name_of)
        shown[claim.target] = shown[claim.target].astype(object).where(shown[claim.target].notna(), explore.UNSURE)
        plain = SimpleNamespace(all=q_all, predicted={})    # same colours as the witness table
        shown = shown.reset_index(drop=True)
        repeat = shown.duplicated(subset=list(q_all))    # a combination already shown above
        table = style.style_witness(shown[[*order, claim.target, "from table"]], plain).apply(
            lambda r: [style.REPEAT if repeat[r.name] else ""] * len(r), axis=1)
        st.dataframe(table, hide_index=True, width="stretch", height=table_h)
        n_rep = int(repeat.sum())
        st.caption(("Every row returned satisfies at least one condition. " if diverse else
                    "Every row returned satisfies every condition. ")
                   + (f"Grey: {n_rep} row{'s' if n_rep != 1 else ''} repeating a combination "
                      "already shown, the redundancy." if n_rep else
                      "" if diverse or not len(shown) else "No row repeats a combination."))
        got = sql["rows"]
        _measure_tiles([(style.TWR, round(vev(q_all, got), 4)), (style.DWR, round(cev(q_all, got), 4))]
                       if diverse else [(style.AWR, round(aev(q_all, got), 4))],
                       target_met(q_all, got, th, q["semantics"]))
    with right, st.container(border=True, key="card_witness"):
        kind = "Diverse" if diverse else "Associative"
        st.markdown(f'<div class="sub">Witness<span>{kind} witness table</span></div>',
                    unsafe_allow_html=True)
        st.code(steps, language="yaml", wrap_lines=True, height=box)
        m = st.columns(3)
        m[0].metric("Tables scanned", s["sources"])
        m[1].metric("Rows returned", s["|T|"])
        m[2].metric("Expensive predictions", s["model_invocations"])
        w = explore.witness_table(rec.T, rec.query, s["visited"], rec_lake_sources(inst, stored))
        if w.empty:
            st.info("No row of these tables is close enough to show.")
        else:
            w["from table"] = w["from source"].map(name_of)
            q_ = rec.query
            keep = [a + (explore.PRED_SUFFIX if a in q_.predicted else "")
                    for a in [*order, claim.target]]
            if not stored:
                conf = rec.T.reset_index(drop=True)[f"_conf:{claim.target}"]
                w["confidence"] = conf.map(lambda v: f"{v:.2f}")
                keep += ["confidence"]
            st.dataframe(style.style_witness(w[keep + ["from table"]], q_),
                         hide_index=True, width="stretch", height=table_h)
        st.caption("Teal: a requested value. Amber: a value that was not requested.")
        _measure_tiles([(style.TWR, s["VEv"]), (style.DWR, s["CEv"])] if diverse
                       else [(style.AWR, s["AEv"])], rec.met)
    st.info(sql_explain(claim, stored, diverse, one_per_case, sql, rec))
    st.markdown(f'<div class="sub">The five tables<span>tinted: {claim.who} · filled: the rows '
                'in the witness table</span></div>', unsafe_allow_html=True)
    columns = list(dict.fromkeys([*order, *inst.claim.static, *inst.claim.stricter[:1]]))[:4]
    kept = set(rec.T["row_id"]) if len(rec.T) and "row_id" in rec.T.columns else set()
    for col, (name, table) in zip(st.columns(5), inst.stored.sources.items()):
        col.markdown(f"**{name_of(name)}**")
        hit = pd.Series(True, index=table.index)
        for a, vals in claim.static.items():
            hit &= table[a].isin(vals)
        varying = [a for a in columns if table[a].nunique() > 1]    # the title says the rest
        shown = table[[*varying, claim.target]].rename(
            columns={"age_group": "age", "marital_status": "marital status"}).replace(
            {"No high school diploma": "No diploma"})
        if not stored:
            shown[claim.target] = "to predict"
        col.dataframe(style.style_small_table(shown, hit, table["row_id"].isin(kept)),
                      hide_index=True, width="stretch", height=400)
    st.caption("Real records from the lake. Predicted values are accepted only at confidence "
               f"≥ {scenarios.CONFIDENCE:g}. Unsure: the model's confidence was below the limit, so no value is "
               "used. This page shows what the answer means, not what it costs.")


def rec_lake_sources(inst, stored: bool) -> dict:
    return (inst.stored if stored else inst.predicted).row_source


def view_comparison(df, variants, prices_now: dict, x: float, y: float, z: float,
                    money: bool, diverse: bool) -> None:
    total = prices_now["column"]
    ordered = df.assign(_o=df["variant"].map(lambda v: [i for i, u in enumerate(variants)
                                                        if u.replace("PTVD", "Witness") == v][0]))
    ordered = ordered.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    ordered[total] = x * ordered["cheap"] + y * ordered["expensive"] + z * ordered["requests"]
    charts = st.columns(3, gap="medium")
    for col, (column, title, is_money) in zip(charts, (
            (total, "Total cost" + (" ($)" if money else " (s)"), money),
            ("expensive", "Expensive predictions", False),
            ("sources", "Tables scanned", False))):
        col.altair_chart(style.method_bars(ordered, column, title, is_money), width="stretch")
    cols = (["variant", "met"]
            + ["expensive", "sources", "rows shipped", "requests", "cheap"]
            + (["VEv", "CEv"] if diverse else ["AEv"]) + [total])
    st.dataframe(style.style_comparison(ordered[cols]), width="stretch", hide_index=True)
    st.caption("Expensive and cheap count predictions. Table accesses count every COUNT(*) "
               "query, sample and scan.")
    with st.expander("What each method does"):
        for name in ordered["variant"].unique():
            text = VARIANT_HELP.get(name.replace("Witness", "PTVD"), "")
            st.markdown(f"**{style.short_name(name)}** — {text.replace('PTVD', 'Witness')}")


def view_effort(records, lake, dataset: str) -> None:
    names = [r.label for r in records]
    chosen = st.multiselect("Methods", names, default=names[:3], key="effort_of",
                            format_func=style.short_name)
    x_axis, y_axis = split_axes(lake.split)
    st.caption(f"One cell per table, laid out by {x_axis} and "
               f"{y_axis}. Teal: expensive predictions spent on that table, "
               "darker for more. Dark grey: the table was scanned but got no "
               "prediction. A teal dot: it was only sampled, a count and a few rows. "
               "Light grey: it was never accessed. White: no such "
               "table, because it would have fewer than 150 rows. Hover over "
               "a cell to see the table.")
    grid = [(name, source_axes(dataset, name)) for name in lake.sources]
    spent = {r.label: explore.effort_by_source(r.trace, lake.row_source)
             for r in records if r.label in chosen}
    read_by = {r.label: set(r.summary["visited"]) for r in records}
    sampled_by = {r.label: explore.sampled_tables(r.trace) for r in records}
    vmax = max((max(d.values()) for d in spent.values() if d), default=1)
    for i, name in enumerate(chosen):
        cells = pd.DataFrame(
            [{"source": source_label(dataset, s), "col": c, "row": r,
              "predictions": int(spent[name].get(s, 0)), "read": s in read_by[name],
              "sampled": s in sampled_by[name]}
             for s, (c, r) in grid])
        n_used = int((cells.predictions > 0).sum())
        only = int((cells.sampled & ~cells.read).sum())
        st.markdown(f"**{style.short_name(name)}** · scanned **{int(cells.read.sum())} of "
                    f"{len(cells)} tables**" + (f", sampled {only} more" if only else "")
                    + f" · {int(cells.predictions.sum()):,} expensive "
                    f"predictions in {n_used} of them")
        st.altair_chart(style.effort_map(cells, x_axis, y_axis,
                                         vmax, last=i == len(chosen) - 1, first=i == 0),
                        width="stretch")


def view_plans(records, dataset: str) -> None:
    labels = [r.label for r in records]
    first = next((i for i, r in enumerate(records) if r.spec.variant == "PTVD"), 0)
    second = next((i for i in range(len(records)) if i != first), first)
    cols = st.columns(2)
    picked = [cols[0].selectbox("First run", labels, index=first, key="plan_a",
                                format_func=style.short_name),
              cols[1].selectbox("Second run", labels, index=second, key="plan_b",
                                format_func=style.short_name)]
    st.caption("Each run repeats one iteration per table until the answer reaches "
               "the target. Shaded: the rows the expensive model labels.")
    for i, name in enumerate(picked):
        if i:
            st.divider()
        plan = explore.run_plan(next(r for r in records if r.label == name))
        st.graphviz_chart(style.plan_dot(
            style.short_name(name), plan, label_of=lambda s: source_label(dataset, s)), width="stretch")


def answer_section(records, lake, dataset: str, diverse: bool) -> None:
    """What the user came for, first: the witness table, how good it is, and what
    it took. The answer of any other method is one click away."""
    st.html('<div id="witness-answer"></div>')
    st.markdown(style.step("2", "Answer"), unsafe_allow_html=True)
    with st.container(border=True, key="card_answer"):
        labels = [r.label for r in records]
        first = next((i for i, r in enumerate(records) if r.spec.variant == "PTVD"), 0)
        title, pick_col = st.columns([3, 1.2], vertical_alignment="bottom")
        pick = pick_col.selectbox("Answer of", labels, index=first, key="answer_of",
                                  format_func=style.short_name)
        rec = next(r for r in records if r.label == pick)
        s = rec.summary
        title.markdown(f'<div class="sub">{style.short_name(pick)}<span>witness table</span></div>',
                       unsafe_allow_html=True)
        measures = ([(style.TWR, s["VEv"]), (style.DWR, s["CEv"])] if diverse
                    else [(style.AWR, s["AEv"]), (style.DWR, s["CEv"])])
        n_rows, shipped = s["|T|"], s["rows_retrieved"]
        st.markdown(style.answer_strip(
            [(name, f"{value:.2f}") for name, value in measures]
            + [("Rows", f"{n_rows:,}", f"pruned from {shipped:,} scanned"),
               ("Tables scanned", f"{s['sources']:,}", f"of {len(lake.sources):,}"),
               ("Expensive predictions", f"{s['model_invocations']:,}")],
            rec.met, rec.spec.target_text()), unsafe_allow_html=True)
        if rec.routing_note:
            st.info(rec.routing_note)
        render_answer(rec, lake, dataset, coverage=diverse)
    if st.session_state.pop("scroll_to_answer", False):      # right after Run: show the answer
        import streamlit.components.v1 as components
        components.html("<script>const a = window.parent.document.getElementById('witness-answer');"
                        "if (a) a.scrollIntoView({behavior: 'smooth', block: 'start'});</script>",
                        height=0)


def view_sweep(records, lake, expensive, cheap, variants, unit: str, x: float, y: float, z: float) -> None:
    spec = records[0].spec
    if spec.semantics != EVIDENCE:
        st.info("This view is for the Associative problem: choose Associative above and "
                "press Run.")
        return
    money = unit.startswith("Cost")
    st.caption("The same query at every target θ on AWR. Total cost uses the prices above.")
    cap = st.number_input("Budget ($)" if money else "Budget (seconds)", min_value=0.0,
                          value=1.0 if money else 15000.0, step=0.5 if money else 1000.0,
                          key=f"budget_{unit}")
    cheap_proxy = style.short_name(PROXY)
    if st.button("Run every θ", type="primary"):
        bar = st.progress(0.0, text="starting…")
        methods = [v for v in variants if v not in (EXHAUSTIVE, TVD_STYLE, NO_CHUNKS)]
        thetas = [0.6, 0.7, 0.8, 0.9, 1.0]
        rows, done = [], 0
        for th in thetas:
            for v in methods:
                shown_as = style.short_name(v.replace("PTVD", "Witness"))
                bar.progress(done / (len(thetas) * len(methods)), text=f"θ = {th:g} · {shown_as}")
                r = run_once(lake, expensive, cheap, replace(spec, variant=v, theta_aev=th))
                row = r.row()
                rows.append({"θ": th, "method": shown_as,
                             "expensive": row["expensive"], "tables scanned": row["sources"],
                             "requests": row["requests"], "cheap": row["cheap"],
                             "reached θ": "yes" if r.met else "no", style.AWR: row["AEv"]})
                done += 1
        bar.empty()
        st.session_state["sweep"] = pd.DataFrame(rows)
    sweep = st.session_state.get("sweep")
    if sweep is None or "tables scanned" not in sweep.columns:
        return
    sweep = sweep.assign(total=x * sweep.cheap + y * sweep.expensive + z * sweep.requests)
    label, axis = ("$", "$.2f") if money else ("s", ",.0f")
    left, right = st.columns(2)
    left.altair_chart(style.theta_chart(sweep, "total", f"Total cost ({label})", axis, budget=cap),
                      width="stretch")
    right.altair_chart(style.theta_chart(sweep, "tables scanned", "Tables scanned", ",.0f"),
                       width="stretch")
    st.caption("One line per method, across the targets θ: a higher θ asks for more complete answers. "
               "Dashed: the budget.")
    def reached(method: str):
        ok = sweep[(sweep.method == method) & (sweep["reached θ"] == "yes") & (sweep.total <= cap)]
        return f"θ = {ok.θ.max():g}" if len(ok) else "none of these targets"
    amount = f"\\${cap:,.2f}" if money else f"{cap:,.0f} s"
    st.markdown(f"With a budget of **{amount}**, Witness reaches **{reached('Witness')}** "
                + (f"and {cheap_proxy} reaches **{reached(cheap_proxy)}**." if PROXY in variants else "."))
    total = f"total ({label})"
    view = sweep.drop(columns=["requests", "cheap"]).rename(columns={"total": total})
    st.dataframe(view.style.format({"θ": "{:g}", "expensive": "{:,.0f}", "tables scanned": "{:,.0f}",
                                    style.AWR: "{:.2f}", total: "{:,.2f}"}),
                 hide_index=True, width="stretch")


@st.fragment
def render_results(records, lake, dataset: str, expensive, cheap, variants, stale: bool) -> None:
    """A fragment: changing a view or a price redraws this part only."""
    df = comparison(records)
    diverse = records[0].spec.semantics == COUNTER_EVIDENCE
    if stale:
        st.caption("The query changed since this run. Press **Run** to update the results.")
    answer_section(records, lake, dataset, diverse)
    st.markdown(style.step("3", "How it was found"), unsafe_allow_html=True)
    with st.container(border=True, key="card_summary"):
        unit_col, kind_col, edit_col = st.columns([1.5, 2.6, 1], vertical_alignment="bottom")
        unit = unit_col.radio("Measure the cost in", list(COST_UNITS), key="cost_unit", horizontal=True)
        prices_now = COST_UNITS[unit]
        kind = kind_col.selectbox("Kind of table access", list(prices_now["z"]), key=f"kind_{unit}")
        fmt = prices_now["fmt"]
        with edit_col.popover("Edit prices", width="stretch"):
            x = st.number_input("x · one cheap prediction", min_value=0.0, value=prices_now["x"],
                                step=0.00001, format=fmt, key=f"x_{unit}")
            y = st.number_input("y · one expensive prediction", min_value=0.0, value=prices_now["y"],
                                step=0.00001, format=fmt, key=f"y_{unit}")
            z = st.number_input("z · one table access", min_value=0.0,
                                value=prices_now["z"][kind], step=0.0001, format=fmt,
                                key=f"z_{unit}_{kind}")
            st.caption(prices_now["note"])
        money = unit.startswith("Cost")
        by_name = {r.spec.variant: r for r in records}
        shown = [(n, by_name[v]) for n, v in (("Witness", "PTVD"), (style.short_name(PROXY), PROXY)) if v in by_name]
        def total_of(r):
            row = r.row()
            return x * row["cheap"] + y * row["expensive"] + z * row["requests"]
        in_unit = (lambda v: f"${v:,.2f}" if v >= 0.01 else f"${v:,.4f}") if money else (lambda v: f"{v:,.1f} s")
        def covered(r):
            """How much of the query the answer shows: combinations fully matched, or requested values."""
            q_all = r.query.all
            if diverse:
                total = sum(len(v) for v in q_all.values())
                got = sum(len(set(v) & set(r.T[a].dropna().unique()))
                          for a, v in q_all.items() if len(r.T) and a in r.T.columns)
                return f"{got} of {total}"
            best = r_T_all(r.T, q_all)
            return f"{sum(v == len(q_all) for v in best.values())} of {len(best)}"
        st.markdown(style.glance([
            ("Reached the target", [(n, "yes" if r.met else "no") for n, r in shown]),
            ("Requested values shown" if diverse else "Combinations fully matched",
             [(n, covered(r)) for n, r in shown]),
            (f"Tables scanned (of {len(lake.sources):,}) / table accesses",
             [(n, f"{r.summary['sources']:,} / {r.summary['sql_queries'] + r.summary['sample_requests']:,}")
              for n, r in shown]),
            ("Expensive predictions", [(n, f"{r.summary['model_invocations']:,}") for n, r in shown]),
            ("Total cost", [(n, in_unit(total_of(r))) for n, r in shown]),
        ]), unsafe_allow_html=True)
        def priced(v):
            text = f"{v:.9f}".rstrip("0").rstrip(".")
            return f"\\${text}" if money else f"{text} s"      # escaped: two $ would start a formula
        st.caption(f"Target: {records[0].spec.target_text()}")
        st.caption(f"Prices used: one expensive prediction {priced(y)}, one table access "
                   f"{priced(z)}, one cheap prediction {priced(x)}. Total cost adds them up.")
    if not any(r.met for r in records):
        st.info("No method reached this target. A Diverse target can be out of reach when the "
                "query has few attributes, because there is little room for contrast: try a "
                "lower diversity target, or add conditions."
                if diverse else "No method reached this target. Try a lower θ or fewer conditions.")
    if st.session_state.get("results_view") not in VIEWS:     # e.g. a view that no longer exists
        st.session_state["results_view"] = VIEWS[0]
    view = st.segmented_control("View", VIEWS, key="results_view",
                                label_visibility="collapsed") or VIEWS[0]
    with st.container(border=True, key="card_view"):
        if view == "Comparison":
            view_comparison(df, variants, prices_now, x, y, z, money, diverse)
        elif view == "Where the predictions went":
            view_effort(records, lake, dataset)
        elif view == "How each run went":
            view_plans(records, dataset)
        else:
            view_sweep(records, lake, expensive, cheap, variants, unit, x, y, z)


def ask_panel(lake, expensive, dataset: str) -> dict:
    """The query, shared by the results and by the comparison with SQL."""
    state = st.session_state
    static: dict[str, list] = {}
    predicted: dict[str, list] = {}
    with st.container(border=True, key="card_ask"):
        examples = scenarios.examples_for(dataset)
        if examples:
            options = [scenarios.OWN, *examples]
            if state.get("example") not in options:
                state["example"] = scenarios.OWN
            st.selectbox("Start from an example", options, key="example",
                         on_change=scenarios.prefill_query, args=(state,))
        wide = len(lake.predicted_attrs) > 1          # several predicted attributes need room
        stored_col, predicted_col = st.columns([2.6, 1.4] if wide else [3, 1], gap="medium")
        with stored_col:
            st.markdown('<div class="sub">Stored conditions<span>answered by the tables</span></div>',
                        unsafe_allow_html=True)
            grid = st.columns(3)
            for i, a in enumerate(lake.static_attrs):
                vals = grid[i % 3].multiselect(a.replace("_", " "), lake.static_values.get(a, []),
                                               key=f"s_{a}", placeholder="Any",
                                               on_change=scenarios.edited, args=(state,))
                if vals:
                    static[a] = vals
        with predicted_col:
            st.markdown('<div class="sub">Predicted conditions<span>by a model</span></div>',
                        unsafe_allow_html=True)
            for a in lake.predicted_attrs:
                vals = st.multiselect(a.replace("_", " "), label_space(expensive, a), key=f"p_{a}",
                                      placeholder="Any", on_change=scenarios.edited, args=(state,))
                if vals:
                    predicted[a] = vals
        st.divider()
        problem_col, *slider_cols = st.columns([1.5, 1, 1, 1], gap="large")
        semantics = SEMANTICS[problem_col.radio("Problem", list(SEMANTICS), key="problem",
                                                captions=PROBLEM_HELP, horizontal=True)]
        if semantics == COUNTER_EVIDENCE:
            theta_vev = slider_cols[0].slider("Target Witness Ratio, θ (TWR)", 0.0, 1.0,
                                              step=0.05, key="theta_vev",
                                              help="Share of the requested values the answer must show.")
            cev_min = slider_cols[1].slider("Diversity Witness Ratio, θ (DWR)", 0.0, 1.0, step=0.05,
                                            key="cev_min",
                                            help="Share of the values shown that were not requested.")
            eps = slider_cols[2].slider("Stop when a table adds less than ε", 0.0, 0.20,
                                        step=0.01, key="eps")
            theta_aev = 0.8
        else:
            theta_aev = slider_cols[0].slider("Association Witness Ratio, θ (AWR)", 0.0, 1.0, step=0.05,
                                              key="theta_aev",
                                              help="The AWR the answer must reach. At 1, every "
                                                   "requested combination is fully matched by one row.")
            theta_vev, cev_min, eps = 1.0, 0.40, 0.05
    return dict(static=static, predicted=predicted, semantics=semantics, theta_aev=theta_aev,
                theta_vev=theta_vev, cev_min=cev_min, eps=eps)


# ─── controls ────────────────────────────────────────────────────────────────

state = st.session_state
if "_ready" not in state:
    scenarios.first_visit(state)
# Assigned on every run, not only when missing: a slider that was hidden (the
# other problem was selected) is then drawn again with its own value. With
# setdefault it came back at 0 after a run made while it was hidden.
for _key, _value in scenarios.DEFAULTS.items():
    state[_key] = state.get(_key, _value)
state["split_acs-income"] = state.get("split_acs-income", scenarios.DEFAULT_SPLIT)



with st.sidebar:
    st.markdown(style.BRAND, unsafe_allow_html=True)
    st.markdown("## Data")
    dataset = st.selectbox("Dataset", list(DATASETS), key="dataset",
                           label_visibility="collapsed")
    split = st.selectbox("How the dataset is split into tables",
                         split_options(dataset), key=f"split_{dataset}")
    free_other_datasets(dataset, split)
    lake = get_lake(dataset, split)
    expensive, cheap, R, _ = get_tiers(dataset, EXPENSIVE_KIND, CHEAP_KIND, lake)
    st.caption(f"{len(lake.sources):,} tables · {lake.n_rows:,} rows · predicted: "
               + ", ".join(lake.predicted_attrs))

    st.markdown("## Compare Witness with")
    competitors = st.multiselect("Competitors", COMPETITORS, key="competitors",
                                 format_func=style.short_name)
    ablations = st.multiselect("Witness without one of its components",
                               ABLATIONS, key="ablations",
                               format_func=lambda v: style.short_name(v.replace("PTVD", "Witness")))
    variants = ["PTVD"] + competitors + ablations
    if EXHAUSTIVE in variants:
        st.caption("Exhaustive Prediction makes an expensive prediction on every matching row, so it is the slowest run.")

    with st.expander("Settings"):
        ANS = {"theta": "Reach θ",
               "theta+k": "Reach θ with at most k rows",
               "theta+min_k": "Reach θ and show at least k rows"}
        answer = st.radio("Answer size", list(ANS), format_func=ANS.get)
        k_rows = (int(st.number_input("k (rows in the answer)", min_value=1,
                                      max_value=50, value=4))
                  if answer != "theta" else None)
        ROW = {"auto": "Automatic (default)", "storage": "As stored",
               "static": "Best stored match first",
               "static+cheap": "Best stored match, then cheap-model score"}
        row_choice = st.selectbox(
            "Order of the rows to label", ["auto", *ROW_ORDERS], format_func=ROW.get,
            help="Automatic ranks the best stored matches of a table with the cheap model "
                 "on Associative queries over ACSIncome, where that saves expensive calls. "
                 "On Diverse queries it costs more, so they keep the stored order.")
        use_cascade = st.checkbox(
            "Keep confident cheap labels",
            help="A row keeps the cheap model's label when it is confident "
                 "enough, and skips the expensive prediction.")
        tau = (st.slider("Minimum confidence", 0.5, 0.99, 0.9, 0.01)
               if use_cascade else None)
        min_conf = st.slider(
            "Accept a predicted value only at confidence", 0.5, 0.95, 0.8, 0.05,
            key="min_conf",
            help="Below this confidence the expensive model's label is not used "
                 "as evidence: the value stays unknown. 0.5 accepts every label.")
        budget = st.number_input("Budget (0 = none)", min_value=0.0, value=0.0,
                                 step=50.0,
                                 help="Stops a run once its total cost reaches it.")
        price_cheap = st.number_input(
            "Price of a cheap prediction", min_value=0.0, value=round(1.0 / R, 5),
            step=0.001, format="%.5f",
            help=f"In expensive predictions. Default 1/{R:.0f}, the measured "
                 f"speed ratio of the two models.")
        seed = st.number_input("Seed", value=42, step=1)
        show_models = st.toggle("Show model accuracy")

prices = Prices(expensive=1.0, cheap=price_cheap)

# ─── 1 · ask ─────────────────────────────────────────────────────────────────

st.markdown('<p class="lede">Find a small table of evidence in a data lake where some '
            'attributes are stored and others need to be predicted by an expensive model, '
            'while accessing as few tables as possible.</p>', unsafe_allow_html=True)

if show_models:
    st.dataframe(pd.DataFrame([
        {"attribute": a, "model": role,
         "accuracy": round(m.calibration_report.get("accuracy", float("nan")), 2)}
        for a in lake.predicted_attrs
        for role, m in (("expensive", expensive[a]), ("cheap", cheap[a]))]),
        hide_index=True)

st.markdown(style.step("1", "Ask"), unsafe_allow_html=True)
q = ask_panel(lake, expensive, dataset)
static, predicted, semantics = q["static"], q["predicted"], q["semantics"]
q["row_order"] = row_order_for(row_choice, semantics, dataset)
opts = Options(row_order=q["row_order"], budget=budget or None, cascade_tau=tau,
               min_confidence=min_conf if min_conf > 0.5 else None,
               answer=answer, k=k_rows)
theta_aev, theta_vev, cev_min, eps = q["theta_aev"], q["theta_vev"], q["cev_min"], q["eps"]

with st.container(horizontal=True, vertical_alignment="center", gap="small"):   # buttons as wide as their labels
    go = st.button("Run", type="primary", disabled=not (static or predicted), width="content")
    st.button("Compare with SQL", on_click=scenarios.open_sql, args=(state,), width="content",
              disabled=dataset != "acs-income" or not (static and predicted),
              help="Runs this query on five small tables, next to the SQL that answers it.")
    if not (static or predicted):
        st.caption("Pick some conditions above, or start from an example.")
    elif static and not predicted:
        st.caption("No predicted condition: no model is needed.")

# ─── 2 · results, or the comparison with SQL ─────────────────────────────────

if state.get("records_of") != (dataset, split):
    state.pop("records", None)
    state.pop("sweep", None)
if go:
    base = RunSpec(
        variant=variants[0], static=static, predicted=predicted,
        semantics=semantics, theta_aev=theta_aev, theta_vev=theta_vev,
        cev_min=cev_min, eps=eps, seed=int(seed), options=opts, prices=prices)
    state["view"] = "results"
    bar = st.progress(0.0, text="starting…")
    live = st.empty()
    runs = []
    for i, v in enumerate(variants):
        bar.progress(i / len(variants), text=f"running  {style.short_name(v.replace('PTVD', 'Witness'))}")
        runs.append(run_once(lake, expensive, cheap, replace(base, variant=v)))
        done = comparison(runs)
        live.dataframe(done.assign(variant=done["variant"].map(style.short_name))
                       [["variant", "met", "expensive", "sources", "requests"]]
                       .rename(columns={"variant": "method", "met": "reached θ", "sources": "tables scanned",
                                        "requests": "table accesses"}),
                       hide_index=True, width="stretch")
    live.empty()
    bar.empty()
    state["records"] = runs
    state["records_of"] = (dataset, split)
    state["scroll_to_answer"] = True
    state.pop("sweep", None)

if state.get("view") == "sql":
    render_sql_section(q, lake, expensive, cheap, dataset)
elif state.get("records"):
    spec = state["records"][0].spec
    now = (static, predicted, semantics) + ((theta_aev,) if semantics == EVIDENCE else (theta_vev, cev_min, eps))
    then = (spec.static, spec.predicted, spec.semantics) + (
        (spec.theta_aev,) if spec.semantics == EVIDENCE else (spec.theta_vev, spec.cev_min, spec.eps))
    render_results(state["records"], lake, dataset, expensive, cheap, variants, now != then)


# ─── benchmark ───────────────────────────────────────────────────────────────

with st.expander("Benchmark · saved results"):
    here = (pd.read_csv(SAVED_BENCHMARK).query("dataset == @dataset")
            if SAVED_BENCHMARK.exists() else pd.DataFrame())
    if here.empty:
        st.info("The saved results for this dataset are not ready yet.")
    else:
        st.caption("Cheap Proxy and Exhaustive Prediction take a long time on this many queries, so every "
                   "method was run in advance on the same queries and the results are saved: "
                   "opening this panel computes nothing. Each query needs several tables. "
                   "Costs are averages over the queries that both Witness and Cheap Proxy reached.")
        splits = list(here.split.unique())
        split_col, problem_col = st.columns(2)
        chosen = split_col.selectbox("How the dataset is split into tables", splits, key="bench_split",
                                     index=splits.index(split) if split in splits else 0)
        problems = ["Associative", "Diverse"]
        problem = problem_col.radio("Problem", problems, horizontal=True, key="bench_problem",
                                    index=0 if semantics == EVIDENCE else 1)
        shown = here[(here.split == chosen) & (here.problem == problem)]
        if shown.empty:
            st.info("No saved results for this choice yet.")
        elif shown.averaged_on.sum() == 0:
            st.info(f"On the {int(shown.queries.max())} queries that ran, Witness and Cheap Proxy never "
                    "both reached the target, so there are no costs to average.")
        else:
            st.dataframe(style.style_benchmark(shown), hide_index=True, width="stretch")
            slow = shown[shown.method.str.startswith("Exhaustive")]
            few = (f" Exhaustive Prediction's clock comes from {int(slow.clock_runs.iloc[0])} runs."
                   if len(slow) and slow.clock_runs.iloc[0] < slow.queries.iloc[0] else "")
            st.caption("Clock is the average wall-clock time of one run, on one CPU core of a lab "
                       "server, with the expensive model really called." + few + " The expensive "
                       "model here is a fast stand-in. With an LLM at about 2.5 seconds per prediction, "
                       "the time would be mostly the expensive predictions.")
