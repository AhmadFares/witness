"""The examples of the demo, and the small instance behind "Compare with SQL".

An example only fills the query builder. Comparing with SQL runs the current
query on five tables of twenty people cut from the lake, because that page
shows what the answer means rather than what it costs.

Pure Python and pandas, no Streamlit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

import pandas as pd

from demo.catalog import DATASETS, Lake, NO_CHUNKS, PROXY, TVD_STYLE, source_label

NO_HS, HS, SOME = "No high school diploma", "High school", "Some college"
EDUCATION = [NO_HS, HS]          # "never attended college": exactly these two levels
CONFIDENCE = 0.8
YOUNG = "16-24"
DEFAULT_SPLIT = "state x education"    # the claims' values live in different tables



@dataclass(frozen=True)
class Claim:
    """Something to check: who it is about, what the evidence looks like, and
    the attribute that defines the cases (one witness per value of `focus`)."""
    text: str                   # as a reader would put it; empty for your own conditions
    who: str                    # the rows highlighted in the five tables
    static: dict                # stored conditions
    predicted: dict             # the value a counterexample has
    focus: str
    stricter: tuple = ()        # (attribute, value): no counterexample in the five tables has it

    @property
    def target(self) -> str:
        return next(iter(self.predicted))

    @property
    def label(self) -> str:
        return self.predicted[self.target][0]

    def key(self) -> str:
        return repr((self.static, self.predicted, self.focus, self.stricter))


INCOME_HIGH, INCOME_LOW = {"income": [">50K"]}, {"income": ["<=50K"]}
AGED = ("age_group", YOUNG)
OWN = "Build your own query"
EXAMPLES = {c.text: c for c in [
    Claim("There is no woman who never attended college and earns more than 50K",
          "women who never attended college",
          {"sex": ["Female"], "education": EDUCATION}, INCOME_HIGH, "education", AGED),
    Claim("There is no never-married person who earns more than 50K",
          "never-married people at these education levels",
          {"marital_status": ["Never married"], "education": [HS, SOME, "Associate's"]},
          INCOME_HIGH, "education", AGED),
    Claim("There is no one with a doctorate who earns 50K or less",
          "doctorate holders in these age groups",
          {"education": ["Doctorate"], "age_group": ["35-44", "45-54", "55-64"]},
          INCOME_LOW, "age_group",
          ("marital_status", "Widowed")),
]}
FIRST = next(iter(EXAMPLES))
FOCUS_WORD = {"education": "education level", "age_group": "age group",
              "marital_status": "marital status"}


def focus_word(attribute: str) -> str:
    return FOCUS_WORD.get(attribute, attribute.replace("_", " "))


def examples_for(dataset: str) -> dict:
    return EXAMPLES if dataset == "acs-income" else {}


DEFAULTS = {"dataset": next(iter(DATASETS)), "problem": "Associative",
            "theta_aev": 0.8, "theta_vev": 1.0, "cev_min": 0.40, "eps": 0.05,
            "competitors": [PROXY, "Random source order"],
            "ablations": [NO_CHUNKS, TVD_STYLE]}


def first_visit(state) -> None:
    """The page as it opens: an empty query, for the visitor to build."""
    for key, value in DEFAULTS.items():
        state[key] = value
    state["split_acs-income"] = DEFAULT_SPLIT
    state["example"] = OWN
    state["view"] = "results"
    state["_ready"] = True


def _clear_query(state) -> None:
    """Empty every condition. Assigned, not deleted: a deleted widget value comes
    back from the browser, and the next example would add to the old one."""
    for key in [k for k in state if str(k).startswith(("s_", "p_"))]:
        state[key] = []


def _fill_query(claim: Claim, state) -> None:
    _clear_query(state)
    for a, vals in claim.static.items():
        state[f"s_{a}"] = list(vals)
    for a, vals in claim.predicted.items():
        state[f"p_{a}"] = list(vals)


def prefill_query(state) -> None:
    """The example dropdown. Choosing "build your own" starts from an empty query."""
    state["sql_base"] = None
    claim = EXAMPLES.get(state.get("example"))
    if claim is None:
        _clear_query(state)
        return
    _fill_query(claim, state)


def edited(state) -> None:
    """A condition was changed by hand: it is no longer the example."""
    state["example"] = OWN


def open_sql(state) -> None:
    state["sql_base"] = None
    state["view"] = "sql"


def close_sql(state) -> None:
    state["view"] = "results"


def new_tables(state) -> None:
    state["sql_base"] = None


def claim_from(example: str, static: dict, predicted: dict) -> Claim:
    """The example when its conditions are untouched, else the user's own."""
    known = EXAMPLES.get(example)
    same = (known is not None and predicted == known.predicted
            and set(static) == set(known.static)
            and all(set(static[a]) == set(known.static[a]) for a in static))
    if same:
        return known
    focus = max(static, key=lambda a: len(static[a])) if static else ""
    return Claim("", "rows that match your conditions", static, predicted, focus)


# ─── Scenario 1: the small instance ──────────────────────────────────────────

@dataclass
class SmallInstance:
    predicted: Lake         # the attribute is not stored
    stored: Lake            # the attribute is an ordinary column
    confidence: pd.Series   # row_id -> the expensive model's P(claim.label)
    claim: Claim
    exact: bool = True      # False: no confident counterexample exists, closest people shown


def _take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed)


def small_instance(lake: Lake, expensive: dict, claim: Claim | None = None,
                   size: int = 20, seed: int = 7) -> SmallInstance | None:
    """Five real tables of `size` real people. They are chosen, not random:
    each requested case has a true counterexample that the model finds with
    confidence at least CONFIDENCE, and the model is right on every row, so
    that the scenario shows the semantics and nothing else. When the lake holds
    no such counterexample, the five tables holding the people closest to the
    query are used instead: SQL then returns nothing, which is the point."""
    claim = claim or EXAMPLES[FIRST]
    attr, label, focus = claim.target, claim.label, claim.focus
    model = expensive[attr]
    every = pd.concat([s.assign(_src=n) for n, s in lake.sources.items()],
                      ignore_index=True)
    matches = pd.Series(True, index=every.index)
    for a, vals in claim.static.items():
        matches &= every[a].isin(vals)
    candidates = every[matches].copy()
    if candidates.empty:
        return _closest_instance(lake, expensive, claim, size, seed)
    candidates["_p"] = model.predict_proba(candidates)[label].to_numpy()
    candidates[attr] = candidates.row_id.map(lake.truth)
    good = candidates[(candidates._p >= CONFIDENCE) & (candidates[attr] == label)]
    if claim.stricter:
        good = good[good[claim.stricter[0]] != claim.stricter[1]]
    values = [v for v in claim.static[focus] if (good[focus] == v).any()]
    if not values:
        model.reset()
        return _closest_instance(lake, expensive, claim, size, seed)

    keys, anchor = [], None
    for value in values[:-1]:
        rows = good[good[focus] == value]
        near = rows[rows._src == anchor] if anchor is not None else rows.iloc[0:0]
        row = (near if len(near) else rows).iloc[0]
        keys.append(row)
        anchor = anchor if anchor is not None else row._src
    last = good[good[focus] == values[-1]]
    busiest = last._src.value_counts().index
    names = list(dict.fromkeys([*(r._src for r in keys), *busiest, *lake.sources]))[:5]
    key_ids = [r.row_id for r in keys]

    sources, truth, confidence = {}, {}, {}
    for i, name in enumerate(names):
        t = lake.sources[name].copy()
        t["_p"] = model.predict_proba(t)[label].to_numpy()
        t[attr] = t.row_id.map(lake.truth)
        right = t[(t._p >= 0.5) == (t[attr] == label)]
        key_rows = t[t.row_id.isin(key_ids)]
        flood = right[right.row_id.isin(last.row_id)
                      & ~right.row_id.isin(key_rows.row_id)]
        on_focus = right[focus].isin(claim.static[focus])
        on_rest = pd.Series(True, index=right.index)
        for a, vals in claim.static.items():
            if a != focus:
                on_rest &= right[a].isin(vals)
        other = right[attr] != label
        picked = pd.concat([
            key_rows, _take(flood, 3, seed + i),
            _take(right[on_rest & on_focus & other], 7, seed + i),
            _take(right[~on_rest & on_focus & other], 4, seed + i)])
        rest = right[~right.row_id.isin(picked.row_id) & ~(on_rest & on_focus)]
        picked = pd.concat([picked, _take(rest, size - len(picked), seed + i)])
        picked = picked.sample(frac=1.0, random_state=seed + i)
        for rid, value, p in zip(picked.row_id, picked[attr], picked._p):
            truth[int(rid)], confidence[int(rid)] = value, float(p)
        sources[name] = picked.drop(columns=[attr, "_p"]).reset_index(drop=True)

    model.reset()
    row_source = {int(r): n for n, s in sources.items() for r in s.row_id}
    with_value = {n: s.assign(**{attr: s.row_id.map(truth)}) for n, s in sources.items()}
    base = dict(df_train=pd.DataFrame(), truth=truth, specs=[], row_source=row_source)
    tag = hashlib.sha1(f"{lake.split}|{claim.key()}".encode()).hexdigest()[:8]
    return SmallInstance(
        predicted=Lake(name=f"small-{tag}", sources=sources, static_attrs=lake.static_attrs,
                       predicted_attrs=[attr], **base),
        stored=Lake(name=f"small-stored-{tag}", sources=with_value,
                    static_attrs=[*lake.static_attrs, attr],
                    predicted_attrs=[], **base),
        confidence=pd.Series(confidence),
        claim=replace(claim, static={**claim.static, focus: values}))


def _quoted(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _condition(attribute: str, values: list) -> str:
    return (f"{attribute} = {_quoted(values[0])}" if len(values) == 1
            else f"{attribute} IN ({', '.join(_quoted(v) for v in values)})")


def _table_name(source: str) -> str:
    return re.sub(r"\W+", "_", source_label("acs-income", source).lower()).strip("_")


def plain_sql(inst: SmallInstance, claim: Claim, stored: bool,
              one_per_case: bool = False) -> dict:
    """What SQL does on the five tables: a CTE unions the sources, then the
    conditions apply. `claim` is the query as it stands now, which may have been
    edited since the tables were chosen. Returns its text, what it reads, what
    it returns, and how many model calls it needs when the attribute is not
    stored. `one_per_case` is the second query: one row per requested
    combination of the stored values."""
    attr, label = claim.target, claim.label
    cond = claim.static
    combo = list(cond)                    # one row per combination of these
    where = "\n    AND ".join(_condition(a, v) for a, v in cond.items())
    union = "\n  UNION ALL ".join(f"SELECT * FROM {_table_name(n)}"
                                  for n in inst.stored.sources)
    steps = [f"WITH all_sources AS (\n  {union}\n)"]
    if stored:
        filtered = f"{where}\n    AND {_condition(attr, [label])}"
        source = "all_sources"
    else:
        steps.append(f"matching AS (\n  SELECT * FROM all_sources\n  WHERE {where}\n)")
        steps.append(f"labelled AS (\n  SELECT *, predict_{attr}(matching.*) AS {attr}\n"
                     "  FROM matching  -- one model call per row\n)")
        filtered = _condition(attr, [label])
        source = "labelled"
    if one_per_case:
        steps.append(f"numbered AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY {', '.join(combo)}) AS rn\n"
                     f"  FROM {source}\n  WHERE {filtered}\n)")
        final = "SELECT * FROM numbered WHERE rn = 1;"
    elif stored:
        top_level = filtered.replace("\n    AND", "\n  AND")
        final = f"SELECT *\nFROM all_sources\nWHERE {top_level};"
    else:
        final = f"SELECT * FROM labelled\nWHERE {filtered};"
    text = ",\n".join(steps) + "\n" + final
    if not stored:
        text = f"-- {attr} is not found in the tables: it needs to be predicted\n" + text

    frames = [s.assign(table=n) for n, s in inst.stored.sources.items()]
    rows = pd.concat(frames, ignore_index=True)
    for a, vals in cond.items():
        rows = rows[rows[a].isin(vals)]
    predictions = 0
    if not stored:
        predictions = len(rows)
        rows = rows[rows.row_id.map(inst.confidence) >= CONFIDENCE]
    else:
        rows = rows[rows[attr] == label]
    if one_per_case:
        rows = rows.drop_duplicates(subset=combo)
    return {"sql": text, "rows": rows.reset_index(drop=True),
            "tables": len(frames), "predictions": predictions}


def diverse_sql(inst: SmallInstance, claim: Claim, stored: bool) -> dict:
    """The SQL closest to a Diverse answer: loosen the conditions so a row
    qualifies if it meets ANY of them, then keep one row per combination of
    values. It shows every value these tables hold next to the requested ones,
    but its size is set by the data, not by a target. When the attribute is not
    stored, every row needs a prediction: any row may carry the requested value."""
    attr, label = claim.target, claim.label
    q_all = {**claim.static, attr: [label]}
    cols = list(q_all)
    union = "\n  UNION ALL ".join(f"SELECT * FROM {_table_name(n)}" for n in inst.stored.sources)
    steps = [f"WITH all_sources AS (\n  {union}\n)"]
    source = "all_sources"
    if not stored:
        steps.append(f"labelled AS (\n  SELECT *, predict_{attr}(all_sources.*) AS {attr}\n"
                     "  FROM all_sources  -- one model call per row\n)")
        source = "labelled"
    any_of = "\n     OR ".join(_condition(a, v) for a, v in q_all.items())
    steps.append(f"candidates AS (\n  SELECT * FROM {source}\n  WHERE {any_of}\n)")
    steps.append(f"one_each AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY {', '.join(cols)}) AS rn\n"
                 "  FROM candidates\n)")
    text = ",\n".join(steps) + "\nSELECT * FROM one_each WHERE rn = 1;"
    if not stored:
        text = f"-- {attr} is not found in the tables: it needs to be predicted\n" + text

    frames = [s.assign(table=n) for n, s in inst.stored.sources.items()]
    rows = pd.concat(frames, ignore_index=True)
    predictions = 0
    if stored:
        rows[attr] = rows.row_id.map(inst.stored.truth)
    else:
        predictions = len(rows)
        p = rows.row_id.map(inst.confidence)
        sure = (p >= CONFIDENCE) | (p <= 1 - CONFIDENCE)       # the model is right on these tables
        rows[attr] = rows.row_id.map(inst.stored.truth).where(sure)
    hit = pd.Series(False, index=rows.index)
    for a, vals in q_all.items():
        hit |= rows[a].isin(vals)
    rows = rows[hit].drop_duplicates(subset=cols)
    return {"sql": text, "rows": rows.reset_index(drop=True),
            "tables": len(frames), "predictions": predictions}


def _closest_instance(lake: Lake, expensive: dict, claim: Claim, size: int, seed: int) -> SmallInstance | None:
    """No person fits every condition with a confident prediction. Take the five
    tables holding the most people who match the most stored conditions, and
    from each the closest people first, so that both SQL and Witness have
    something to work on and the comparison shows what each does then."""
    attr, label = claim.target, claim.label
    model = expensive[attr]

    def level(df: pd.DataFrame) -> pd.Series:
        lvl = pd.Series(0, index=df.index)
        for a, vals in claim.static.items():
            if a in df.columns:
                lvl = lvl + df[a].isin(vals).astype(int)
        return lvl

    score = {}
    for name, df in lake.sources.items():
        lvl = level(df)
        score[name] = (int((lvl == len(claim.static)).sum()), int((lvl == lvl.max()).sum()) if len(lvl) else 0,
                       int(lvl.max()) if len(lvl) else 0)
    names = sorted(score, key=lambda n: (score[n][2], score[n][0], score[n][1]), reverse=True)[:5]
    if not names:
        return None
    sources, truth, confidence = {}, {}, {}
    for i, name in enumerate(names):
        t = lake.sources[name].copy()
        t["_p"] = model.predict_proba(t)[label].to_numpy()
        t[attr] = t.row_id.map(lake.truth)
        right = t[(t._p >= 0.5) == (t[attr] == label)]
        right = right.assign(_lvl=level(right).to_numpy())
        near = right.sort_values("_lvl", ascending=False, kind="stable")
        top = near[near._lvl == near._lvl.max()]
        picked = pd.concat([_take(top, 12, seed + i)])
        rest = right[~right.row_id.isin(picked.row_id)]
        picked = pd.concat([picked, _take(rest, size - len(picked), seed + i)])
        picked = picked.sample(frac=1.0, random_state=seed + i)
        for rid, value, p in zip(picked.row_id, picked[attr], picked._p):
            truth[int(rid)], confidence[int(rid)] = value, float(p)
        sources[name] = picked.drop(columns=[attr, "_p", "_lvl"]).reset_index(drop=True)
    model.reset()
    row_source = {int(r): n for n, s in sources.items() for r in s.row_id}
    with_value = {n: s.assign(**{attr: s.row_id.map(truth)}) for n, s in sources.items()}
    base = dict(df_train=pd.DataFrame(), truth=truth, specs=[], row_source=row_source)
    tag = hashlib.sha1(f"closest|{lake.split}|{claim.key()}".encode()).hexdigest()[:8]
    return SmallInstance(
        predicted=Lake(name=f"small-{tag}", sources=sources, static_attrs=lake.static_attrs,
                       predicted_attrs=[attr], **base),
        stored=Lake(name=f"small-stored-{tag}", sources=with_value,
                    static_attrs=[*lake.static_attrs, attr], predicted_attrs=[], **base),
        confidence=pd.Series(confidence), claim=claim, exact=False)
