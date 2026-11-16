"""Presentation helpers: the page's look, table styling, and the charts.

Kept out of app.py so the page reads as layout, and out of runner.py so that
nothing about appearance can reach the pipeline. Colour stays quiet: teal is
Witness, or a requested value; amber is a value that was not requested.
"""

from __future__ import annotations

import pandas as pd

TEAL = "#0F766E"
TEAL_SOFT = "#D9F0EA"
AMBER = "#B45309"
AMBER_SOFT = "#FCEFD2"
INK = "#1F1B16"
TEXT2 = "#44403C"
MUTED = "#57534E"
LINE = "#E8E4DC"
NEUTRAL_BAR = "#CFC9BE"
AWR, TWR, DWR = "AWR", "TWR", "DWR"
REPEAT = "background-color: #ECE8E0; color: #8A847B; font-weight: 400"   # a row adding nothing new
SERIES = ["#0D9488", "#5B5BD6", "#D9770B", "#B5478F"]   # checked for colour-blind separation
SHORT = {"Proxy ranking (whole lake)": "Cheap Proxy", "Random source order": "Random order",
         "Exhaustive (predict everything)": "Exhaustive",
         "Witness — no incremental consumption": "Witness, no batches",
         "Witness — no cheap sampling": "Witness, no sampling",
         "Static selection (histogram-only)": "Static selection"}

# The mark of the tool: a table with one row picked out, the witness.
LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 236 44" width="236" height="44">
<rect x="2" y="4" width="40" height="36" rx="7" fill="#FFFFFF" stroke="#0F766E" stroke-width="2.4"/>
<rect x="2" y="4" width="40" height="10" rx="5" fill="#0F766E"/>
<rect x="3.2" y="9" width="37.6" height="5" fill="#0F766E"/>
<rect x="7" y="18.5" width="30" height="4" rx="2" fill="#D6D0C4"/>
<rect x="5.5" y="25" width="33" height="6.5" rx="2.5" fill="#0F766E"/>
<path d="M27.5 28.3l2.2 2.1 4-4.2" fill="none" stroke="#FFFFFF" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="7" y="34" width="30" height="3" rx="1.5" fill="#D6D0C4"/>
<text x="54" y="31" font-family="Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="25" font-weight="700" letter-spacing="3.2" fill="#1F1B16">WITNESS</text>
</svg>"""
LOGO_ICON = LOGO.replace('viewBox="0 0 236 44" width="236"', 'viewBox="0 0 44 44" width="44"')

# The name of the tool, top left: an eye (the witness) whose iris is a table.
BRAND = """
<div class="brand">
<svg class="brand-mark" viewBox="0 0 64 64" width="54" height="54" aria-hidden="true">
<defs>
<linearGradient id="brand-bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2DD4BF"/><stop offset=".5" stop-color="#0F766E"/><stop offset="1" stop-color="#123F3B"/></linearGradient>
<radialGradient id="brand-iris" cx=".4" cy=".35" r=".7"><stop offset="0" stop-color="#14B8A6"/><stop offset="1" stop-color="#0B4F4A"/></radialGradient>
</defs>
<rect x="2" y="2" width="60" height="60" rx="17" fill="url(#brand-bg)"/>
<path d="M8 32 C17 17.5 47 17.5 56 32 C47 46.5 17 46.5 8 32 Z" fill="#FFFFFF"/>
<circle cx="32" cy="32" r="9" fill="url(#brand-iris)"/>
<rect x="27.6" y="28" width="8.8" height="8" rx="1.6" fill="none" stroke="#FFFFFF" stroke-width="1.4"/>
<path d="M27.6 31 H36.4 M32 31 V36" stroke="#FFFFFF" stroke-width="1.2"/>
<circle cx="35.9" cy="27.1" r="1.6" fill="#CCFBF1"/>
</svg>
<span class="brand-name">WITNESS</span>
</div>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,800&display=swap');
.brand { display: flex; align-items: center; gap: .75rem; margin: -2.6rem 0 1.4rem .1rem; }
.brand-mark { flex: none; filter: drop-shadow(0 4px 10px rgba(15,118,110,.32)); }
.brand-name { font-family: 'Fraunces', Georgia, 'Times New Roman', serif; font-weight: 800; font-size: 2.15rem;
              letter-spacing: .08em; line-height: 1.1; white-space: nowrap; padding-right: .1em;
              background: linear-gradient(120deg, #14B8A6 0%, #0F766E 45%, #123F3B 100%);
              -webkit-background-clip: text; background-clip: text; color: transparent;
              filter: drop-shadow(0 2px 2px rgba(18,63,59,.22)); }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1360px; }
footer, [data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"], [data-testid="stToolbar"] [data-testid="stBaseButton-header"] { display: none !important; }
[data-testid="stExpandSidebarButton"] { display: flex !important; }
[data-testid="stHeader"] { background: transparent; }

h1 { font-weight: 700; letter-spacing: -.022em; font-size: 2.05rem !important; padding-bottom: .1rem !important; }
.lede { color: #57534E; font-size: 1.02rem; line-height: 1.5; margin: 0 0 .3rem 0; max-width: 62rem; }

.step { display: flex; align-items: center; gap: .6rem; font-size: 1.22rem; font-weight: 650;
        letter-spacing: -.012em; color: #1F1B16; margin: 1.1rem 0 .35rem 0; }
.step b { width: 1.65rem; height: 1.65rem; border-radius: 50%; background: #0F766E; color: #fff;
          font-size: .86rem; font-weight: 650; display: inline-flex; align-items: center; justify-content: center; }
.sub { font-size: .92rem; font-weight: 600; color: #1F1B16; margin-bottom: .15rem; }
.sub span { font-weight: 400; color: #57534E; margin-left: .35rem; }

/* cards: every bordered container has a card_ key */
div[class*="st-key-card"] {
    background: #FFFFFF; border: 1px solid #E8E4DC !important; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(31,27,22,.03), 0 8px 24px rgba(31,27,22,.035);
    padding: 1.15rem 1.35rem 1.2rem 1.35rem;
}

/* chips: quiet teal instead of a saturated fill */
span[data-tag] { background: #E3F1EE !important; color: #0F5E57 !important; border-radius: 7px !important;
                 font-weight: 500; border: none !important; }
span[data-tag] button, span[data-tag] svg { color: #0F5E57 !important; fill: #0F5E57 !important; }
[data-testid="stMultiSelectTagsContainer"] { flex-wrap: wrap !important; overflow: visible !important; row-gap: 3px;
                                             flex: 1 1 0 !important; min-width: 0 !important; }
[data-testid="stMultiSelectTagsContainer"] > span { max-width: 100%; }
div:has(> [data-testid="stMultiSelectTagsContainer"]) { max-height: none !important; height: auto !important; min-height: 40px; }

/* labels of the controls */
[data-testid="stWidgetLabel"] p { font-size: .86rem; color: #44403C; font-weight: 500; }
[data-testid="stCaptionContainer"] p { color: #57534E; font-size: .86rem; line-height: 1.45; }
[data-testid="stRadioCaption"] p { color: #57534E; }

/* at a glance */
.glance { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: .8rem; margin: .7rem 0 .2rem 0; }
.tile { background: #FAF8F5; border-radius: 11px; padding: .75rem 1rem .7rem 1rem; }
.tile h4 { font-size: .8rem !important; font-weight: 600; color: #44403C; margin: 0 0 .4rem 0; padding: 0 !important; letter-spacing: 0; }
.tile div { display: flex; justify-content: space-between; align-items: baseline; font-size: .92rem; color: #44403C; line-height: 1.55; }
.tile div b { font-weight: 600; color: #1F1B16; font-variant-numeric: tabular-nums; }
.tile div.us { color: #0F766E; font-weight: 600; }
.tile div.us b { font-size: 1.32rem; font-weight: 650; color: #0F766E; letter-spacing: -.015em; }

/* one answer's figures, on a line above its table */
.astrip { display: flex; flex-wrap: wrap; gap: .35rem 1.6rem; font-size: .95rem; color: #1F1B16;
          margin: .15rem 0 .55rem 0; padding: .55rem .2rem .6rem .2rem; border-bottom: 1px solid #E8E4DC; }
.astrip b { font-weight: 650; }
.astrip i { font-style: normal; color: #57534E; font-size: .86rem; }
.astrip .ok { color: #0F766E; font-weight: 600; }
.astrip .miss { color: #B45309; font-weight: 600; }
[data-testid="stMetric"] { background: #FAF8F5; border-radius: 11px; padding: 10px 14px 8px 14px; }
[data-testid="stMetricLabel"] p { font-size: .78rem !important; font-weight: 600; color: #44403C; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 650; letter-spacing: -.02em; }

[data-testid="stSidebar"] { border-right: 1px solid #E8E4DC; }
[data-testid="stSidebar"] { min-width: 292px !important; max-width: 292px !important; width: 292px !important;
                             transform: none !important; }
/* a fixed sidebar: no button to fold it, no handle to resize it */
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebar"] > div:not([data-testid="stSidebarContent"]) { display: none !important; }
[data-testid="stSidebar"] h2 { font-size: 1rem; margin-top: .7rem; padding-bottom: 0; }
[data-testid="stDataFrame"] { border-radius: 10px; }
[data-testid="stCode"] { border-radius: 10px; }
[data-testid="stLogo"] { height: 2.5rem; max-width: 15rem; }
[data-testid="stSidebarHeader"] { padding-top: 1.1rem; padding-bottom: .4rem; }
[data-testid="stExpander"] details { border-radius: 11px; border-color: #E8E4DC; background: #FFFFFF; }
hr { margin: .7rem 0 !important; border-color: #EDE9E1 !important; }

[data-testid="stAlertContainer"] { background: #EEF6F4 !important; color: #1F1B16 !important; border-radius: 11px; }
[data-testid="stAlertContainer"] p { color: #1F1B16 !important; }

[data-testid="stMain"], [data-testid="stMain"] * { overflow-anchor: none !important; }
</style>
"""


def step(number: str, title: str) -> str:
    """A numbered section heading, as HTML."""
    return f'<div class="step"><b>{number}</b>{title}</div>'


def short_name(method: str) -> str:
    return SHORT.get(method, method)


def glance(rows: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Facts side by side: [(tile title, [(method, value), ...]), ...]."""
    tiles = []
    for title, values in rows:
        lines = "".join(f'<div class="{"us" if m == "Witness" else ""}"><span>{m}</span><b>{v}</b></div>'
                        for m, v in values)
        tiles.append(f'<div class="tile"><h4>{title}</h4>{lines}</div>')
    return f'<div class="glance">{"".join(tiles)}</div>'


def answer_strip(items: list[tuple], met: bool, target: str) -> str:
    """The figures of one answer on a single line, above its table:
    [(label, value) or (label, value, note)], then whether the target was reached."""
    from html import escape
    parts = []
    for item in items:
        label, value = item[0], item[1]
        note = f' <i>({escape(item[2])})</i>' if len(item) > 2 else ""
        parts.append(f'<span><b>{escape(label)}:</b> {escape(value)}{note}</span>')
    verdict = ("reached" if met else "not reached")
    parts.append(f'<span class="{"ok" if met else "miss"}">{"&#10003;" if met else "&#10007;"} '
                 f'target {verdict} <i>({escape(target.split(",")[0])})</i></span>')
    return f'<div class="astrip">{"".join(parts)}</div>'


def style_comparison(df: pd.DataFrame):
    """The comparison table. One quiet emphasis: the Witness row. A run that
    missed the target says so in its own column."""
    df = df.rename(columns={"variant": "method", "rows shipped": "rows scanned",
                            "sources": "tables scanned",
                            "AEv": AWR, "VEv": TWR, "CEv": DWR}).copy()
    df["method"] = df["method"].map(short_name)
    if "met" in df.columns:
        df.insert(df.columns.get_loc("met"), "reached θ",
                  df.pop("met").map({True: "yes", False: "no"}))

    def row_style(row):
        base = (f"background-color: {TEAL_SOFT}; color:#0F4F49; font-weight:600"
                if row["method"] == "Witness" else "background-color: #FFFFFF")
        return [base] * len(row)

    def reached_style(col):
        return ["color:#92400E; font-weight:600" if v == "no" else "" for v in col]

    # One format call: each call without a column list resets the format of
    # every column it does not name.
    fmt = {c: "{:,.0f}" for c in ("expensive", "cheap", "tables scanned", "rows scanned",
                                  "requests") if c in df.columns}
    fmt.update({c: f for c, f in ((AWR, "{:.2f}"), (TWR, "{:.2f}"), (DWR, "{:.2f}"),
                                  ("total (s)", "{:,.1f}"), ("total ($)", "{:,.2f}"))
                if c in df.columns})
    return (df.style.apply(row_style, axis=1)
            .apply(reached_style, subset=["reached θ"])
            .format(fmt, subset=list(fmt), na_rep="—"))


def style_benchmark(df: pd.DataFrame):
    """The saved benchmark, one row per method."""
    order = ["Witness", "Proxy ranking (whole lake)", "Random source order",
             "Exhaustive (predict everything)", "Witness — no incremental consumption",
             "Static selection (histogram-only)"]
    diverse = df["problem"].iloc[0] == "Diverse"
    keep = {"method": "method", "queries": "queries", "reached": "reached θ %", "expensive": "expensive",
            "cheap": "cheap", "tables": "tables scanned", "rows": "rows scanned", "requests": "requests",
            **({"TWR": TWR, "DWR": DWR} if diverse else {"AWR": AWR}), "clock": "clock per query (s)"}
    view = df.assign(_o=df["method"].map(lambda m: order.index(m) if m in order else 99)).sort_values("_o")
    view = view[list(keep)].rename(columns=keep)
    view["method"] = view["method"].map(short_name)

    def row_style(row):
        base = (f"background-color: {TEAL_SOFT}; color:#0F4F49; font-weight:600"
                if row["method"] == "Witness" else "background-color: #FFFFFF")
        return [base] * len(row)

    fmt = {c: "{:,.0f}" for c in ("queries", "reached θ %", "expensive", "cheap", "rows scanned", "requests")}
    fmt.update({"tables scanned": "{:,.1f}", "clock per query (s)": "{:,.1f}"})
    fmt.update({c: "{:.2f}" for c in (AWR, TWR, DWR) if c in view.columns})
    return view.style.apply(row_style, axis=1).format(fmt, na_rep="—")


def _check(q: dict, plan: dict) -> tuple[str, bool]:
    """Quality after an iteration against the target, in words."""
    if plan["diverse"]:
        ok = q["vev"] >= plan["theta_vev"] and q["cev"] >= plan["cev_min"]
        return (f"TWR {q['vev']:.2f} (target {plan['theta_vev']:.2f})<br/>"
                f"DWR {q['cev']:.2f} (target {plan['cev_min']:.2f})"), ok
    ok = q["aev"] >= plan["theta_aev"]
    return f"AWR {q['aev']:.2f}, target {plan['theta_aev']:.2f}", ok


def plan_dot(name: str, plan: dict, label_of=lambda s: s, shown: int = 3) -> str:
    """A run as its loop unrolled (Graphviz DOT): one row per iteration, read
    left to right — pick a table, scan it, label rows with the expensive
    model (shaded: the paid part), check the quality against the target —
    and an arrow back to the next iteration until the target is reached.
    Long runs fold their tail into one row."""
    from html import escape
    from math import ceil

    def src(name_):
        if str(name_).startswith("all "):          # Cheap Proxy's single step: every table at once
            return f"<b>{escape(str(name_).replace(' sources', ' tables'))}</b>"
        try:
            shown = label_of(name_)
        except ValueError:
            shown = str(name_)
        return f"<b>{escape(shown)}</b>"

    out = ['digraph plan {',
           'rankdir=LR; nodesep=0.25; ranksep=0.35; bgcolor="transparent"; newrank=true;',
           'node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", '
           'color="#A8A29E", fontname="Helvetica", fontsize=11, margin="0.12,0.06"];',
           'edge [color="#78716C", arrowsize=0.6];',
           f'labelloc=t; labeljust=l; fontname="Helvetica"; fontsize=13; '
           f'label=<<b>{escape(name)}</b>: {plan["expensive"]:,} expensive predictions, '
           f'{plan["cheap"]:,} cheap>;']
    steps = plan["steps"]
    rows = []
    if len(steps) == shown + 1:         # no point collapsing a single iteration
        shown += 1
    for i, st_ in enumerate(steps[:shown]):
        n = i + 1
        cands = st_["candidates"]
        ranked = max(0, st_["cheap"] - st_.get("sampled", 0))   # cheap calls that ordered rows
        if cands:
            tested = sum(c["state"] != "skipped" for c in cands)
            how = (f"asked {len(cands)} tables how many rows match<br/>"
                   f"sampled {tested} of them: {st_.get('sampled', 0):,} rows, "
                   "labelled by the cheap model")
        else:
            how = {"Statistics pick": "chosen from statistics, no model",
                   "Random pick": "chosen at random",
                   "Next source": "every table, one after another",
                   "Proxy ranking": "rows of every table, ranked together by "
                                    "stored match, then cheap-model score"
                   }.get(plan["pick"], "chosen to supply what the target still lacks")
        out.append(f'it{n} [shape=plaintext, style="", label=<<b>Iteration {n}</b>>];')
        out.append(f'pick{n} [label=<Pick a table: {src(st_["source"])}<br/>'
                   f'<font point-size="10">{how}</font>>];')
        order = (f'<br/><font point-size="10">cheap model ranks {ranked:,} of them</font>'
                 if ranked else "")
        out.append(f'read{n} [label=<Scan it<br/>{st_["shipped"]:,} rows returned{order}>];')
        if st_["labelled"] < st_["shipped"]:
            batches = ceil(st_["labelled"] / plan["chunk"])
            lab = (f'Expensive model labels<br/><b>{st_["labelled"]:,}</b> of {st_["shipped"]:,} rows<br/>'
                   f'<font point-size="10">{batches} batch{"es" if batches != 1 else ""} of {plan["chunk"]}, then stop</font>')
        else:
            lab = f'Expensive model labels<br/><b>all {st_["shipped"]:,}</b> rows'
        out.append(f'label{n} [fillcolor="#CCFBF1", color="#0F766E", label=<{lab}>];')
        text, ok = _check(st_, plan)
        last = i == len(steps) - 1
        if ok:
            out.append(f'check{n} [fillcolor="#DCFCE7", color="#15803D", '
                       f'label=<{text}<br/><b>&#10003; target reached</b>>];')
        elif last and plan["gave_up"]:
            out.append(f'check{n} [fillcolor="#FEF3C7", color="#B45309", '
                       f'label=<{text}<br/><b>no gain above ε: give up</b>>];')
        else:
            out.append(f'check{n} [fillcolor="#FEF3C7", color="#B45309", '
                       f'label=<{text}<br/><b>not yet: next iteration</b>>];')
        out.append(f"it{n} -> pick{n} -> read{n} -> label{n} -> check{n};")
        rows.append(n)
    rest = steps[shown:]
    if rest:
        n = shown + 1
        out.append(f'it{n} [shape=plaintext, style="", label=<<b>Iterations {n}&#8211;{len(steps)}</b>>];')
        out.append(f'more{n} [fillcolor="#CCFBF1", color="#0F766E", label=<{len(rest)} more tables scanned<br/>'
                   f'{sum(r["shipped"] for r in rest):,} rows returned, '
                   f'<b>{sum(r["labelled"] for r in rest):,}</b> labelled>];')
        text, ok = _check(rest[-1], plan)
        out.append(f'check{n} [fillcolor="{"#DCFCE7" if ok else "#FEF3C7"}", '
                   f'color="{"#15803D" if ok else "#B45309"}", label=<{text}>];')
        out.append(f"it{n} -> more{n} -> check{n};")
        rows.append(n)
    for a, b in zip(rows, rows[1:]):
        out.append(f'check{a} -> it{b} [constraint=false, style=dashed, color="#B45309"];')
    if rows:
        out.append("{rank=same; " + " ".join(f"it{n};" for n in rows) + "}")
        for a, b in zip(rows, rows[1:]):
            out.append(f"it{a} -> it{b} [style=invis];")
        verdict = ("" if plan["met"] else
                   f'<br/><font color="#B45309">stopped: {escape(plan["stop"])}</font>')
        n_rows = plan["answer_rows"]
        out.append(f'answer [penwidth=2, color="#1C1917", label=<<b>Answer</b><br/>'
                   f'{n_rows} row{"s" if n_rows != 1 else ""}{verdict}>];')
        out.append(f"check{rows[-1]} -> answer;")
    else:
        out.append('none [label=<no table was scanned>];')
    out.append("}")
    return "\n".join(out)


def method_bars(df: pd.DataFrame, column: str, title: str, money: bool = False):
    """One measure per method on a log axis, every bar labelled, in the same
    method order in every chart. Witness is the one coloured bar."""
    import altair as alt
    data = df.assign(name=df["variant"].map(short_name), value=df[column].astype(float))
    positive = data.value[data.value > 0]
    low = (positive.min() if len(positive) else 1.0) / 2.5
    high = (positive.max() if len(positive) else 2.0)
    missed = not data.met.all()
    high = high * (high / low) ** (0.95 if missed else 0.55)   # room for the labels on the right
    def label(v, met):
        text = f"${v:,.3g}" if money else f"{v:,.0f}"
        return text + ("  · missed" if not met else "")
    data = data.assign(label=[label(v, m) for v, m in zip(data.value, data.met)],
                       reached=data.met.map({True: "yes", False: "no"}),
                       drawn=data.value.clip(lower=low))
    order = list(data.name)
    y = alt.Y("name:N", sort=order, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelColor=INK, labelFontSize=13,
                            labelLimit=170, labelPadding=10))
    x = alt.X("drawn:Q", title=None, axis=None,
              scale=alt.Scale(type="log", domain=[low, high], nice=False, clamp=True))
    tip = [alt.Tooltip("variant:N", title="method"), alt.Tooltip("value:Q", title=title, format=",.4~g"),
           alt.Tooltip("reached:N", title="reached θ")]
    bars = (alt.Chart(data).mark_bar(height=18, cornerRadiusEnd=4)
            .encode(x=x, y=y, tooltip=tip,
                    color=alt.condition(alt.datum.name == "Witness", alt.value(SERIES[0]),
                                        alt.value(NEUTRAL_BAR))))
    text = (alt.Chart(data).mark_text(align="left", dx=7, fontSize=13, color=INK)
            .encode(x=x, y=y, text="label:N", tooltip=tip))
    return ((bars + text)
            .properties(height=max(90, 36 * len(data)), background="transparent",
                        title=alt.TitleParams(title, anchor="start", color=INK, fontSize=15,
                                              fontWeight=600, subtitle="log scale", subtitleColor=TEXT2,
                                              subtitleFontSize=12, offset=10))
            .configure_view(strokeWidth=0).configure(font="Inter, Helvetica, sans-serif"))


def effort_map(cells: pd.DataFrame, x_title: str, y_title: str, vmax: int,
               last: bool, first: bool):
    """One method's expensive predictions per table, on the dataset's 2-D grid
    of tables. Log colour scale, shared by the methods. Dark grey: the table
    was scanned but got no prediction. Light grey: it was never scanned. A teal
    dot: it was only sampled (a count and a few rows)."""
    import altair as alt
    x = alt.X("col:N", title=x_title if last else None, sort=sorted(cells.col.unique()),
              axis=alt.Axis(labels=last, ticks=False, domain=False, labelAngle=-90,
                            labelColor=TEXT2, titleColor=TEXT2, labelFontSize=11))
    y = alt.Y("row:N", title=None, sort=sorted(cells.row.unique()),
              axis=alt.Axis(ticks=False, domain=False, labelColor=INK,
                            labelFontSize=11, labelLimit=220, labelOverlap=False))
    tip = [alt.Tooltip("source:N", title="table"),
           alt.Tooltip("predictions:Q", title="expensive predictions", format=",")]
    rect = dict(stroke="#FFFFFF", strokeWidth=2, cornerRadius=2)
    never = (alt.Chart(cells[(cells.predictions == 0) & ~cells.read]).mark_rect(color="#ECE8E0", **rect)
             .encode(x=x, y=y, tooltip=tip))
    read = (alt.Chart(cells[(cells.predictions == 0) & cells.read]).mark_rect(color="#78716C", **rect)
            .encode(x=x, y=y, tooltip=tip))
    legend = (alt.Legend(orient="top", title="expensive predictions (log scale)",
                         titleColor=TEXT2, labelColor=TEXT2, format="~s", gradientLength=220,
                         titleLimit=320)
              if first else None)
    spent = (alt.Chart(cells[cells.predictions > 0]).mark_rect(**rect)
             .encode(x=x, y=y, tooltip=tip,
                     color=alt.Color("predictions:Q", legend=legend,
                                     scale=alt.Scale(type="log", domain=[1, max(vmax, 2)],
                                                     range=[TEAL_SOFT, "#134E4A"]))))
    height = (max(170, 22 * cells.row.nunique())
              + (70 if first else 0) + (100 if last else 0))    # room for legend, labels
    layers = never + read + spent
    only_sampled = (cells[cells.sampled & ~cells.read] if "sampled" in cells.columns
                    else cells.iloc[0:0])
    if len(only_sampled):
        layers = layers + (alt.Chart(only_sampled).mark_point(filled=True, size=34, color=TEAL, opacity=1)
                           .encode(x=x, y=y, tooltip=tip))
    return (layers.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0).configure(font="Inter, Helvetica, sans-serif"))


def theta_chart(df: pd.DataFrame, value: str, title: str, fmt: str, budget: float | None = None):
    """One cost against the target θ, one line per method, on a log axis zoomed
    to the data. Witness is emphasised and carries its values."""
    import altair as alt
    df = df.assign(method=df.method.map(short_name))
    methods = list(df.method.unique())
    colors = [SERIES[i % len(SERIES)] for i in range(len(methods))]
    dashes = [[1, 0] if m == "Witness" else d for m, d in zip(methods, [[6, 4], [2, 4], [8, 3, 2, 3], [1, 0]])]
    widths = [3.5 if m == "Witness" else 2.25 for m in methods]
    low, high = df[value].min() * 0.6, df[value].max() * 1.7
    if budget:
        low, high = min(low, budget * 0.6), max(high, budget * 1.7)
    x = alt.X("θ:O", title="target θ (AWR)",
              axis=alt.Axis(labelAngle=0, ticks=False, domain=False, grid=False,
                            labelColor=INK, titleColor=TEXT2, labelFontSize=12))
    marks = [m * 10.0 ** k for k in range(-4, 8) for m in (1, 3) if low <= m * 10.0 ** k <= high]
    y = alt.Y(f"{value}:Q", title=None, scale=alt.Scale(type="log", domain=[low, high], nice=False),
              axis=alt.Axis(values=marks, ticks=False, domain=False, gridColor=LINE, labelColor=INK,
                            format="$.3~f" if fmt.startswith("$") else fmt, labelFontSize=11))
    name = alt.Scale(domain=methods, range=colors)
    base = alt.Chart(df).encode(x=x, y=y, tooltip=["method", "θ", alt.Tooltip(f"{value}:Q", format=fmt)])
    lines = base.mark_line(interpolate="monotone").encode(
        color=alt.Color("method:N", sort=methods, scale=name,
                        legend=alt.Legend(orient="top", title=None, labelColor=INK, labelFontSize=12,
                                          labelLimit=320, symbolType="stroke", symbolStrokeWidth=3)),
        strokeDash=alt.StrokeDash("method:N", sort=methods, legend=None,
                                  scale=alt.Scale(domain=methods, range=dashes)),
        strokeWidth=alt.StrokeWidth("method:N", sort=methods, legend=None,
                                    scale=alt.Scale(domain=methods, range=widths)))
    points = base.mark_point(filled=True, size=85, opacity=1, stroke="#FFFFFF", strokeWidth=1.5).encode(
        color=alt.Color("method:N", sort=methods, scale=name, legend=None))
    labels = (alt.Chart(df[df.method == "Witness"]).mark_text(dy=-15, fontSize=12, fontWeight=600, color=INK)
              .encode(x=x, y=y, text=alt.Text(f"{value}:Q", format=fmt)))
    layers = [lines, points, labels]
    if budget:
        line = pd.DataFrame({value: [budget], "note": ["budget"]})
        layers += [alt.Chart(line).mark_rule(color=AMBER, strokeWidth=2, strokeDash=[4, 4]).encode(y=y),
                   alt.Chart(line).mark_text(align="left", dx=6, dy=-8, color=AMBER, fontSize=12, fontWeight=600)
                   .encode(x=alt.value(0), y=y, text="note:N")]
    return (alt.layer(*layers).resolve_scale(color="shared")
            .properties(height=320, background="transparent",
                        title=alt.TitleParams(title, anchor="start", color=INK, fontSize=15, fontWeight=600))
            .configure_view(strokeWidth=0).configure(font="Inter, Helvetica, sans-serif"))


def style_small_table(df: pd.DataFrame, hit: pd.Series, chosen: pd.Series | None = None):
    """The five small tables: tint the rows the query is about, and fill the
    rows that ended up in the witness table."""
    def row(r):
        if chosen is not None and chosen[r.name]:
            return [f"background-color: {TEAL}; color: #FFFFFF; font-weight: 600"] * len(r)
        return [f"background-color: {TEAL_SOFT}" if hit[r.name] else ""] * len(r)
    return df.style.apply(row, axis=1)


def style_witness(df: pd.DataFrame, query):
    """Tint query-attribute cells: teal = a requested value, amber = a value
    that was not requested (what a Diverse answer is built from)."""
    from demo.explore import PRED_SUFFIX, UNSURE
    Q = query.all
    colmap = {a + (PRED_SUFFIX if a in query.predicted else ""): a for a in Q}

    def cell(v, attr):
        if pd.isna(v):
            return ""
        if v == UNSURE:
            return "color:#57534E; font-style:italic"
        if v in Q[attr]:
            return f"background-color: {TEAL_SOFT}; color:#115E59; font-weight:600"
        return f"background-color: {AMBER_SOFT}; color:#78350F"

    sty = df.style
    for col, attr in colmap.items():
        if col in df.columns:
            sty = sty.map(lambda v, a=attr: cell(v, a), subset=[col])
    return sty


def pareto_front(df: pd.DataFrame, cost: str, quality: str) -> pd.DataFrame:
    """The runs no other run beats on both counts: cheaper AND at least as good.
    Returned by increasing cost, so they also come by increasing quality."""
    pts = df.sort_values([cost, quality], ascending=[True, False])
    keep, best = [], -1.0
    for i, q in zip(pts.index, pts[quality]):
        if q > best + 1e-9:
            keep.append(i)
            best = q
    return pts.loc[keep]


def pareto_chart(df: pd.DataFrame, cost: str, title: str, fmt: str, quality: str = AWR,
                 budget: float | None = None, x_title: str | None = None):
    """Quality reached against what it cost, one point per run (a method at one
    target θ), on a log cost axis. The black stepped line is the Pareto front:
    the runs that no other run beats on both cost and quality."""
    import altair as alt
    df = df.assign(method=df.method.map(short_name))
    # runs of one method that land on the same point share one label
    pts = (df.groupby(["method", cost, quality], sort=False)["θ"]
             .apply(lambda t: "θ " + ", ".join(f"{v:g}" for v in sorted(t))).reset_index(name="targets"))
    methods = list(df.method.unique())
    colors = [SERIES[i % len(SERIES)] for i in range(len(methods))]
    low, high = pts[cost].min() * 0.55, pts[cost].max() * 1.9
    if budget:
        low, high = min(low, budget * 0.55), max(high, budget * 1.9)
    ylow = max(0.0, min(pts[quality].min(), 0.95) - 0.06)
    marks = [m * 10.0 ** k for k in range(-4, 8) for m in (1, 3) if low <= m * 10.0 ** k <= high]
    x = alt.X(f"{cost}:Q", title=x_title or title, scale=alt.Scale(type="log", domain=[low, high], nice=False),
              axis=alt.Axis(values=marks, ticks=False, domain=False, gridColor=LINE, labelColor=INK,
                            titleColor=TEXT2, format="$.3~f" if fmt.startswith("$") else fmt, labelFontSize=11))
    y = alt.Y(f"{quality}:Q", title=f"{quality} reached", scale=alt.Scale(domain=[ylow, 1.03], nice=False),
              axis=alt.Axis(ticks=False, domain=False, gridColor=LINE, labelColor=INK, titleColor=TEXT2,
                            format=".2f", labelFontSize=11, tickCount=6))
    tip = ["method", "targets", alt.Tooltip(f"{cost}:Q", format=fmt), alt.Tooltip(f"{quality}:Q", format=".2f")]
    front = pareto_front(pts, cost, quality)
    # the front as steps: quality stays where it is until the next, dearer run
    steps = pd.concat([front[[cost, quality]],
                       pd.DataFrame({cost: [high], quality: [front[quality].iloc[-1]]})], ignore_index=True)
    line = (alt.Chart(steps).mark_line(interpolate="step-after", color=INK, strokeWidth=1.6, opacity=.75)
            .encode(x=x, y=y))
    name = alt.Scale(domain=methods, range=colors)
    points = (alt.Chart(pts).mark_point(filled=True, size=120, opacity=1, stroke="#FFFFFF", strokeWidth=1.5)
              .encode(x=x, y=y, tooltip=tip,
                      color=alt.Color("method:N", sort=methods, scale=name,
                                      legend=alt.Legend(orient="top", title=None, labelColor=INK,
                                                        labelFontSize=12, labelLimit=320, symbolSize=110))))
    ring = (alt.Chart(front).mark_point(filled=False, size=300, stroke=INK, strokeWidth=1.6, opacity=1)
            .encode(x=x, y=y, tooltip=tip))
    labels = (alt.Chart(pts[pts.method == "Witness"])
              .mark_text(dy=-16, fontSize=11.5, fontWeight=600, color=INK)
              .encode(x=x, y=y, text="targets:N"))
    layers = [line, ring, points, labels]
    if budget:
        cap = pd.DataFrame({cost: [budget], "note": ["budget"]})
        layers += [alt.Chart(cap).mark_rule(color=AMBER, strokeWidth=2, strokeDash=[4, 4]).encode(x=x),
                   alt.Chart(cap).mark_text(align="left", dx=6, color=AMBER, fontSize=12, fontWeight=600)
                   .encode(x=x, y=alt.value(176), text="note:N")]
    return (alt.layer(*layers).resolve_scale(color="shared")
            .properties(height=340, background="transparent",
                        title=alt.TitleParams(title, anchor="start", color=INK, fontSize=15, fontWeight=600,
                                              subtitle="black line: the Pareto front", subtitleColor=TEXT2,
                                              subtitleFontSize=12))
            .configure_view(strokeWidth=0).configure(font="Inter, Helvetica, sans-serif"))
