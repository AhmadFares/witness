"""Predict — turn retrieved rows into labelled rows.

Baselines annotate EVERY retrieved row of the source they open: no incremental
consumption, no plateau stop. Those belong to PTVD and must not leak into a
baseline, or the comparison stops isolating selection.

PTVD consumes the selected source incrementally, which is what licenses
ranking on gain WITHOUT dividing by source size (Selection criterion): what a
source costs is governed by the stopping rule, not by n_S. Two mechanisms:

  Row-level skip (free).  A row whose static values match combination c at
  level j can never exceed j + |A_p| attributes of c. If that ceiling fails to
  beat r_T(c) for EVERY combination, the row cannot raise AEv, and — since
  pruning breaks ties in favour of incumbents — it cannot be retained either.
  It is dropped without ever being labelled.

  Stop rules.  A scan stops when the target is reached (anything further is
  waste: quality beyond theta is worth nothing), or when no remaining row can
  clear the ceiling any more. Every labelled row raises some r_T(c), which
  lowers what the remaining rows must clear; re-testing them against the
  updated T uses the same admissible bound as the row skip, so the stop is
  lossless rather than a guess about whether the last chunk "felt productive".

Options (core.options), all off by default: the order in which rows are
labelled (best stored-value match first, then cheap-model confidence), and the
cascade, which keeps a cheap label when its calibrated confidence reaches tau.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.measures import combinations, r_T_all
from core.query import COUNTER_EVIDENCE
from core.state import RunState


def _cheap_raw(state: RunState, attr: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Raw cheap-model scores for `rows`, from the run's cache when the row was
    already scored (sampling, ordering), else from ONE counted call."""
    model = state.cheap_models[attr]
    cache = state.cheap_raw.get(attr)
    ids = rows["row_id"].to_numpy()
    known = (np.isin(ids, cache.index.to_numpy()) if cache is not None
             else np.zeros(len(rows), dtype=bool))
    if (~known).any():
        fresh = model.predict_proba(rows.loc[~known], calibrated=False)
        fresh = fresh.set_axis(ids[~known], axis=0)
        cache = fresh if cache is None else pd.concat([cache, fresh])
        cache = cache[~cache.index.duplicated(keep="first")]
        state.cheap_raw[attr] = cache
    return cache.loc[ids].set_axis(rows.index, axis=0)


def _calibrated(model, raw: pd.DataFrame) -> pd.DataFrame:
    return model.calibrator.transform(raw) if model.calibrator is not None else raw


def _label(state: RunState, rows: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Attach predicted values. Published behaviour: the expensive model labels
    every row. With the cascade on, a row keeps the cheap label when the cheap
    model's calibrated probability is >= tau, and only the rest go to the
    expensive model. Every row then records
    who labelled it and with what confidence, so the answer's meaning is
    explicit: cheap labels are correct with probability >= tau."""
    out = rows.copy()
    if wanted:
        state.rows_labelled += len(rows)
    t0 = time.perf_counter()
    tau = state.options.cascade_tau
    min_conf = state.options.min_confidence
    for attr in wanted:
        cm = (state.cheap_models or {}).get(attr)
        if tau is None or cm is None or len(rows) == 0:
            state.note("expensive", row_ids=rows["row_id"].to_numpy())
            if min_conf is None or len(rows) == 0:
                out[attr] = state.models[attr].predict(rows)
                continue
            em = state.models[attr]
            raw_e = em.predict_proba(rows, calibrated=False)   # counted
            labels = raw_e.idxmax(axis=1).to_numpy().astype(object)
            cal_e = _calibrated(em, raw_e)
            conf = cal_e.to_numpy()[np.arange(len(cal_e)),
                                    [cal_e.columns.get_loc(l) for l in labels]]
            labels[conf < min_conf] = np.nan
            out[attr] = labels
            out[f"_by:{attr}"] = "expensive"
            out[f"_conf:{attr}"] = conf
            continue
        raw_c = _cheap_raw(state, attr, rows)
        cal_c = _calibrated(cm, raw_c)
        ok = (cal_c.max(axis=1) >= tau).to_numpy()
        labels = raw_c.idxmax(axis=1).to_numpy().astype(object)
        conf = np.array(cal_c.max(axis=1), dtype=float)        # writable copy
        if (~ok).any():
            state.note("expensive", row_ids=rows.loc[~ok, "row_id"].to_numpy())
            em = state.models[attr]
            raw_e = em.predict_proba(rows.loc[~ok], calibrated=False)   # counted
            labels[~ok] = raw_e.idxmax(axis=1).to_numpy()
            cal_e = _calibrated(em, raw_e)
            conf[~ok] = cal_e.to_numpy()[np.arange(len(cal_e)),
                                         [cal_e.columns.get_loc(l) for l in labels[~ok]]]
        if min_conf is not None:
            labels[conf < min_conf] = np.nan
        out[attr] = labels
        out[f"_by:{attr}"] = np.where(ok, "cheap", "expensive")
        out[f"_conf:{attr}"] = conf
        state.note("cascade", attr=attr, rows=len(rows), kept_cheap=int(ok.sum()),
                   tau=tau)
    state.t_inference += time.perf_counter() - t0
    if state.targets_diverse_gap:
        _remember_labelled(state, rows)
    return out


def _shown(state: RunState, attr: str) -> set:
    """Values of `attr` in T, plus those already labelled in the current source:
    labelling another row with the same stored values adds nothing."""
    T = state.T
    shown = (set(T[attr].dropna().unique())
             if T is not None and len(T) and attr in T.columns else set())
    return shown | state.seen.get(attr, set()) if state.targets_diverse_gap else shown


def _remember_labelled(state: RunState, rows: pd.DataFrame) -> None:
    source = state.visited[-1] if state.visited else None
    if state.seen_source != source:
        state.seen_source, state.seen = source, {}
    for a in state.Q_full.static:
        if a in rows.columns:
            state.seen.setdefault(a, set()).update(rows[a].dropna().unique())


def _could_diversify(state: RunState, rows: pd.DataFrame) -> pd.Series:
    """DIVERSE: TWR and DWR only grow when T gains a value it does not show
    yet, so a row can help only if it carries such a value on a query
    attribute. Predicted values are unknown before labelling: while T lacks
    some label of a predicted attribute, every row may bring it."""
    for a in state.Q_full.predicted:
        m = state.models.get(a)
        if m is None or not set(m.labels) <= _shown(state, a):
            return pd.Series(True, index=rows.index)
    keep = pd.Series(False, index=rows.index)
    for a in state.Q_full.static:
        if a in rows.columns:
            keep |= ~rows[a].isin(_shown(state, a))
    return keep


def _could_improve(state: RunState, rows: pd.DataFrame) -> pd.Series:
    """Rows whose static values leave room to beat T on some combination."""
    if state.semantics == COUNTER_EVIDENCE:
        return _could_diversify(state, rows)
    Q_all = state.Q.all
    cols = list(Q_all.keys())
    k = len(cols)
    n_pred = len(state.Q.predicted)
    rT = r_T_all(state.T, Q_all)

    keep = pd.Series(False, index=rows.index)
    for c in combinations(Q_all):
        level = pd.Series(0, index=rows.index, dtype=int)
        for col, v in zip(cols, c):
            if col in rows.columns and col not in state.Q.predicted:
                level = level + (rows[col] == v).astype(int)
        keep |= (level + n_pred).clip(upper=k) > rT[c]
        if keep.all():
            break
    if state.needs_evidence():
        # Still collecting evidence: a row whose stored values are all
        # requested may fully match a combination already witnessed — useful
        # now, even though it cannot raise r_T.
        every = pd.Series(True, index=rows.index)
        for a, vals in state.Q.static.items():
            every &= rows[a].isin(vals) if a in rows.columns else False
        keep |= every
    return keep


def _static_level(state: RunState, rows: pd.DataFrame) -> pd.Series:
    """How many requested STORED values a row carries — known before any model
    runs, so ordering by it is free."""
    lvl = pd.Series(0, index=rows.index, dtype=int)
    for a, vals in state.Q.static.items():
        if a in rows.columns:
            lvl = lvl + rows[a].isin(vals).astype(int)
    return lvl


def _diverse_priority(state: RunState, rows: pd.DataFrame) -> pd.Series:
    """DIVERSE order. Rows showing a requested stored value T still lacks come
    first, since TWR is part of the target, then rows showing the most
    unrequested values T does not have yet, which is what raises DWR and what
    CEprune prefers. Known before any model runs, so ordering by it is free."""
    static = state.Q_full.static
    missing = pd.Series(0, index=rows.index, dtype=int)
    fresh = pd.Series(0, index=rows.index, dtype=int)
    for a, vals in static.items():
        if a in rows.columns:
            have = _shown(state, a)
            missing = missing + rows[a].isin(set(vals) - have).astype(int)
            fresh = fresh + (~rows[a].isin(set(vals) | have)).astype(int)
    return missing * (len(static) + 1) + fresh


def _cheap_score(state: RunState, rows: pd.DataFrame) -> tuple[pd.Series, int]:
    """Calibrated cheap-model probability that a row carries a requested value
    on EVERY predicted attribute (independence across attributes). Rows already
    sampled reuse their stored probabilities; only the rest are scored, and
    those calls are counted against the cheap tier."""
    score = pd.Series(1.0, index=rows.index)
    newly = 0
    for attr, wanted in state.Q.predicted.items():
        model = (state.cheap_models or {}).get(attr)
        if model is None or not wanted:
            continue
        before = model.calls
        cal = _calibrated(model, _cheap_raw(state, attr, rows))
        newly += model.calls - before
        cols = [c for c in wanted if c in cal.columns]
        score = score * cal[cols].sum(axis=1)
    return score, newly


def _order_candidates(state: RunState, cands: pd.DataFrame,
                      quiet: bool = False) -> pd.DataFrame:
    """Order the rows of the chosen source before they are labelled.

    storage       as stored (published behaviour)
    static        most requested stored values first; ties keep storage order
    static+cheap  as static, and inside the TOP tier by the cheap model's
                  calibrated confidence. Lower tiers are not scored: they can
                  only matter once the top tier is spent.
    """
    mode = state.options.row_order
    if mode == "storage" or len(cands) == 0:
        return cands
    lvl = (_diverse_priority(state, cands) if state.semantics == COUNTER_EVIDENCE
           else _static_level(state, cands))
    key = pd.DataFrame({"lvl": lvl.to_numpy(), "pos": np.arange(len(cands)),
                        "score": 0.0}, index=cands.index)
    scored = 0
    if mode == "static+cheap" and state.cheap_models:
        top = lvl == lvl.max()
        sc, scored = _cheap_score(state, cands.loc[top])
        key.loc[top, "score"] = sc.to_numpy()
        key = key.sort_values(["lvl", "score", "pos"], ascending=[False, False, True])
    else:
        key = key.sort_values(["lvl", "pos"], ascending=[False, True])
    if not quiet:
        state.note("row_order", order=mode,
                   tiers={int(k): int(v) for k, v in
                          lvl.value_counts().sort_index(ascending=False).items()},
                   cheap_scored=scored)
    return cands.loc[key.index]


def _predict_incremental(state: RunState, rows: pd.DataFrame,
                         wanted: list[str]) -> pd.DataFrame:
    from boxes.check_target import goal_reached
    from boxes.prune import pool_of

    t0 = time.perf_counter()
    candidates = rows.loc[_could_improve(state, rows)]
    state.t_local += time.perf_counter() - t0
    state.rows_skipped += len(rows) - len(candidates)
    state.note("row_skip", retrieved=len(rows), candidates=len(candidates),
               skipped=len(rows) - len(candidates),
               reason=("no value T does not already show"
                       if state.semantics == COUNTER_EVIDENCE else
                       "static match level + |A_p| cannot beat r_T(c)"))
    if len(candidates) == 0:
        return rows.iloc[0:0]
    candidates = _order_candidates(state, candidates)

    Q_all = state.Q.all
    src_name = state.visited[-1] if state.visited else None
    W = state.T
    produced: list[pd.DataFrame] = []
    n = len(candidates)
    remaining = candidates
    start = 0
    verdicts, signature = None, None    # see below

    while len(remaining):
        if (start and state.options.budget is not None
                and state.cost >= state.options.budget):
            state.note("scan_stop", labelled=start, of_candidates=n,
                       why="budget spent")
            break
        chunk = remaining.iloc[:state.options.chunk]
        labelled = _label(state, chunk, wanted)
        produced.append(labelled)

        # Per-chunk bookkeeping. Eprune and target_met are called directly
        # rather than through the timed `prune` box, so they are timed here —
        # on a wide query this inner loop, not inference, dominates the clock.
        t1 = time.perf_counter()
        merged = pd.concat([W, labelled], ignore_index=True) if len(W) else labelled
        W = pool_of(state, merged, n_incumbent=len(W))
        reached = goal_reached(state, W)
        state.t_local += time.perf_counter() - t1

        # Reached the target: anything further is waste.
        if reached:
            state.note("scan_stop", labelled=start + len(chunk),
                       of_candidates=n, why="target reached")
            break

        # Otherwise: does anything LEFT in this source still have room to help?
        # Every labelled row raises some r_T(c), which lowers the ceiling that
        # the remaining rows must clear. Re-testing them against the updated T
        # is the same admissible bound used to skip rows in the first place, so
        # stopping when none survive is lossless — not a guess about whether
        # the last chunk "felt productive".
        rest = remaining.iloc[state.options.chunk:]
        if len(rest) == 0:
            state.note("scan_stop", labelled=n, of_candidates=n,
                       why="source consumed")
            break
        t2 = time.perf_counter()
        state.T, saved = W, state.T          # bound is relative to current T
        # The verdicts (and, under DIVERSE, the order) depend only on what T
        # shows so far. While that stays the same, re-testing every remaining
        # row after each chunk would change nothing and takes hours on a lake.
        now = (tuple(frozenset(_shown(state, a)) for a in state.Q_full.all)
               if state.semantics == COUNTER_EVIDENCE
               else tuple(r_T_all(state.T, state.Q.all).values()))
        if now == signature:
            still = verdicts.iloc[len(chunk):]
        else:
            still = _could_improve(state, rest)
            if state.semantics == COUNTER_EVIDENCE:
                rest = _order_candidates(state, rest, quiet=True)
                still = still.loc[rest.index]
            signature = now
        verdicts = still
        state.T = saved
        state.t_local += time.perf_counter() - t2
        if not bool(still.any()):
            state.plateau_stops += 1         # nothing left here can improve
            state.rows_skipped += len(rest)
            state.note("scan_stop", labelled=start + len(chunk), of_candidates=n,
                       skipped=len(rest),
                       why="no remaining row can clear the updated ceiling")
            break
        remaining = rest
        start += len(chunk)

    if state.semantics == COUNTER_EVIDENCE:
        state.scan_pool = W     # CEprune is greedy: re-pruning could lose the target
    return pd.concat(produced, ignore_index=True) if produced else rows.iloc[0:0]


def predict(state: RunState, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach predicted columns. Only attributes the query asks about are
    predicted — with Q_p = ∅ no model is invoked at all, which is what makes
    the pipeline reduce to plain TVD (spec §7 check #1)."""
    incremental = rows is None
    if rows is None:
        rows = state.retrieved
    wanted = [a for a in state.Q_full.predicted if a in state.models]
    if len(rows) == 0 or not wanted:
        return rows

    if incremental and state.incremental:
        return _predict_incremental(state, rows, wanted)
    return _label(state, rows, wanted)
