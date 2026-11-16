"""What the demo can offer: lakes, predictors, variants, label spaces.

Pure Python — no Streamlit import — so the same catalog is usable from a
notebook or a script. The UI layer owns in-memory caching; trained predictors
are saved to disk here, so a restart reloads them instead of retraining.

Nothing here computes a result; it only describes what exists, so that adding
a dataset or a model kind is a one-line change visible in the interface.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from dataclasses import replace

from core.options import Options
from core.query import COUNTER_EVIDENCE, EVIDENCE
from setup.models import build_cheap_models, build_models, fit_confidence

# dataset -> (module, build_lake kwargs). Two datasets common in the
# community: ACSIncome (the Census successor to Adult) and MovieLens 25M joined
# with IMDb. Adult, Covertype and MovieLens 1M remain in setup/ for the
# experiments and checks, but are no longer offered by the demo.
DATASETS = {
    "acs-income":         ("setup.acs", {}),
    "movielens-25m-imdb": ("setup.movielens25", {}),
}

# How each dataset is cut into sources, shown under its name in the UI.
SPLITS = {
    "acs-income":         "Split by state × class of worker (public, private, self-employed…)",
    "movielens-25m-imdb": "Split by year of rating × decade of release",
}

# What one source of each dataset is, in a sentence, and its readable name.
SOURCE_HINTS = {
    "acs-income": "A source is one table: the census records "
                  "of one US state for one type of employer.",
    "movielens-25m-imdb": "A source is one table: the ratings "
                          "given in one year to films of one decade.",
}


def source_axes(dataset: str, name: str) -> tuple[str, str]:
    """Where a source sits on its split's grid: (column, row)."""
    if dataset == "movielens-25m-imdb" and " / " not in name:
        year, decade = name.split("_", 1)
        return year, decade
    first, _, second = name.partition(" / ")
    return first, second


def split_options(dataset: str) -> list[str]:
    """The ways this dataset can be cut into sources, the default first."""
    ds = importlib.import_module(DATASETS[dataset][0])
    return [ds.DEFAULT_SPLIT] + [s for s in ds.SPLITS if s != ds.DEFAULT_SPLIT]


def split_axes(split: str) -> tuple[str, str]:
    """A split's two axes, as named in the interface: (column, row)."""
    first, _, second = split.partition(" x ")
    return first, second


def source_label(dataset: str, name: str) -> str:
    if dataset == "movielens-25m-imdb" and " / " not in name:
        year, decade = name.split("_", 1)
        return f"{year} ratings of {decade} films"
    return name.replace(" / ", ", ")


# The cost of a run: x · cheap calls + y · expensive calls + z · requests to
# tables. Default prices are orders of magnitude, editable in the interface.
COST_UNITS = {
    "Cost ($)": dict(
        x=0.000000001, y=0.00016, column="total ($)", fmt="%.9f",
        z={"Paid API: $0.005 per call": 0.005,
           "Sold or approved access: $0.50 per request (illustrative)": 0.5},
        note="y: one LLM call (about 1,000 input tokens at $0.15 per million). "
             "x: a small local model, close to free but never zero. z: depends "
             "on the kind of source."),
    "Time (seconds)": dict(
        x=0.00001, y=2.5, column="total (s)", fmt="%.6f",
        z={"Fast network: 0.06 s per request (measured)": 0.06,
           "Restricted: 1 minute per request (illustrative)": 60.0},
        note="y: one call to Llama 3.1 (8B) on CPU, measured. x: a small local "
             "model, which takes microseconds. z: depends on the kind of source."),
}

# Semantics, under the names the paper uses.
SEMANTICS = {
    "Associative": EVIDENCE,
    "Diverse": COUNTER_EVIDENCE,
}

# Search variants. The statistics regime is a SEPARATE, orthogonal control —
# any variant can be run under any regime, which is the comparison that makes
# the incomplete-statistics case interesting.
VARIANTS = {
    "PTVD": dict(mode="PTVD"),
    "PTVD — no incremental consumption": dict(mode="PTVD", incremental=False),
    "Static selection (histogram-only)": dict(mode="PredictEachSource"),
    # PTVD's labelling (in chunks, stop at theta) with sources chosen from
    # statistics alone: the ablation that isolates what cheap sampling adds.
    "PTVD — no cheap sampling": dict(mode="PredictEachSource", incremental=True),
    "Exhaustive (predict everything)": dict(mode="PredictEverything"),
    "Random source order": dict(mode="random"),
    "Proxy ranking (whole lake)": dict(mode="Proxy", incremental=True),
}

# Only PTVD variants get a cheap tier; baselines must stay dumb or the
# comparison stops isolating selection.
USES_CHEAP = {"PTVD",
              "PTVD — no incremental consumption"}

# What each variant actually does. Shown in the UI, because "which baseline is
# this?" is the first question anyone asks of the comparison.
VARIANT_HELP = {
    "PTVD":
        "Picks one table at a time. It asks a few candidate tables for exact counts "
        "of the stored conditions and for a small sample, labelled by the cheap "
        "model, and prefers the table where matching rows should come fastest. "
        "Under Diverse it needs no sample: the stored statistics show which table "
        "closes what the target still lacks. It labels 10 rows at a time and stops "
        "as soon as the target is reached.",
    "Proxy ranking (whole lake)":
        "Proxy-based query processing in the style of SUPG: scans every table, "
        "ranks all rows by stored match and then by the cheap model's score, and "
        "labels them in that order with the expensive model, 10 at a time, until "
        "the target is reached.",
    "Random source order":
        "Scans tables in random order and labels every row it gets, until the "
        "target is reached.",
    "Exhaustive (predict everything)":
        "Scans every table and labels every row that matches at least one stored "
        "condition: the cost of not choosing at all.",
    "PTVD — no cheap sampling":
        "Witness, but tables are chosen from the stored statistics only, without "
        "the cheap model.",
    "PTVD — no incremental consumption":
        "Witness, but a chosen table is labelled in full instead of 10 rows at a "
        "time.",
    "Static selection (histogram-only)":
        "Tables chosen from the stored statistics only, and every row of a chosen "
        "table labelled. Removes both of Witness's components.",
}

NO_SAMPLING = "PTVD — no cheap sampling"
NO_CHUNKS = "PTVD — no incremental consumption"
TVD_STYLE = "Static selection (histogram-only)"

# What each run is for. PTVD always runs. Competitors are other ways to answer
# the same query; ablations remove one of PTVD's ideas to show what it is
# worth (the TVD-style static selection removes both: no cheap sampling, and
# every row of a chosen source is labelled).
PROXY = "Proxy ranking (whole lake)"
COMPETITORS = [PROXY, "Random source order", "Exhaustive (predict everything)"]
ABLATIONS = [NO_SAMPLING, NO_CHUNKS, TVD_STYLE]
ROLE = {"PTVD": "Witness", **{v: "competitor" for v in COMPETITORS},
        **{v: "ablation" for v in ABLATIONS}}
# Which options make sense for which variant. The pure baselines stay as
# published, so the comparison keeps isolating what each mechanism buys.
CHUNKED = {"PTVD", NO_SAMPLING, "Proxy ranking (whole lake)"}
STATIC_SELECTION = {"Static selection (histogram-only)", NO_SAMPLING}


def options_for(variant: str, options: Options) -> Options:
    """The options actually applied to one variant.

    Labelling options (row order, cascade) apply to variants that label in
    chunks; sampling to the PTVD variants, which are the only ones that sample; the static scoring to every variant that scores
    sources. The answer shape and the budget define the problem itself, so
    they apply to all.
    """
    o = options
    if variant not in CHUNKED:
        o = replace(o, row_order="storage", cascade_tau=None)
    if variant not in USES_CHEAP:
        o = replace(o, sampling="uniform")
    if variant not in USES_CHEAP | STATIC_SELECTION:
        o = replace(o, static_scoring="default")
    if variant == "Proxy ranking (whole lake)":   # ranking by the cheap model IS the method
        o = replace(o, row_order="static+cheap")
    return o


def needs_cheap(variant: str, options: Options) -> bool:
    return variant in USES_CHEAP or options_for(variant, options).needs_cheap



@dataclass
class Lake:
    """A loaded dataset: sources, training split, and its module's declarations."""
    name: str
    df_train: pd.DataFrame
    sources: dict[str, pd.DataFrame]
    truth: dict
    static_attrs: list[str]
    predicted_attrs: list[str]
    specs: list
    static_values: dict[str, list] = field(default_factory=dict)
    # row_id -> source name. row_id is global and unique across sources, so
    # this recovers exactly where every tuple of an answer came from, without
    # the pipeline having to carry provenance itself.
    row_source: dict[int, str] = field(default_factory=dict)
    split: str = ""         # how the dataset was cut into these sources

    @property
    def n_rows(self) -> int:
        return sum(len(s) for s in self.sources.values())


def load_lake(name: str, split: str | None = None, max_values: int = 40) -> Lake:
    """Build a lake and index the static values the query builder can offer."""
    mod_name, kw = DATASETS[name]
    ds = importlib.import_module(mod_name)
    split = split or ds.DEFAULT_SPLIT
    df_train, sources, truth = ds.build_lake(**kw, split=split)

    # Distinct static values, ordered by frequency so the common ones are first.
    lake = pd.concat(sources.values(), ignore_index=True)
    values = {}
    for a in ds.STATIC_ATTRS:
        if a in lake.columns:
            values[a] = lake[a].value_counts().index.tolist()[:max_values]

    row_source = {int(rid): src for src, df in sources.items()
                  for rid in df["row_id"]}

    return Lake(name=name, df_train=df_train, sources=sources, truth=truth,
                static_attrs=list(ds.STATIC_ATTRS),
                predicted_attrs=list(ds.PREDICTED_ATTRS),
                specs=list(ds.SPECS), static_values=values,
                row_source=row_source, split=split)


def build_tiers(lake: Lake, expensive_kind: str, cheap_kind: str) -> tuple[dict, dict]:
    """Train one predictor per predicted attribute for each role.

    The two roles always hold DISTINCT instances, even when the same kind is
    chosen for both: each model object carries its own call counter, so sharing
    one would make the expensive and cheap tallies count the same calls twice.
    """
    expensive = build_models(
        lake.df_train, lake.specs,
        kinds={s.target: expensive_kind for s in lake.specs})
    cheap = build_cheap_models(
        lake.df_train, lake.specs,
        kinds={s.target: cheap_kind for s in lake.specs})
    # Model-agnostic confidence for both tiers, fitted on out-of-fold
    # predictions from the TRAINING split (never the lake's ground truth).
    for spec in lake.specs:
        fit_confidence(expensive[spec.target], lake.df_train, spec, expensive_kind)
        fit_confidence(cheap[spec.target], lake.df_train, spec, cheap_kind)
    return expensive, cheap


def speed_ratio(expensive: dict, cheap: dict, lake: "Lake") -> float:
    """Measured expensive:cheap time per prediction on this lake's rows. The
    default price of a cheap prediction is 1 / this ratio."""
    from experiments.run_baselines import measure_speed_ratio
    rows = pd.concat(list(lake.sources.values()), ignore_index=True)
    return measure_speed_ratio(expensive, cheap, rows)


# One file per (dataset, expensive kind, cheap kind). Delete the folder to
# force retraining.
SAVED_MODELS = Path(__file__).resolve().parent / "saved_models"


def _fingerprint(name: str, expensive_kind: str, cheap_kind: str) -> str:
    """Everything the saved models depend on: the code that builds the dataset
    and trains and calibrates the models, the settings, and the library
    versions. If any of it changes, the saved file stops matching and the
    models are retrained, so a stale model is never loaded."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for f in sorted((root / "setup").glob("*.py")) + [root / "core" / "calibration.py"]:
        h.update(f.read_bytes())
    h.update(repr((name, DATASETS[name], expensive_kind, cheap_kind,
                   sklearn.__version__, sys.version_info[:2])).encode())
    return h.hexdigest()


def load_or_build_tiers(lake: Lake, expensive_kind: str,
                        cheap_kind: str) -> tuple[dict, dict, float, str]:
    """Both tiers and their speed ratio: reloaded from disk when a matching
    file exists, otherwise trained, timed and saved for the next start.

    The speed ratio is saved with the models, so the default price of a cheap
    prediction (and with it every total cost) is the same from one day to the
    next instead of moving with the machine's load.

    Returns (expensive, cheap, speed ratio, where the models came from).
    """
    path = SAVED_MODELS / f"{lake.name}__{expensive_kind}__{cheap_kind}.joblib"
    key = _fingerprint(lake.name, expensive_kind, cheap_kind)
    if path.exists():
        try:
            saved = joblib.load(path)
            if saved["fingerprint"] == key:
                return (saved["expensive"], saved["cheap"], saved["speed"],
                        f"Saved models, trained {saved['trained_at']}")
        except Exception:
            pass                     # unreadable: retrain and overwrite below
    expensive, cheap = build_tiers(lake, expensive_kind, cheap_kind)
    speed = speed_ratio(expensive, cheap, lake)
    for m in (*expensive.values(), *cheap.values()):
        m.reset()                    # timing calls are not part of any run
    trained_at = datetime.now().strftime("%d %b %Y, %H:%M")
    SAVED_MODELS.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(dict(fingerprint=key, trained_at=trained_at, expensive=expensive,
                     cheap=cheap, speed=speed), tmp)
    os.replace(tmp, path)            # never leave a half-written file behind
    return expensive, cheap, speed, f"Trained {trained_at} and saved"


def _prediction_file(name: str, expensive_kind: str) -> Path:
    return SAVED_MODELS / f"{name}__{expensive_kind}__predictions.npz"


def build_saved_predictions(name: str, expensive_kind: str, cheap_kind: str) -> Path:
    """Compute the expensive model on every row once, for every split. Run once
    per trained model. The model is called ten rows at a time during a search,
    and most of the time of such a call is overhead, not prediction."""
    lakes = [load_lake(name, split) for split in split_options(name)]
    expensive, _, _, _ = load_or_build_tiers(lakes[0], expensive_kind, cheap_kind)
    rows = (pd.concat([s for lake in lakes for s in lake.sources.values()], ignore_index=True)
            .drop_duplicates("row_id"))
    out = {"fingerprint": np.array(_fingerprint(name, expensive_kind, cheap_kind))}
    for attr, model in expensive.items():
        if model.entity is not None:
            continue
        raw = model.estimator.predict_proba(rows[model.features])
        table = np.full((int(rows.row_id.max()) + 1, raw.shape[1]), np.nan)
        table[rows.row_id.to_numpy(dtype=np.int64)] = raw
        out[attr] = table
    path = _prediction_file(name, expensive_kind)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return path


def _serve_from_table(model, table: np.ndarray) -> None:
    """Answer from the stored predictions. The numbers are the model's own and
    every row still counts as one call; anything not stored goes to the model."""
    slow_proba, slow_predict = model.predict_proba, model.predict
    labels = np.array(list(model.labels), dtype=object)

    def stored(rows: pd.DataFrame):
        if len(rows) == 0 or "row_id" not in rows.columns:
            return None
        ids = rows["row_id"].to_numpy(dtype=np.int64)
        if ids.max() >= len(table):
            return None
        raw = table[ids]
        return None if np.isnan(raw).any() else raw

    def predict_proba(rows: pd.DataFrame, calibrated: bool = True) -> pd.DataFrame:
        raw = stored(rows)
        if raw is None:
            return slow_proba(rows, calibrated=calibrated)
        model.calls += len(rows)
        P = pd.DataFrame(raw, index=rows.index, columns=list(model.labels))
        if calibrated and model.calibrator is not None:
            P = model.calibrator.transform(P)
        return P

    def predict(rows: pd.DataFrame) -> np.ndarray:
        raw = stored(rows)
        if raw is None:
            return slow_predict(rows)
        model.calls += len(rows)
        return labels[raw.argmax(axis=1)]

    model.predict_proba, model.predict = predict_proba, predict


def attach_saved_predictions(name: str, expensive_kind: str, cheap_kind: str,
                             expensive: dict) -> bool:
    """Use the stored predictions when they match these models."""
    path = _prediction_file(name, expensive_kind)
    if not path.exists():
        return False
    with np.load(path) as data:
        if str(data["fingerprint"]) != _fingerprint(name, expensive_kind, cheap_kind):
            return False
        for attr, model in expensive.items():
            if attr in data.files:
                _serve_from_table(model, data[attr])
    return True


def label_space(models: dict, attr: str) -> list:
    """The labels a predicted attribute can take, from the MODEL.

    Deliberately not from ground truth: predicted attributes are stripped from
    the sources by design and the truth is evaluation-only, so reading it here
    would make the query builder a back door into the answer.
    """
    m = models.get(attr)
    return list(m.labels) if m is not None else []
