"""Models, wrapped so EVERY invocation is counted.

Dataset-agnostic and multi-class: a lake declares a ModelSpec per predicted
attribute (which columns the model may read, and which column it produces),
and this module trains and wraps them.

Two tiers:
  CountingModel  the EXPENSIVE predictor. Its invocation count is the paper's
                 headline metric, and it is the only source of values that
                 enter T.
  CheapModel     a deliberately weak, near-free predictor used ONLY to sample
                 sources during selection. Its labels never enter T; its calls
                 are counted separately so the headline stays honest and the
                 cheap traffic stays visible.

Ground truth is not available to either — it lives outside the pipeline, for
evaluation only.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from setup.config import SEED

# Gradient boosting predicts tree by tree, each tree across parallel OpenMP
# threads. Starting the threads costs ~0.8 s per call whatever the batch size,
# so the 50-row chunks PTVD labels ran ~47x slower than on one thread (MovieLens
# genre: 18 classes x 300 rounds = 5,400 trees). Below this many rows one thread
# is faster; above it the threads win. Labels are identical either way.
PARALLEL_FROM = 10_000
_controller = None


def _threads_for(n_rows: int):
    global _controller
    if n_rows >= PARALLEL_FROM:
        return nullcontext()
    if _controller is None:
        from threadpoolctl import ThreadpoolController
        _controller = ThreadpoolController()
    return _controller.limit(limits=1, user_api="openmp")


@dataclass
class ModelSpec:
    """How to build a model for one predicted attribute."""
    target: str                     # the column it produces
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    cheap_cols: list[str] = field(default_factory=list)  # columns the cheap model reads
    cheap_depth: int = 1            # kept shallow: the cheap tier must stay cheap
    max_depth: int = 8
    entity: str | None = None       # predictions belong to an entity (e.g.
                                    # movie_id): one call per entity, reused
                                    # for its other rows
    kind: str = "tree"              # "tree" (light) | "gb" (gradient boosting —
                                    # hundreds of trees, so inference genuinely
                                    # dominates and the counted call is EXPENSIVE)

    @property
    def features(self) -> list[str]:
        return self.num_cols + self.cat_cols


def _preprocessor(spec: ModelSpec) -> ColumnTransformer:
    steps = []
    if spec.num_cols:
        steps.append(("num", StandardScaler(), spec.num_cols))
    if spec.cat_cols:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False), spec.cat_cols))
    return ColumnTransformer(steps)


@dataclass
class CountingModel:
    """predict(rows) -> labels. One invocation per row. Multi-class."""
    attribute: str
    estimator: Pipeline
    features: list[str]
    labels: tuple
    calls: int = 0
    calibrator: object = None       # core.calibration.Calibrator
    calibration_report: dict = field(default_factory=dict)
    entity: str | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if len(rows) == 0:
            return np.empty(0, dtype=object)
        if self.entity is not None:
            return np.array(self.labels, dtype=object)[
                _entity_proba(self, rows).argmax(axis=1)]
        self.calls += len(rows)
        with _threads_for(len(rows)):
            return self.estimator.predict(rows[self.features])

    def predict_proba(self, rows: pd.DataFrame,
                      calibrated: bool = True) -> pd.DataFrame:
        """Per-class probabilities, one invocation per row (counted exactly
        like predict). The argmax equals predict(), so a caller that needs
        both the label and its confidence pays once."""
        if len(rows) == 0:
            return pd.DataFrame(columns=list(self.labels), dtype=float)
        if self.entity is not None:
            raw = _entity_proba(self, rows)
        else:
            self.calls += len(rows)
            with _threads_for(len(rows)):
                raw = self.estimator.predict_proba(rows[self.features])
        P = pd.DataFrame(raw, index=rows.index, columns=list(self.labels))
        if calibrated and self.calibrator is not None:
            P = self.calibrator.transform(P)
        return P

    def reset(self) -> None:
        self.calls = 0
        self._cache.clear()


@dataclass
class CheapModel:
    """Depth-1 stump on a single column — one comparison per row."""
    attribute: str
    estimator: DecisionTreeClassifier
    features: list[str]
    labels: tuple
    calls: int = 0
    calibrator: object = None       # core.calibration.Calibrator
    calibration_report: dict = field(default_factory=dict)
    entity: str | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if len(rows) == 0:
            return np.empty(0, dtype=object)
        if self.entity is not None:
            return np.array(self.labels, dtype=object)[
                _entity_proba(self, rows).argmax(axis=1)]
        self.calls += len(rows)
        with _threads_for(len(rows)):
            return self.estimator.predict(rows[self.features])

    def predict_proba(self, rows: pd.DataFrame,
                      calibrated: bool = True) -> pd.DataFrame:
        """Per-class probabilities, one invocation per row (counted exactly
        like predict). The argmax equals predict(), so a caller that needs
        both the label and its confidence pays once."""
        if len(rows) == 0:
            return pd.DataFrame(columns=list(self.labels), dtype=float)
        if self.entity is not None:
            raw = _entity_proba(self, rows)
        else:
            self.calls += len(rows)
            with _threads_for(len(rows)):
                raw = self.estimator.predict_proba(rows[self.features])
        P = pd.DataFrame(raw, index=rows.index, columns=list(self.labels))
        if calibrated and self.calibrator is not None:
            P = self.calibrator.transform(P)
        return P

    def reset(self) -> None:
        self.calls = 0
        self._cache.clear()


KINDS = {
    "gb":    "gradient boosting, ~300 trees per prediction (expensive)",
    "tree":  "single decision tree, depth 8 (middling)",
    "stump": "shallow tree on a few raw numeric columns (near-free)",
}


def _entity_proba(model, rows: pd.DataFrame) -> np.ndarray:
    """Raw class probabilities when a prediction belongs to an entity (one
    movie, many ratings): each entity is predicted once and counted once in
    a run; its other rows reuse the result."""
    keys = rows[model.entity].to_numpy()
    new = [k for k in pd.unique(keys) if k not in model._cache]
    if new:
        first = rows.drop_duplicates(model.entity).set_index(model.entity).loc[new]
        model.calls += len(new)
        with _threads_for(len(first)):
            raw = model.estimator.predict_proba(first[model.features])
        model._cache.update(zip(new, raw))
    return np.vstack([model._cache[k] for k in keys])


def _labels(est) -> tuple:
    clf = est.named_steps["clf"] if isinstance(est, Pipeline) else est
    return tuple(clf.classes_)


def _fit(df_train: pd.DataFrame, spec: ModelSpec, kind: str):
    """Train one estimator for `spec` and return (estimator, feature columns).

    The kind is independent of the role: any of these can be put in the
    expensive or the cheap slot, which is what lets the demo vary the gap
    between the two tiers. Roles must hold DISTINCT instances, since each
    object carries its own call counter.
    """
    if kind == "gb":
        # Gradient boosting: ~300 trees per prediction. Ordinal-code the
        # categoricals and tell HistGB which columns they are.
        pre = ColumnTransformer(
            ([("num", "passthrough", spec.num_cols)] if spec.num_cols else [])
            + ([("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1), spec.cat_cols)]
               if spec.cat_cols else []))
        mask = [False] * len(spec.num_cols) + [True] * len(spec.cat_cols)
        clf = HistGradientBoostingClassifier(
            max_iter=300, categorical_features=mask, random_state=SEED)
        est = Pipeline([("pre", pre), ("clf", clf)]).fit(
            df_train[spec.features], df_train[spec.target])
        return est, spec.features

    if kind == "tree":
        est = Pipeline([
            ("pre", _preprocessor(spec)),
            ("clf", DecisionTreeClassifier(max_depth=spec.max_depth,
                                           random_state=SEED)),
        ]).fit(df_train[spec.features], df_train[spec.target])
        return est, spec.features

    if kind == "stump":
        # A few NUMERIC columns, shallow tree, no preprocessing. It must be weak
        # but not useless: a model that always returns the majority class gives
        # the same rate in every source, so selection would be blind. It stays
        # cheap by reading raw numeric columns with no scaling or one-hot
        # encoding, and by being far shallower than the expensive model.
        cols = spec.cheap_cols or spec.num_cols[:1] or spec.features[:1]
        cols = [c for c in cols if c in df_train.columns and
                df_train[c].dtype != object] or spec.num_cols[:1]
        est = DecisionTreeClassifier(max_depth=spec.cheap_depth,
                                     random_state=SEED).fit(
            df_train[cols], df_train[spec.target])
        return est, cols

    raise ValueError(f"unknown model kind {kind!r}; expected one of {list(KINDS)}")


def train_model(df_train: pd.DataFrame, spec: ModelSpec,
                kind: str | None = None) -> CountingModel:
    """The EXPENSIVE role. Defaults to the spec's own kind, so experiment
    numbers are unaffected by the demo's ability to override it."""
    est, cols = _fit(df_train, spec, kind or spec.kind)
    return CountingModel(attribute=spec.target, estimator=est, features=cols,
                         labels=_labels(est), entity=spec.entity)


def train_cheap_model(df_train: pd.DataFrame, spec: ModelSpec,
                      kind: str = "stump") -> CheapModel:
    """The CHEAP role — sampling only; its labels never enter T."""
    est, cols = _fit(df_train, spec, kind)
    return CheapModel(attribute=spec.target, estimator=est, features=cols,
                      labels=_labels(est), entity=spec.entity)


def build_models(df_train: pd.DataFrame, specs: list[ModelSpec],
                 kinds: dict[str, str] | None = None) -> dict:
    """`kinds` maps a predicted attribute to a model kind, overriding the spec."""
    kinds = kinds or {}
    return {s.target: train_model(df_train, s, kinds.get(s.target))
            for s in specs}


def build_cheap_models(df_train: pd.DataFrame, specs: list[ModelSpec],
                       kinds: dict[str, str] | None = None) -> dict:
    kinds = kinds or {}
    return {s.target: train_cheap_model(df_train, s,
                                        kinds.get(s.target, "stump"))
            for s in specs}


def fit_confidence(model, df_train: pd.DataFrame, spec: ModelSpec, kind: str,
                   folds: int = 3, cap: int = 20000) -> dict:
    """Attach a Calibrator to `model`.

    Scores come from OUT-OF-FOLD predictions on the training split: each fold
    is predicted by a model of the same kind trained on the other folds, so
    the calibration data is never seen by the model that scores it, and the
    deployed model (trained on the full split) is left untouched. The data
    lake's ground truth is never used. `cap` bounds the cost on large splits.
    """
    from sklearn.model_selection import KFold
    from core.calibration import Calibrator, ece

    sub = (df_train.sample(n=cap, random_state=SEED)
           if len(df_train) > cap else df_train)
    labels = list(model.labels)
    oof = pd.DataFrame(0.0, index=sub.index, columns=labels)
    for tr, te in KFold(n_splits=folds, shuffle=True,
                        random_state=SEED).split(sub):
        est, cols = _fit(sub.iloc[tr], spec, kind)
        P = est.predict_proba(sub.iloc[te][cols])
        for j, c in enumerate(_labels(est)):
            if c in oof.columns:
                oof.iloc[te, oof.columns.get_loc(c)] = P[:, j]
    y = sub[spec.target]
    # Honest calibration error: fit on 70% of the out-of-fold rows, measure on
    # the other 30%. The calibrator kept for use is then refitted on all rows.
    rng = np.random.default_rng(SEED)
    held = rng.random(len(sub)) < 0.3
    probe = Calibrator().fit(oof[~held], y[~held])
    ece_held = ece(probe.transform(oof[held]), y[held]) if held.any() else float("nan")
    cal = Calibrator().fit(oof, y)
    calp = cal.transform(oof)
    model.calibrator = cal
    model.calibration_report = {
        "rows": len(sub),
        "accuracy": float((calp.idxmax(axis=1) == y).mean()),
        "ece_raw": ece(oof, y),
        "ece_calibrated": ece_held,     # measured on held-out rows
        "share_conf_0.9": float((calp.max(axis=1) >= 0.9).mean()),
        "fit_row_ids": (set(sub["row_id"].tolist())
                        if "row_id" in sub.columns else set()),
    }
    return model.calibration_report
