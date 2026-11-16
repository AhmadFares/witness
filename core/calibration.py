"""Model-agnostic confidence: calibration and conformal prediction.

Raw confidence depends on the model — a tree's leaf frequencies, a boosted
model's probabilities, an LLM's token probabilities or the agreement of
repeated calls. This module only ever sees a matrix of per-class SCORES and
the true labels of a calibration set, so any model that can produce a label
and a score plugs in unchanged.

  Calibrator  maps raw per-class scores to calibrated probabilities with one
              isotonic regression per class (one-vs-rest), then renormalises.
              After it, "0.9" means right about 90% of the time for any model.

Calibration data must never come from the data lake's ground truth, which is
for evaluation only. The fitting helpers in setup.models use the training
split, through out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self) -> None:
        self.maps: dict = {}
        self.classes: list = []

    def fit(self, scores: pd.DataFrame, y: pd.Series) -> "Calibrator":
        self.classes = list(scores.columns)
        target = y.to_numpy()
        for c in self.classes:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(scores[c].to_numpy(dtype=float), (target == c).astype(float))
            self.maps[c] = ir
        return self

    def transform(self, scores: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {c: (self.maps[c].predict(scores[c].to_numpy(dtype=float))
                 if c in self.maps else scores[c].to_numpy(dtype=float))
             for c in scores.columns},
            index=scores.index)
        total = out.sum(axis=1)
        total = total.where(total > 0, 1.0)
        return out.div(total, axis=0)


def ece(probs: pd.DataFrame, y: pd.Series, bins: int = 10) -> float:
    """Expected calibration error of the top label."""
    if len(probs) == 0:
        return 0.0
    conf = probs.max(axis=1).to_numpy()
    pred = probs.idxmax(axis=1).to_numpy()
    acc = (pred == y.to_numpy()).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(total)


