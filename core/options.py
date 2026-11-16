"""Algorithm options — every mechanism beyond the published PTVD is a switch.

Every switch is off by default except `row_order`, which defaults to
"static" since 2026-10-30: ordering by stored values is free and removes
losses that depend only on how rows happen to be stored (ablation in
STATUS.md). Runs before that date used "storage". Each switch is reachable
from the demo's sidebar and from the benchmark.

  row_order       order in which candidate rows of a chosen source are labelled
                    storage       as stored (runs before 2026-10-30)
                    static        best stored-value match first (default, free)
                    static+cheap  then, inside the top tier, by the cheap
                                  model's calibrated confidence
  source_score    how PTVD ranks the sources it sampled (Associative)
                    density       expected gain per expected expensive label:
                                  go where matching rows come fastest (default)
                    gain          expected gain alone (runs before 2026-11-09)
  sample_rates    how the cheap sample becomes a rate
                    probabilities  average of the cheap model's calibrated
                                   probabilities (default)
                    labels         count of its most likely label, smoothed
                                   (runs before 2026-11-09)
  patience        DIVERSE, PTVD: tables in a row that add less than eps before
                  the run stops (TVD's rule is 1)
  chunk           how many rows the expensive model labels between two checks
                  of the target (50 for runs before 2026-11-09)
  sampling        how candidate sources are sampled
                    uniform       uniformly at random (published behaviour)
                    stratified    over-sample rows with a high stored-value
                                  match, re-weighted so rates stay unbiased
  cascade_tau     keep a cheap label instead of calling the expensive model
                  when its confidence is at least tau (None = never)
  min_confidence  accept the expensive model's label as evidence only when its
                  calibrated confidence is at least this; below it the value
                  stays unknown and matches nothing (None = accept every label)
  answer          what the answer must satisfy
                    theta         reach the thresholds (published behaviour)
                    theta+k       reach the thresholds with at most k rows
                    theta+min_k   reach the thresholds AND show at least k rows:
                                  keep collecting rows that fully match a
                                  requested combination until there are k
                                  (Associative semantics)
  budget          stop once the run's total cost reaches it (None = no limit)
  static_scoring  how the STORED part of a source is scored
                    default       each mode as published: statistics for the
                                  static selector, exact counts for PTVD
                    statistics    precomputed histograms (TVD)
                    counts        exact counts fetched per source (STVD)
"""

from __future__ import annotations

from dataclasses import dataclass

ROW_ORDERS = ("storage", "static", "static+cheap")
SAMPLINGS = ("uniform", "stratified")
SOURCE_SCORES = ("density", "gain")
SAMPLE_RATES = ("probabilities", "labels")
ANSWERS = ("theta", "theta+k", "theta+min_k")
STATIC_SCORINGS = ("default", "statistics", "counts")


@dataclass(frozen=True)
class Options:
    row_order: str = "static"
    source_score: str = "density"
    chunk: int = 10
    sample_rates: str = "probabilities"
    patience: int = 3
    sampling: str = "uniform"
    cascade_tau: float | None = None
    min_confidence: float | None = None
    answer: str = "theta"
    k: int | None = None
    budget: float | None = None
    static_scoring: str = "default"

    def __post_init__(self) -> None:
        for name, allowed in (("row_order", ROW_ORDERS), ("sampling", SAMPLINGS),
                              ("source_score", SOURCE_SCORES),
                              ("sample_rates", SAMPLE_RATES),
                              ("answer", ANSWERS),
                              ("static_scoring", STATIC_SCORINGS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}; "
                                 f"expected one of {allowed}")
        if self.answer in ("theta+k", "theta+min_k") and not (self.k and self.k > 0):
            raise ValueError(f"answer={self.answer!r} needs a positive k")
        if self.chunk < 1 or self.patience < 1:
            raise ValueError("chunk and patience must be positive")
        if self.cascade_tau is not None and not 0.0 < self.cascade_tau <= 1.0:
            raise ValueError("cascade_tau must be in (0, 1]")
        if self.min_confidence is not None and not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")

    @property
    def needs_cheap(self) -> bool:
        """Whether these options call the cheap model outside sampling."""
        return self.row_order == "static+cheap" or self.cascade_tau is not None

    @property
    def is_default(self) -> bool:
        return self == Options()
