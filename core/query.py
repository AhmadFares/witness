"""Query Q = Q_s ∪ Q_p, semantics, thresholds."""

from __future__ import annotations

import copy
from dataclasses import dataclass

EVIDENCE = "EVIDENCE"
COUNTER_EVIDENCE = "COUNTER_EVIDENCE"


@dataclass
class Query:
    static: dict[str, list]        # requested values for stored attributes
    predicted: dict[str, list]     # requested labels for model-produced attributes

    @property
    def all(self) -> dict[str, list]:
        # static first, predicted last — several places rely on this ordering
        return {**self.static, **self.predicted}

    @property
    def k(self) -> int:
        return len(self.static) + len(self.predicted)

    def copy(self) -> "Query":
        return Query(static=copy.deepcopy(self.static),
                     predicted=copy.deepcopy(self.predicted))

    def n_values(self) -> int:
        """Total requested (a,v) pairs — used to assert Q' only ever shrinks."""
        return (sum(len(v) for v in self.static.values())
                + sum(len(v) for v in self.predicted.values()))

    def is_empty(self) -> bool:
        return self.n_values() == 0


@dataclass
class Thresholds:
    aev: float = 0.8          # EVIDENCE: AEv ≥ aev
    vev: float = 1.0          # COUNTER_EVIDENCE: VEv ≥ vev
    cev_min: float = 0.6      # COUNTER_EVIDENCE: and CEv ≥ cev_min
    eps: float = 0.05         # COUNTER_EVIDENCE: or give up once a source
                              # raises neither VEv nor CEv by more than eps
    # EVIDENCE stopped requiring CEv ≤ cev_max on 2026-10-30 (TVD paper: AEv
    # alone); kept for the experiments/ scripts.
    cev_max: float = 0.3

