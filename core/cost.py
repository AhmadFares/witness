"""The single cost model: one number per run, from the counts it already keeps.

Prices are expressed in units of ONE expensive prediction, so the default
(cheap, requests and shipped rows free) reduces to the published headline:
total cost = expensive predictions. Every other setting is a deployment
assumption, chosen in the demo's sidebar rather than baked into the algorithm.

  total = expensive · (expensive predictions)
        + cheap     · (cheap predictions)
        + request   · (source requests: SQL queries, summaries, sample draws)
        + row       · (rows shipped: retrieved rows and sampled rows)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    expensive: float = 1.0
    cheap: float = 0.0
    request: float = 0.0
    row: float = 0.0

    def total(self, expensive: float, cheap: float, requests: float,
              rows: float) -> float:
        return (self.expensive * expensive + self.cheap * cheap
                + self.request * requests + self.row * rows)


def run_cost(prices: Prices, state) -> float:
    """Total cost of a run so far, read off its counters."""
    return prices.total(expensive=state.model_invocations,
                        cheap=state.cheap_invocations,
                        requests=state.sql_queries + state.sample_requests,
                        rows=state.rows_retrieved + state.sample_rows)
