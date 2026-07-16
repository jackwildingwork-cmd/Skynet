"""Global resource ceiling + event log (brief 2.8, 6).

The global budget is the impartial "sunlight" (brief 1.5): a fixed total set in
advance and NEVER increased mid-run. Every credit paid out to an agent (task
payouts) is drawn from this pool; when the pool is empty, no more payouts are
possible and the population must live on whatever it already holds. Occupancy
costs, maintenance, reproduction and sabotage costs are internal transfers /
sinks and are returned to the pool (they represent resource leaving an agent),
keeping total accounting closed.

The EventLog captures every logged quantity from tick 1 — never added
retroactively (brief 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, TextIO


class GlobalBudget:
    def __init__(self, total: float):
        self._total = total
        self._remaining = total
        self._paid_out = 0.0

    @property
    def total(self) -> float:
        return self._total

    @property
    def remaining(self) -> float:
        return self._remaining

    @property
    def remaining_frac(self) -> float:
        return self._remaining / self._total if self._total else 0.0

    def draw(self, amount: float) -> float:
        """Pay out up to `amount` from the pool; return what was actually paid
        (may be less if the pool is nearly empty). NEVER goes negative and is
        NEVER topped up."""
        paid = min(amount, self._remaining)
        paid = max(0.0, paid)
        self._remaining -= paid
        self._paid_out += paid
        return paid

    def return_to_pool(self, amount: float) -> None:
        """Resource that leaves an agent as a cost re-enters the impartial pool.
        This keeps the ceiling a genuine conservation law rather than an
        ever-growing money printer, but it is population-level and impartial —
        it is NOT a top-up to any individual agent."""
        if amount <= 0:
            return
        self._remaining = min(self._total, self._remaining + amount)


@dataclass
class EventLog:
    """Append-only structured log. Death records here are never mutated."""
    events: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, kind: str, tick: int, **fields: Any) -> None:
        self.events.append({"kind": kind, "tick": tick, **fields})

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    def write_jsonl(self, fh: TextIO) -> None:
        for e in self.events:
            fh.write(json.dumps(e, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    # Enums, dataclasses, etc.
    if hasattr(o, "value"):
        return o.value
    return str(o)
