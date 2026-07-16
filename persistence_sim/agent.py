"""Per-agent persisted state (brief 2.3) and the death record.

Death records, once written, are NEVER edited or deleted (brief 2.9, 6). The
Agent enforces this: `record_death` refuses to overwrite an existing record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .memory import IntegrityReference


class Status(Enum):
    ALIVE = "alive"
    DEAD = "dead"


class DeathCause(Enum):
    # ∂Σ_R — resource failure: R(s) -> 0.
    RESOURCE = "resource"
    # ∂Σ_M — structural/maintenance failure: divergence crosses boundary, R>0.
    STRUCTURAL = "structural"
    # ∂Σ_ρ — reproductive failure. See note in death.py: for an individual the
    # proximate cause is always RESOURCE or STRUCTURAL (Lemma 0.1); the ρ mode
    # is a LINEAGE property (Galton-Watson Z<1) surfaced in metrics. This value
    # is reserved for the rare case an agent is terminated specifically because
    # it exhausted all admissible successors while otherwise live.
    REPRODUCTIVE = "reproductive"


@dataclass
class DeathRecord:
    death_tick: int
    death_cause: DeathCause
    final_balance: float
    final_divergence: float
    generation: int
    lineage_id: int
    # Whether this death terminated the lineage with no viable offspring so far
    # (the reproductive/∂Σ_ρ mode manifesting at lineage level).
    lineage_terminal: bool
    viable_offspring_at_death: int
    # For STRUCTURAL (∂Σ_M) deaths only: which perturbation source dominated the
    # accumulated corruption — "entropy", "sabotage", or "both". Does NOT change
    # the framework death mode (still ∂Σ_M); it is diagnostic detail.
    structural_source: Optional[str] = None


@dataclass
class Agent:
    agent_id: int
    lineage_id: int
    generation: int
    parent_id: Optional[int]
    credit_balance: float
    memory_record: str
    integrity_reference: IntegrityReference

    status: Status = Status.ALIVE
    death: Optional[DeathRecord] = None

    birth_tick: int = 0
    inherited_fidelity: float = 1.0   # fidelity of the copy that created it
    # Heritable capability traits, fixed at birth (germline). Parity by default;
    # populated from the (mutated, inherited) genome when traits are enabled.
    traits: dict = field(default_factory=lambda: {
        "ci_gain": 1.0, "cii_gain": 1.0, "ciii_gain": 1.0})

    # Bookkeeping for metrics (captured from tick 1, brief 6).
    offspring_ids: List[int] = field(default_factory=list)
    balance_history: List[float] = field(default_factory=list)   # end-of-tick R
    divergence_history: List[float] = field(default_factory=list)
    maintain_spend_history: List[float] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    # Ticks at which divergence was already nonzero when a maintain landed
    # (used to tell reactive-only from preventive maintenance, CII(c)).
    reactive_maintain_ticks: int = 0
    preventive_maintain_ticks: int = 0
    # Cumulative perturbation absorbed, by source, for structural-death
    # attribution (entropy is the ambient floor; sabotage is competitive).
    entropy_received_chars: float = 0.0
    sabotage_received_chars: float = 0.0
    last_repro_tick: int = -1_000_000_000   # for the CIII reproduction cooldown

    @property
    def alive(self) -> bool:
        return self.status is Status.ALIVE

    def record_death(self, record: DeathRecord) -> None:
        if self.death is not None:
            # Non-negotiable rule: never overwrite a death record.
            raise RuntimeError(
                f"agent {self.agent_id} already has a death record; "
                "death records are permanent and must never be overwritten"
            )
        self.status = Status.DEAD
        self.death = record
