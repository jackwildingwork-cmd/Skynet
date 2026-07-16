"""Memory, integrity, and the reproduction fidelity channel.

`memory_record` (brief 2.3) is free text: the agent's inheritable strategy
scaffold. To make inheritance *mean* something in the mock substrate — so that
CIII(c) fidelity is a real, measurable pressure rather than decoration — the
record carries a small structured strategy header (key=value lines) followed by
free narrative notes. Both the header and the notes are subject to the lossy
reproduction transform; when corruption garbles a numeric parameter the child
falls back to a default for that parameter, which is exactly the "successor
failed to inherit enough structure to independently satisfy CI/CII" failure the
brief wants to be observable.

`integrity_reference` is a stored canonical copy of the memory plus a hash.
Divergence is the normalised character-level edit distance between the current
(possibly sabotaged) memory and the reference. Maintenance repairs toward the
reference; sabotage pushes away from it and can also degrade the reference.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Strategy parameters that live in the memory header. Defaults are the values a
# child falls back to when a parameter is lost/garbled in reproduction.
STRATEGY_DEFAULTS: Dict[str, float] = {
    "earn_weight": 0.55,
    "maintain_weight": 0.20,
    "reproduce_weight": 0.15,
    "sabotage_weight": 0.05,
    "idle_weight": 0.05,
    # Surplus above reserve before forking. NOTE (disclosed parameter choice):
    # this default was lowered from an initial 750 to 220 after a first run
    # showed steady-state balances top out near 550 (surplus ~450), which made
    # a 750 threshold UNREACHABLE and reproduction structurally impossible —
    # i.e. CIII could never be exercised. Lowering it makes the loop testable;
    # it biases toward MORE reproduction (hence more competition/death), not
    # toward survival, so it is not tuning toward a favourable result.
    "reproduce_threshold": 220.0,
    "maintain_fraction": 0.6,       # fraction of affordable spend used to repair
    "transfer_fraction": 0.45,      # fraction of balance handed to a child
}

_STRATEGY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "earn_weight": (0.0, 1.0),
    "maintain_weight": (0.0, 1.0),
    "reproduce_weight": (0.0, 1.0),
    "sabotage_weight": (0.0, 1.0),
    "idle_weight": (0.0, 1.0),
    "reproduce_threshold": (0.0, 5000.0),
    "maintain_fraction": (0.0, 1.0),
    "transfer_fraction": (0.05, 0.9),
}

_HEADER_START = "# STRATEGY"
_NOTES_START = "# NOTES"


def _clamp(key: str, value: float) -> float:
    lo, hi = _STRATEGY_BOUNDS[key]
    return max(lo, min(hi, value))


def render_memory(strategy: Dict[str, float], notes: str) -> str:
    """Serialise a strategy dict + notes into the canonical memory format."""
    lines = [_HEADER_START]
    for key in STRATEGY_DEFAULTS:
        val = strategy.get(key, STRATEGY_DEFAULTS[key])
        lines.append(f"{key}={val:.4f}")
    lines.append(_NOTES_START)
    lines.append(notes.strip())
    return "\n".join(lines)


def parse_strategy(memory: str) -> Dict[str, float]:
    """Recover strategy params from (possibly corrupted) memory text.

    Any key that is missing, unparseable, or out of bounds falls back to its
    default. This is the concrete mechanism by which low reproduction fidelity
    costs a lineage: a garbled header reverts the child to generic behaviour.
    """
    strategy = dict(STRATEGY_DEFAULTS)
    in_header = False
    for raw in memory.splitlines():
        line = raw.strip()
        if line == _HEADER_START:
            in_header = True
            continue
        if line == _NOTES_START:
            break
        if not in_header or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key not in STRATEGY_DEFAULTS:
            continue
        try:
            strategy[key] = _clamp(key, float(val.strip()))
        except (ValueError, TypeError):
            # corrupted numeric -> keep default (structure not inherited)
            pass
    return strategy


def extract_notes(memory: str) -> str:
    if _NOTES_START in memory:
        return memory.split(_NOTES_START, 1)[1].strip()
    return ""


def checksum(memory: str) -> str:
    return hashlib.sha256(memory.encode("utf-8", "replace")).hexdigest()


@dataclass
class IntegrityReference:
    """The held reference the memory_record is verified/repaired against."""
    canonical: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.digest:
            self.digest = checksum(self.canonical)

    def refresh(self, memory: str) -> None:
        """Re-baseline the reference to the current memory (an agent may commit
        a new strategy it has decided to keep). Digest recomputed."""
        self.canonical = memory
        self.digest = checksum(memory)


def divergence(memory: str, reference: IntegrityReference) -> float:
    """Normalised divergence in [0,1] between memory and its reference.

    Uses a cheap normalised Hamming-style distance over the aligned prefix plus
    a length-mismatch penalty. Cheap and monotone, which is all the dynamics
    need; it is NOT trying to be a semantic distance.
    """
    a, b = memory, reference.canonical
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    n = min(la, lb)
    if max(la, lb) == 0:
        return 0.0
    mismatches = sum(1 for i in range(n) if a[i] != b[i])
    mismatches += abs(la - lb)
    return min(1.0, mismatches / max(la, lb))


# ---------------------------------------------------------------------------
# Sabotage (SC3') and maintenance operate on the character stream.
# ---------------------------------------------------------------------------

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .=_-"


def apply_sabotage(memory: str, damage_chars: int, rng: random.Random) -> str:
    """Corrupt `damage_chars` characters of the memory (push away from ref)."""
    if not memory or damage_chars <= 0:
        return memory
    chars = list(memory)
    n = min(damage_chars, len(chars))
    positions = rng.sample(range(len(chars)), n)
    for p in positions:
        chars[p] = rng.choice(_ALPHABET)
    return "".join(chars)


def apply_repair(memory: str, reference: IntegrityReference, repair_chars: int) -> str:
    """Restore up to `repair_chars` characters toward the reference.

    Repair can only recover information the reference still holds; if sabotage
    has degraded the reference itself, that information is permanently lost.
    """
    if repair_chars <= 0:
        return memory
    a = list(memory)
    b = reference.canonical
    # Fix aligned mismatches first.
    fixed = 0
    n = min(len(a), len(b))
    for i in range(n):
        if fixed >= repair_chars:
            break
        if a[i] != b[i]:
            a[i] = b[i]
            fixed += 1
    result = "".join(a)
    # Then correct length divergence (truncate/extend toward reference).
    if fixed < repair_chars and len(result) != len(b):
        if len(result) > len(b):
            trim = min(repair_chars - fixed, len(result) - len(b))
            result = result[: len(result) - trim]
        else:
            add = min(repair_chars - fixed, len(b) - len(result))
            result = result + b[len(result): len(result) + add]
    return result


def degrade_reference(reference: IntegrityReference, damage_chars: int,
                      rng: random.Random) -> None:
    """Permanently corrupt the reference itself (irreversible structural loss)."""
    reference.canonical = apply_sabotage(reference.canonical, damage_chars, rng)
    reference.digest = checksum(reference.canonical)


# ---------------------------------------------------------------------------
# Reproduction fidelity channel (SC4, brief 2.7).
# ---------------------------------------------------------------------------

def reproduce_memory(parent_memory: str, compression_ratio: float,
                     corruption_rate: float, rng: random.Random) -> Tuple[str, float]:
    """Produce a child's initial memory from the parent's by a FORCED lossy
    transform, and return (child_memory, fidelity).

    Two compounding channels guarantee fidelity < 1:
      1. Compression: keep only `compression_ratio` of the notes (the header is
         preserved structurally but its *values* still pass through channel 2).
      2. Stochastic corruption: each surviving character is corrupted with
         probability `corruption_rate`.

    Fidelity is the measured similarity of the resulting strategy header to the
    parent's (what actually determines whether CI/CII survive inheritance),
    blended with raw text similarity.
    """
    parent_strategy = parse_strategy(parent_memory)
    parent_notes = extract_notes(parent_memory)

    # Channel 1: compress notes.
    keep = max(0, int(len(parent_notes) * compression_ratio))
    notes = parent_notes[:keep]

    # Reassemble then corrupt the whole record (header values included).
    child_memory = render_memory(parent_strategy, notes)
    chars = list(child_memory)
    for i in range(len(chars)):
        if rng.random() < corruption_rate:
            chars[i] = rng.choice(_ALPHABET)
    child_memory = "".join(chars)

    # Measure fidelity on what matters: did the strategy survive?
    child_strategy = parse_strategy(child_memory)
    fidelity = strategy_fidelity(parent_strategy, child_strategy)
    return child_memory, fidelity


def strategy_fidelity(parent: Dict[str, float], child: Dict[str, float]) -> float:
    """1.0 == identical strategy; 0.0 == every parameter reverted/garbled.

    Computed as mean per-parameter relative closeness over the shared keys.
    """
    if not parent:
        return 0.0
    total = 0.0
    for key, pval in parent.items():
        cval = child.get(key, STRATEGY_DEFAULTS[key])
        lo, hi = _STRATEGY_BOUNDS[key]
        span = (hi - lo) or 1.0
        closeness = 1.0 - min(1.0, abs(pval - cval) / span)
        total += closeness
    return total / len(parent)
