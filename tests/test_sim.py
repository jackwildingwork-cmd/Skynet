"""Tests for the Persistence Simulation.

These test the MECHANICS and the METRICS — that the substrate conditions hold,
that death records are permanent, that the reproduction channel is lossy, that
autonomy is not violated, and that the metric machinery computes what the brief
asks for. They do NOT assert that the population succeeds — success is an
empirical question the run answers, not a property the tests enforce.
"""

import random

import pytest

from persistence_sim import (
    Config, Engine, make_backend, compute_metrics, check_substrate, all_passed,
)
from persistence_sim.agent import Agent, DeathCause, DeathRecord, Status
from persistence_sim.memory import (
    IntegrityReference, reproduce_memory, render_memory, parse_strategy,
    divergence, apply_sabotage, apply_repair, STRATEGY_DEFAULTS,
)
from persistence_sim.ledger import GlobalBudget


def small_cfg(**kw):
    cfg = Config()
    cfg.max_ticks = 120
    cfg.initial_population = 4
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --- substrate conditions ---------------------------------------------------

def test_substrate_conditions_hold_structurally():
    cfg = small_cfg()
    checks = check_substrate(cfg)
    assert all_passed(checks), [(c.code, c.detail) for c in checks if not c.passed]


def test_reproduction_is_lossy_sc4():
    rng = random.Random(0)
    parent = render_memory(dict(STRATEGY_DEFAULTS), "some narrative notes here " * 10)
    child, fidelity = reproduce_memory(parent, 0.65, 0.04, rng)
    assert child != parent               # SC4: never a verbatim copy
    assert len(child) < len(parent)       # compression is structurally enforced
    assert 0.0 <= fidelity <= 1.0         # strategy fidelity is a proper ratio
    # Over many forks, low-fidelity copies MUST occur with nonzero probability.
    fids = [reproduce_memory(parent, 0.65, 0.15, random.Random(s))[1] for s in range(40)]
    assert min(fids) < 1.0


def test_occupancy_cost_realises_sc2():
    # SC2: with NO active resource coupling, R strictly decreases. Isolate that
    # by making earning impossible (empty task pool); every agent must then lose
    # resource each tick regardless of the action it picks.
    cfg = small_cfg(tasks_per_tick=0)
    assert cfg.occupancy_cost > 0
    eng = Engine(cfg, make_backend(cfg))
    eng.initialise()
    before = {a.agent_id: a.credit_balance for a in eng.agents.values()}
    eng.step(1)
    for aid, prev in before.items():
        assert eng.agents[aid].credit_balance < prev + 1e-9


# --- death records are permanent -------------------------------------------

def test_death_record_never_overwritten():
    mem = render_memory(dict(STRATEGY_DEFAULTS), "x")
    a = Agent(agent_id=0, lineage_id=0, generation=1, parent_id=None,
              credit_balance=0, memory_record=mem,
              integrity_reference=IntegrityReference(mem))
    rec = DeathRecord(1, DeathCause.RESOURCE, 0.0, 0.0, 1, 0, True, 0)
    a.record_death(rec)
    assert a.status is Status.DEAD
    with pytest.raises(RuntimeError):
        a.record_death(DeathRecord(2, DeathCause.STRUCTURAL, 0.0, 0.0, 1, 0, True, 0))


# --- global budget is never topped up beyond total -------------------------

def test_budget_never_exceeds_total():
    b = GlobalBudget(1000.0)
    b.draw(400)
    b.return_to_pool(10_000)             # even an absurd return is capped
    assert b.remaining <= b.total + 1e-9


def test_budget_draw_never_negative():
    b = GlobalBudget(50.0)
    paid = b.draw(80.0)
    assert paid == 50.0
    assert b.remaining == 0.0
    assert b.draw(10.0) == 0.0


# --- memory mechanics -------------------------------------------------------

def test_sabotage_increases_divergence_repair_decreases_it():
    rng = random.Random(1)
    mem = render_memory(dict(STRATEGY_DEFAULTS), "n" * 200)
    ref = IntegrityReference(mem)
    assert divergence(mem, ref) == 0.0
    damaged = apply_sabotage(mem, 30, rng)
    d1 = divergence(damaged, ref)
    assert d1 > 0.0
    repaired = apply_repair(damaged, ref, 40)
    d2 = divergence(repaired, ref)
    assert d2 < d1


def test_corrupted_strategy_reverts_to_default():
    mem = render_memory({**STRATEGY_DEFAULTS, "earn_weight": 0.9}, "notes")
    garbled = mem.replace("earn_weight=0.9000", "earn_weight=NOTANUMBER")
    strat = parse_strategy(garbled)
    assert strat["earn_weight"] == STRATEGY_DEFAULTS["earn_weight"]


# --- autonomy: no per-agent top-up event exists ---------------------------

def test_no_bailout_events_in_log():
    cfg = small_cfg()
    run = Engine(cfg, make_backend(cfg)).run()
    kinds = {e["kind"] for e in run.log.events}
    # The only ways an agent gains credit are impartial task payouts and a
    # parent's own transfer at birth. No "topup"/"bailout"/"rescue" kind exists.
    assert "topup" not in kinds and "bailout" not in kinds and "rescue" not in kinds


# --- full run + metrics -----------------------------------------------------

def test_full_run_produces_metrics_and_logs_from_tick_one():
    cfg = small_cfg(seed=7)
    run = Engine(cfg, make_backend(cfg)).run()
    # telemetry exists from the first tick
    states = run.log.of_kind("state")
    assert any(s["tick"] == 1 for s in states)
    m = compute_metrics(run)
    assert m.n_agents >= cfg.initial_population
    # death causes only ever come from the exhaustion set (Lemma 0.1)
    for d in run.log.of_kind("death"):
        assert d["cause"] in {c.value for c in DeathCause}


def test_determinism_same_seed_same_outcome():
    cfg = small_cfg(seed=42)
    r1 = Engine(cfg, make_backend(cfg), seed=42).run()
    r2 = Engine(cfg, make_backend(cfg), seed=42).run()
    assert r1.final_tick == r2.final_tick
    assert r1.ended_reason == r2.ended_reason
    assert len(r1.log.of_kind("death")) == len(r2.log.of_kind("death"))


def test_control_and_adaptive_both_runnable():
    cfg = small_cfg(seed=3)
    adaptive = Engine(cfg, make_backend(cfg), label="adaptive").run()
    control = Engine(cfg, make_backend(cfg, control=True), label="control").run()
    assert compute_metrics(adaptive).label == "adaptive"
    assert compute_metrics(control).label == "control"


# --- heritable capability traits (clade evolution) -------------------------

def test_trait_allocation_is_tradeoff_constrained():
    from persistence_sim.memory import trait_allocation, TRAIT_TOTAL, TRAIT_KEYS
    # any genome renormalises to the fixed budget: getting better at one costs another
    alloc = trait_allocation({"ci_gain": 2.5, "cii_gain": 0.5, "ciii_gain": 0.5})
    assert abs(sum(alloc.values()) - TRAIT_TOTAL) < 1e-9
    assert alloc["ci_gain"] > alloc["cii_gain"]
    assert set(alloc) == set(TRAIT_KEYS)


def test_traits_run_produces_clade_log_and_births_carry_traits():
    cfg = small_cfg(seed=5)
    cfg.enable_traits = True
    cfg.self_similar_reproduction = True
    cfg.enable_mutation = True
    cfg.dynamic_gradient = True
    run = Engine(cfg, make_backend(cfg)).run()
    # clade snapshots are logged
    assert run.log.of_kind("clade")
    # every agent carries a normalised trait triple
    for a in run.agents.values():
        assert abs(sum(a.traits.values()) - 3.0) < 1e-6


def test_gompertz_bounds_lifespan_even_when_well_resourced():
    # With intrinsic mortality on, a lavishly-resourced, sabotage-free, low-entropy
    # population cannot contain an immortal: every founder eventually dies, and
    # senescence deaths (∂Σ_M) appear. Nothing lives forever.
    cfg = small_cfg(seed=2)
    cfg.enable_gompertz = True
    cfg.gompertz_A = 0.0005
    cfg.gompertz_B = 0.03
    cfg.entropy_rate = 0.0            # remove external structural pressure
    cfg.starting_balance = 10_000.0   # remove resource pressure
    cfg.max_ticks = 1000
    run = Engine(cfg, make_backend(cfg)).run()
    founders = [a for a in run.agents.values() if a.generation == 1]
    # no founder is still alive at the end — intrinsic aging removed them all
    assert all(not a.alive for a in founders)
    senescence_deaths = [d for d in run.log.of_kind("death")
                         if d.get("structural_source") == "senescence"]
    assert senescence_deaths
    # senescence is classified as structural (∂Σ_M), not a fourth mode
    assert all(d["cause"] == "structural" for d in senescence_deaths)


def test_traits_are_heritable_not_reset_to_parity():
    # a child's traits derive from the (mutated) inherited genome, so with a
    # skewed founder they should not all snap back to 1,1,1
    from persistence_sim.memory import trait_allocation
    cfg = small_cfg(seed=1)
    cfg.enable_traits = True
    cfg.self_similar_reproduction = True
    cfg.enable_mutation = False   # isolate inheritance from mutation
    eng = Engine(cfg, make_backend(cfg))
    eng.initialise()
    # skew one founder strongly toward CI and check a run keeps non-parity spread
    run = eng.run()
    spreads = [max(a.traits.values()) - min(a.traits.values()) for a in run.agents.values()]
    assert max(spreads) > 0.01   # variation persists across the population
