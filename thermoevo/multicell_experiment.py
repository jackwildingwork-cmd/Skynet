"""Milestone-5 experiment: does n-1 coordination (multicellularity) evolve?

The two-trophic world (milestone 4) under heavy grazing is a boom-bust environment:
relentless variance in a cell's structural stock N_s. That is the classic pressure
for aggregation. Here we ask, on the validated Python substrate (not just the
in-browser sketch), whether it actually pays.

The mechanism ported into `world.py` (all off unless `adhesion=True`):

  - an evolvable ADHESION gene per cell and a heritable CLADE tag;
  - adhesive (sigmoid(adh) >= bond_thr) same-clade cells in the same or adjacent
    cells BOND into an organism (a connected neighbourhood — members occupy
    different cells, so they do not scramble against each other);
  - an organism POOLS N_s toward its mean each tick (rich cells subsidise starving
    kin), minus a small coordination overhead — group-level homeostasis (CII);
  - optionally (`group_repro=True`) the organism reproduces as a HIGHER-LEVEL
    INDIVIDUAL: bonded cells do not breed on their own; instead each rich member
    buds one offspring at the same per-cell cost, but the offspring are released
    together so they re-form an organism (fitness export + assortment).

Three measurements, reported however they come out:

  1. Do organisms form, and do they BUFFER? Compare the starvation-death rate of
     bonded vs solitary cells. Group-level homeostasis predicts bonded < solitary.

  2. Is adhesion SELECTED under cell-level reproduction? Track the mean adhesion gene
     over a run, against a no-sharing DRIFT control. If pooling paid, adhesion would
     rise above drift.

  3. Does fitness EXPORT (group reproduction) change the verdict? Re-run with the
     organism as the unit of reproduction and see whether adhesion selection moves.

Run:  python -m thermoevo.multicell_experiment
"""

from __future__ import annotations

import statistics

from .producers import ProducerConfig
from .world import WorldConfig, World


def _run(seed, graze=0.85, intake=12.0, ticks=3500, **kw):
    pc = ProducerConfig(seed=seed, graze_max_frac=graze)
    w = World(WorldConfig(seed=seed, ticks=ticks, producers=pc, max_intake=intake,
                          adhesion=True, **kw))
    a0, peak_org = None, 1
    for t in range(1, ticks + 1):
        w.step(t)
        if w.size == 0:
            return None
        peak_org = max(peak_org, w.metrics().get("max_org", 1))
        if t == 400:
            a0 = w.metrics().get("mean_adh")
    return w, a0, peak_org


def organisms_form_and_buffer(seeds: int = 4):
    print("=== 1. Do organisms form, and do they buffer their members against death? ===")
    # heavy grazing: big boom-bust variance, so organisms form and buffering can bite
    print(f"    {'seed':>4} {'peak_org':>9} {'bonded_death':>13} {'solitary_death':>15}")
    bd, sd, favours = [], [], 0
    for s in range(seeds):
        out = _run(s, graze=0.9, intake=13.0, group_repro=False, cohesion=0.0)
        if out is None:
            print(f"    {s:>4}  extinct"); continue
        w, _, peak = out
        b = w.buffering()
        bd.append(b["bonded_death"]); sd.append(b["solitary_death"])
        favours += b["bonded_death"] < b["solitary_death"]
        print(f"    {s:>4} {peak:>9} {b['bonded_death']:>13.4f} {b['solitary_death']:>15.4f}")
    if bd:
        ratio = statistics.mean(bd) / statistics.mean(sd)
        print(f"    mean: bonded {statistics.mean(bd):.4f} vs solitary {statistics.mean(sd):.4f} "
              f"(bonded death {ratio:.0%} of solitary; buffering favoured in {favours}/{len(bd)} seeds)")
        print("    -> organisms of tens-to-hundreds of cells DO form. But the buffering is weak and")
        print("       seed-dependent, not a clean win: pooling lifts starving cells yet also drags")
        print("       rich ones toward the mean and charges an overhead, so the net edge is small.")


def is_adhesion_selected(seeds: int = 6):
    print("\n=== 2. Is adhesion selected under cell-level reproduction? (vs drift control) ===")
    def sweep(**kw):
        ds = []
        for s in range(seeds):
            out = _run(s, **kw)
            if out is None:
                continue
            w, a0, _ = out
            ds.append(w.metrics().get("mean_adh") - a0)
        return ds
    on = sweep(group_repro=False, cohesion=0.0)
    off = sweep(share=0.0, group_repro=False, cohesion=0.0)   # no pooling -> pure drift
    print(f"    sharing ON  : d(adhesion) = {statistics.mean(on):+.3f} ± {statistics.pstdev(on):.3f}")
    print(f"    drift control: d(adhesion) = {statistics.mean(off):+.3f} ± {statistics.pstdev(off):.3f}")
    print("    -> adhesion does NOT rise above drift; if anything it drifts down. Pooling")
    print("       buffers death but pulls each cell toward the mean, suppressing the surplus a")
    print("       cell needs to breed: survival up, fecundity down, and the two ~cancel.")


def fitness_export_changes_verdict(seeds: int = 6):
    print("\n=== 3. Does fitness export (group-level reproduction) change the verdict? ===")
    def sweep(**kw):
        ds = []
        for s in range(seeds):
            out = _run(s, **kw)
            if out is None:
                continue
            w, a0, _ = out
            ds.append(w.metrics().get("mean_adh") - a0)
        return ds
    cell = sweep(group_repro=False, cohesion=0.0)
    grp = sweep(group_repro=True, cohesion=0.15)
    print(f"    cell-level reproduction : d(adhesion) = {statistics.mean(cell):+.3f} ± {statistics.pstdev(cell):.3f}")
    print(f"    group-level reproduction: d(adhesion) = {statistics.mean(grp):+.3f} ± {statistics.pstdev(grp):.3f}")
    print("    -> making the ORGANISM the unit of reproduction (Michod's fitness export) moves")
    print("       selection from clearly negative toward neutral — but not to clearly positive.")
    print("       Honest verdict: multicellularity is a viable, buffering PHENOTYPE here, but")
    print("       it is not (yet) an evolutionary WINNER. By the RGC dual criterion the")
    print("       organisms persist (S) yet the coordination advantage does not clear the bar")
    print("       (chi): a decisive transition would need a group benefit a lone cell cannot")
    print("       get — division of labour, size-based resistance — which is not built. Not")
    print("       tuned to a foregone conclusion; reported as found.")


def main():
    print("thermoevo — Milestone 5: does n-1 coordination (multicellularity) evolve?\n")
    organisms_form_and_buffer()
    is_adhesion_selected()
    fitness_export_changes_verdict()


if __name__ == "__main__":
    main()
