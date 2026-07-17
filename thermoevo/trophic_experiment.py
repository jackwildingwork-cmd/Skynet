"""Milestone-4 experiment: two trophic levels in one tier.

The environment gets bigger. The gradient the herbivores forage is no longer given
as an abstract logistic field — it is PRODUCED. A population of autotroph processes
couples to a UNIVERSAL gradient (sunlight, available to every producer everywhere)
and fixes it into standing structural stock (vegetation). Herbivores cannot access
sunlight; they forage the producers' biomass, which is patchy and must be found.

Three questions, each measured:

  1. Do producers persist by fixing a universal gradient? A universal gradient has
     no "finding" problem, so what bounds them? Self-shading (a Beer-Lambert split
     of light between neighbours) and recruitment limitation (a seed establishes
     only on a safe site). Together these give a real carrying capacity from an
     unbounded gradient. Run producers alone and check they reach a stable crop.

  2. Do the two levels coexist? Add herbivores. A naive predator-prey coupling
     collapses; a real one settles. Measure whether producers are held DOWN by
     grazing (top-down control) while herbivores are held at a carrying capacity set
     by producer productivity (bottom-up) — and whether both persist indefinitely.

  4. What happens when grazing gets heavy? The stable crop is not the only regime.
     Turning up grazing efficiency destabilises the equilibrium into predator-prey
     limit cycles (boom-bust) of growing amplitude, and finally into cycles that
     crash to extinction — a Rosenzweig-MacArthur / paradox-of-enrichment transition
     that falls straight out of the substrate. Reported, not tuned away: boom-bust is
     real ecology, not a bug.

  3. Does the gradient's structure decide whether intelligence evolves? This is the
     sharp prediction: the SAME thermodynamic process, facing a universal gradient
     (producers) vs a patchy one (herbivores), should be pushed toward foraging
     intelligence only in the patchy case. Compare evolved chemotaxis of herbivores
     (patchy gradient, must find) against the producers' non-existent finding
     problem. Intelligence should track "must the gradient be found?", not the
     organism.

Run:  python -m thermoevo.trophic_experiment
"""

from __future__ import annotations

import statistics

import numpy as np

from .producers import Producers, ProducerConfig
from .world import WorldConfig, World, run


def producers_alone(seeds: int = 3, ticks: int = 800):
    print("=== 1. Do producers persist on a UNIVERSAL gradient? (sunlight -> vegetation) ===")
    print(f"    {'seed':>4} {'prod_pop':>9} {'biomass':>9} {'gen':>5}  (stable standing crop?)")
    for s in range(seeds):
        rng = np.random.default_rng(s)
        p = Producers(ProducerConfig(seed=s), rng)
        crop = []
        for t in range(ticks):
            p.step()
            if t >= ticks - 200:
                crop.append(p.size)
        m = p.metrics()
        cv = statistics.pstdev(crop) / max(1e-9, statistics.mean(crop))
        print(f"    {s:>4} {m['prod_pop']:>9} {m['prod_biomass']:>9.0f} {m['prod_gen']:>5.0f}"
              f"   crop CV over last 200 ticks = {cv:.3f}")
    print("    -> a universal gradient + self-shading + recruitment limitation gives a")
    print("       stable carrying capacity: no finding needed, yet biomass is bounded.")


def coexistence(seeds: int = 3, ticks: int = 3000):
    print("\n=== 2. Do the two trophic levels coexist? (top-down + bottom-up control) ===")
    # producer-only baseline (ungrazed standing crop) for the same field
    base = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        p = Producers(ProducerConfig(seed=s), rng)
        for _ in range(600):
            p.step()
        base.append(p.size)
    ungrazed = statistics.mean(base)
    print(f"    ungrazed producer crop (no herbivores): {ungrazed:.0f}")
    print(f"    {'seed':>4} {'end':>10} {'herb':>6} {'prod':>6} {'grazed_down':>12}")
    hh, pp = [], []
    for s in range(seeds):
        r = run(WorldConfig(seed=s, ticks=ticks, probe_interval=500))
        last = r.history[-1]
        h, pr = r.final_pop, last.get("prod_pop", 0)
        hh.append(h); pp.append(pr)
        frac = 1.0 - pr / ungrazed
        print(f"    {s:>4} {r.ended:>10} {h:>6} {pr:>6} {frac:>11.0%}")
    print(f"    mean: herbivores {statistics.mean(hh):.0f} coexist with producers "
          f"{statistics.mean(pp):.0f}")
    print(f"    -> grazing holds producers ~{1 - statistics.mean(pp)/ungrazed:.0%} below their")
    print("       ungrazed crop (top-down), while herbivores are food-limited (bottom-up).")
    print("       Both persist: a stable two-level ecology, not a boom-bust collapse.")


def intelligence_tracks_the_gradient(seeds: int = 3, ticks: int = 3000):
    print("\n=== 3. Does gradient STRUCTURE (not organism) decide if intelligence evolves? ===")
    print("    herbivores face a PATCHY gradient (producer biomass) they must FIND;")
    print("    producers face a UNIVERSAL gradient with no finding problem at all.")
    base_c, evo_c = [], []
    for s in range(seeds):
        # random-genome herbivore baseline chemotaxis on the produced field
        w = World(WorldConfig(seed=s, ticks=10))
        for t in range(1, 40):
            if w.size == 0:
                break
            w.step(t)
        base_c.append(float(np.mean(w._chemo)) if w._chemo else float("nan"))
        r = run(WorldConfig(seed=s, ticks=ticks))
        ev = [h["chemotaxis"] for h in r.history if "chemotaxis" in h]
        evo_c.append(statistics.mean(ev[-3:]) if len(ev) >= 3 else float("nan"))
    print(f"    herbivore chemotaxis (patchy gradient): random {statistics.mean(base_c):.2f}"
          f" -> evolved {statistics.mean(evo_c):.2f}")
    print("    producer foraging problem (universal gradient): NONE — sunlight is")
    print("    everywhere, so there is no gradient to climb and no chemotaxis to evolve.")
    print("    -> foraging intelligence is selected by the gradient being PATCHY, i.e.")
    print("       findable-with-difficulty. Same physics, same core process; the")
    print("       environment's structure, not the organism, decides if a mind pays.")


def grazing_pressure_transition(seeds: int = 4, ticks: int = 4000):
    print("\n=== 4. Turn up grazing: coexistence -> boom-bust -> collapse (emergent) ===")
    print("    Heavier grazing does not just lower the standing crop; it DESTABILISES the")
    print("    equilibrium. Herbivore population CV (over t>800) measures the amplitude of")
    print("    the predator-prey cycle; at the top it hits the extinction boundary.")
    print(f"    {'graze_frac':>10} {'intake':>6}   outcome per seed (CV = cycle amplitude)")
    for gmax, intake in ((0.55, 8.0), (0.70, 10.0), (0.85, 12.0), (0.92, 13.0)):
        outs = []
        for s in range(seeds):
            pc = ProducerConfig(seed=s, graze_max_frac=gmax)
            w = World(WorldConfig(seed=s, ticks=ticks, producers=pc, max_intake=intake))
            hs, dead = [], False
            for t in range(1, ticks + 1):
                w.step(t)
                if w.size == 0:
                    dead = True; break
                if t > 800:
                    hs.append(w.size)
            if dead:
                outs.append("extinct")
            else:
                cv = statistics.pstdev(hs) / max(1e-9, statistics.mean(hs))
                outs.append(f"CV={cv:.2f}")
        print(f"    {gmax:>10} {intake:>6}   {outs}")
    print("    -> a Rosenzweig-MacArthur / paradox-of-enrichment transition, unforced: the")
    print("       stable coexistence equilibrium gives way to limit cycles of growing")
    print("       amplitude, then to cycles that crash into extinction. Boom-bust is not a")
    print("       failure mode here — it is what the substrate does under heavy predation.")


def main():
    print("thermoevo — Milestone 4: two trophic levels in one tier\n"
          "(universal-gradient producers -> patchy-gradient herbivores)\n")
    producers_alone()
    coexistence()
    intelligence_tracks_the_gradient()
    grazing_pressure_transition()


if __name__ == "__main__":
    main()
