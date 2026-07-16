"""Rigorous Gompertz aging sweep: fixed gradient_max, n>=32, two base configs.

Config A (favorable):      scale-free gradient, alpha=1.0
Config B (less favorable): near-crowding gradient, alpha=0.6
Both: gradient_max=20 (FIXED across all rows), gradient_half=8, self_similar
reproduction on, relaxed costs (occ6/rmin50/cooldown8), traits off, mutation on,
ticks=900, n=32.

Per cell reports:
  - extinction count k/n, and for k=0 the rule-of-three 95% upper bound on the
    true extinction probability (~3/n).
  - P(founder alive at final tick): analytic S_aging(T) from the fitted hazard
    (aging-only ceiling) AND the empirical gen-1 survival fraction (all causes).
  - survivor generation (mean gen of agents alive at final tick).
  - final population.
"""
import sys, math, statistics
sys.path.insert(0, "/home/user/Skynet")
from persistence_sim import Config, Engine, make_backend, compute_metrics

OUT = open("/tmp/claude-0/-home-user-Skynet/d47007c0-653d-5fa3-88a6-997aff0324b4/scratchpad/gompertz2_report.txt", "w")
def p(*a):
    print(*a); print(*a, file=OUT); OUT.flush()
def mean(x): return statistics.mean(x) if x else float("nan")

A = 0.0001
TICKS = 900
N = 32
GMAX = 20

def S_aging(B, T=TICKS):
    if B is None or B == 0: return 1.0
    return math.exp(-(A / B) * (math.exp(B * T) - 1))

def cfg_for(seed, B, alpha):
    c = Config(); c.seed = seed; c.max_ticks = TICKS
    c.enable_mutation = True; c.enable_traits = False
    c.self_similar_reproduction = True
    c.dynamic_gradient = True; c.gradient_scaling_exp = alpha
    c.gradient_max = GMAX; c.gradient_half = 8
    c.occupancy_cost = 6.0; c.r_min = 50.0; c.reproduce_cooldown = 8.0
    if B is None: c.enable_gompertz = False
    else: c.enable_gompertz = True; c.gompertz_A = A; c.gompertz_B = B
    return c

def sweep(config_name, alpha):
    p(""); p("### %s (alpha=%.1f), gradient_max=%d FIXED, n=%d, ticks=%d" % (
        config_name, alpha, GMAX, N, TICKS))
    p("%-12s %8s %10s %12s %11s %9s %8s" % (
        "aging B", "ext k/n", "ext<=(95%)", "P(found alive)", "found_emp", "survGen", "pop"))
    for B in (None, 0.02, 0.03, 0.05, 0.08, 0.12):
        ext = 0; founders_alive = 0; founders_total = 0; sg = []; fp = []
        for s in range(N):
            run = Engine(cfg_for(s, B, alpha), make_backend(cfg_for(s, B, alpha)), seed=s).run()
            alive = [a for a in run.agents.values() if a.alive]
            ext += run.ended_reason == "extinction"
            g1 = [a for a in run.agents.values() if a.generation == 1]
            founders_total += len(g1); founders_alive += sum(1 for a in g1 if a.alive)
            sg.append(mean([a.generation for a in alive]) if alive else 0.0)
            fp.append(len(alive))
        r3 = "<=%.3f" % (3.0 / N) if ext == 0 else "  n/a"
        femp = founders_alive / founders_total if founders_total else float("nan")
        label = "off" if B is None else ("%.2f" % B)
        p("%-12s %6d/%d %10s %12.2e %11.3f %9.1f %8.1f" % (
            label, ext, N, r3, S_aging(B), femp, mean(sg), mean(fp)))

p("GOMPERTZ AGING SWEEP - fixed gradient_max=%d, n=%d, ticks=%d, A=%.4f" % (GMAX, N, A, A))
p("P(found alive) = analytic aging-only survival to tick %d (ceiling); found_emp" % TICKS)
p("= empirical fraction of generation-1 agents alive at final tick (all causes).")
sweep("Config A: FAVORABLE scale-free", 1.0)
sweep("Config B: LESS-FAVORABLE near-crowding", 0.6)
p(""); p("DONE")
OUT.close()
