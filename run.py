#!/usr/bin/env python3
"""CLI entry point for the Persistence Simulation.

Runs an adaptive population and a matched fixed-policy control under identical
conditions, verifies the substrate preconditions, computes the framework
metrics, and prints an honest report. Optionally writes the full event log
(including permanent death records) to JSONL.

Usage:
    python run.py                       # default mock run + control
    python run.py --ticks 600 --pop 5
    python run.py --backend claude      # live cognitive substrate (a spending
                                        # decision; requires anthropic + API key)
    python run.py --log-out run.jsonl
"""

from __future__ import annotations

import argparse
import sys

from persistence_sim import (
    Config, Engine, make_backend, compute_metrics, compare,
    check_substrate, run_ensemble, format_ensemble,
)
from persistence_sim.report import format_substrate, format_metrics, format_comparison


def build_config(args) -> Config:
    cfg = Config()
    if args.ticks is not None:
        cfg.max_ticks = args.ticks
    if args.pop is not None:
        cfg.initial_population = args.pop
    if args.budget is not None:
        cfg.global_budget = args.budget
    if args.seed is not None:
        cfg.seed = args.seed
    if args.backend is not None:
        cfg.cognition_backend = args.backend
    if args.control_policy is not None:
        cfg.control_policy = args.control_policy
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Persistence Simulation v1")
    p.add_argument("--ticks", type=int, default=None, help="external tick cutoff")
    p.add_argument("--pop", type=int, default=None, help="initial population")
    p.add_argument("--budget", type=float, default=None, help="global budget ceiling")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", choices=["mock", "claude"], default=None)
    p.add_argument("--control-policy", choices=["always_earn", "uniform_random"],
                   default=None)
    p.add_argument("--log-out", type=str, default=None,
                   help="write full event log (with death records) to JSONL")
    p.add_argument("--no-control", action="store_true",
                   help="skip the control population")
    p.add_argument("--ensemble", type=int, default=None, metavar="N",
                   help="run N seeds and report the DISTRIBUTION (recommended: "
                        "single seeds are misleading)")
    args = p.parse_args(argv)

    cfg = build_config(args)

    # Ensemble mode: the honest interface. Runs the substrate check + a detailed
    # single-seed report for seed 0 as an illustrative example, then the
    # across-seed distribution that the conclusions actually rest on.
    if args.ensemble is not None:
        if cfg.cognition_backend == "claude":
            print("Refusing to run an ensemble on the live Claude backend: that "
                  "is a large, real spending decision. Use --backend mock for "
                  "ensembles, or a single --backend claude run explicitly.")
            return 2
        print(f"Persistence Simulation v1 — ENSEMBLE of {args.ensemble} seeds, "
              f"backend={cfg.cognition_backend}, pop={cfg.initial_population}, "
              f"budget={cfg.global_budget}, cutoff={cfg.max_ticks}\n")
        example = Engine(cfg, make_backend(cfg), label="adaptive", seed=0).run()
        print(format_substrate(check_substrate(cfg, example)))
        print()
        rep = run_ensemble(cfg, args.ensemble)
        print(format_ensemble(rep))
        return 0

    print(f"Persistence Simulation v1 — backend={cfg.cognition_backend}, "
          f"pop={cfg.initial_population}, budget={cfg.global_budget}, "
          f"cutoff={cfg.max_ticks}, seed={cfg.seed}\n")

    # Adaptive population.
    adaptive_engine = Engine(cfg, make_backend(cfg), label="adaptive", seed=cfg.seed)
    adaptive_run = adaptive_engine.run()

    checks = check_substrate(cfg, adaptive_run)
    print(format_substrate(checks))
    print()

    adaptive_metrics = compute_metrics(adaptive_run)
    print(format_metrics(adaptive_metrics))
    print()

    if not args.no_control:
        # Control: same config & seed, fixed non-adaptive policy.
        control_engine = Engine(cfg, make_backend(cfg, control=True),
                                label="control", seed=cfg.seed)
        control_run = control_engine.run()
        control_metrics = compute_metrics(control_run)
        print(format_metrics(control_metrics))
        print()
        cmp = compare(adaptive_metrics, control_metrics)
        print(format_comparison(cmp))
        print()

    if args.log_out:
        with open(args.log_out, "w") as fh:
            adaptive_run.log.write_jsonl(fh)
        print(f"Full adaptive event log written to {args.log_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
