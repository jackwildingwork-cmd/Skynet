#!/usr/bin/env python3
"""Run and print the Milestone-1 validation battery.

Verifies the thermodynamic substrate reproduces the Persistence Theorem's three
condition-theorems and the prediction-reward mechanism, before any evolutionary
layer is built on top.

Usage:  python run_milestone.py
"""

from __future__ import annotations

from thermoevo.validate import run_all


def main() -> int:
    checks = run_all(seed=0)
    print("=" * 76)
    print("thermoevo — Milestone 1: thermodynamic core validation")
    print("=" * 76)
    all_ok = True
    for key, c in checks.items():
        mark = "PASS" if c.passed else "FAIL"
        all_ok &= c.passed
        print(f"\n[{mark}] {key}: {c.name}")
        print(f"       {c.detail}")
        for row in c.rows:
            print("       " + "  ".join(f"{x!s:>10}" for x in row))
    print("\n" + "=" * 76)
    print("ALL THEOREMS REPRODUCED — core is trustworthy" if all_ok
          else "SOME CHECKS FAILED — do not build on this core")
    print("=" * 76)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
