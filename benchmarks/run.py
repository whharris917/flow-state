"""
CLI entry point for the LJ benchmark suite.

Usage:
    python -m benchmarks                       # default: lj_liquid at N=5000
    python -m benchmarks -N 10000              # override N
    python -m benchmarks --samples 100         # more samples for tighter stats
    python -m benchmarks --json out.json       # also write JSON

Run from the flow-state/ directory so `engine.simulation` resolves.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.harness import run_bench, format_result
from benchmarks.scenarios import lj_liquid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LJ benchmark suite")
    parser.add_argument("-N", type=int, default=5000, help="atom count (default 5000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--equilibration", type=int, default=200,
                        help="discarded substeps before timing (default 200)")
    parser.add_argument("--physics-steps", type=int, default=10,
                        help="substeps per timed step() call (default 10)")
    parser.add_argument("--json", type=str, default=None,
                        help="optional path to write JSON result")
    args = parser.parse_args(argv)

    sim, metadata = lj_liquid(
        N=args.N,
        seed=args.seed,
        physics_steps=args.physics_steps,
    )

    print(f"# scenario=lj_liquid N={args.N} seed={args.seed} "
          f"samples={args.samples} equilibration={args.equilibration}")
    print(f"# rho*={metadata['rho_star']} T*={metadata['T_star']} "
          f"world_size={metadata['world_size']:.4f} dt={metadata['dt']} "
          f"r_skin={metadata['r_skin']} wall_damping={metadata['wall_damping']}")

    result = run_bench(
        scenario_name="lj_liquid",
        sim=sim,
        physics_steps=args.physics_steps,
        equilibration_substeps=args.equilibration,
        samples=args.samples,
        metadata=metadata,
    )

    print(format_result(result))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result.as_dict(), f, indent=2)
        print(f"# wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
