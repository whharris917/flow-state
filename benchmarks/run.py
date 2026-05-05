"""
CLI entry point for the LJ benchmark suite.

Usage:
    python -m benchmarks                                    # lj_liquid at N=5000
    python -m benchmarks --scenario all                     # all 3 scenarios at N=5000
    python -m benchmarks -N 500,1000,5000                   # grid of Ns
    python -m benchmarks -N grid                            # canonical grid
    python -m benchmarks --scenario all -N grid             # full matrix
    python -m benchmarks --baseline                         # write per-host baseline JSON
    python -m benchmarks --scenario lj_dense -N 5000 --json out.json

Run from the flow-state/ directory so `engine.simulation` resolves.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.harness import run_bench, format_result
from benchmarks.scenarios import SCENARIOS


CANONICAL_GRID = (500, 1000, 2000, 5000, 10000, 20000)
DEFAULT_N = 5000

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LJ benchmark suite")
    parser.add_argument(
        "--scenario", default="lj_liquid",
        choices=[*SCENARIOS.keys(), "all"],
        help="scenario to run, or 'all' (default lj_liquid)",
    )
    parser.add_argument(
        "-N", default=str(DEFAULT_N),
        help=f"atom count(s): single int, comma-list, or 'grid' for {CANONICAL_GRID} (default {DEFAULT_N})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--equilibration", type=int, default=200,
                        help="discarded substeps before timing (default 200)")
    parser.add_argument("--physics-steps", type=int, default=10,
                        help="substeps per timed step() call (default 10)")
    parser.add_argument("--json", type=str, default=None,
                        help="write all results to this JSON path")
    parser.add_argument("--baseline", action="store_true",
                        help=f"write {BASELINES_DIR}/<hostname>.json (overrides --json)")
    args = parser.parse_args(argv)

    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    Ns = _parse_N(args.N)

    print(f"# host={socket.gethostname()} python={platform.python_version()} "
          f"platform={platform.platform()}")
    print(f"# samples={args.samples} equilibration={args.equilibration} "
          f"physics_steps={args.physics_steps} seed={args.seed}")

    all_results = []
    for scenario_name in scenario_names:
        builder = SCENARIOS[scenario_name]
        for N in Ns:
            sim, metadata = builder(
                N=N,
                seed=args.seed,
                physics_steps=args.physics_steps,
            )
            print(
                f"# {scenario_name}: rho*={metadata['rho_star']} "
                f"T*={metadata['T_star']} world_size={metadata['world_size']:.4f} "
                f"dt={metadata['dt']} r_skin={metadata['r_skin']} "
                f"wall_damping={metadata['wall_damping']}"
            )
            result = run_bench(
                scenario_name=scenario_name,
                sim=sim,
                physics_steps=args.physics_steps,
                equilibration_substeps=args.equilibration,
                samples=args.samples,
                metadata=metadata,
            )
            print(format_result(result))
            all_results.append(result)

    out_path = _resolve_output_path(args)
    if out_path is not None:
        _write_results_json(out_path, all_results, args)
        print(f"# wrote {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_N(spec: str) -> list[int]:
    """Parse the -N argument: 'grid' | int | comma-separated ints."""
    if spec == "grid":
        return list(CANONICAL_GRID)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"invalid -N value {spec!r}: {exc}")


def _resolve_output_path(args) -> Path | None:
    """Decide where (if anywhere) to write JSON output."""
    if args.baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        return BASELINES_DIR / f"{socket.gethostname()}.json"
    if args.json:
        return Path(args.json)
    return None


def _write_results_json(out_path: Path, results, args) -> None:
    """Write a structured JSON document with results + run-environment context."""
    payload = {
        "schema": "flow-state.benchmarks.v1",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": args.samples,
        "equilibration": args.equilibration,
        "physics_steps": args.physics_steps,
        "seed": args.seed,
        "results": [r.as_dict() for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
