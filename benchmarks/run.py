"""
CLI entry point for the LJ benchmark suite.

Usage:
    python -m benchmarks                                    # lj_liquid at N=5000
    python -m benchmarks --scenario all                     # all 3 scenarios at N=5000
    python -m benchmarks -N 500,1000,5000                   # grid of Ns
    python -m benchmarks -N grid                            # canonical grid
    python -m benchmarks --scenario all -N grid             # full matrix
    python -m benchmarks --baseline                         # write per-host baseline JSON
    python -m benchmarks --compare                          # diff vs on-disk baseline
    python -m benchmarks --scenario lj_sweep --sweep r_skin=0.1,0.3,0.5
    python -m benchmarks --scenario lj_dense -N 5000 --json out.json

Run from the flow-state/ directory so `engine.simulation` resolves.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.breakdown import format_breakdown, run_breakdown
from benchmarks.harness import run_bench, format_result
from benchmarks.scenarios import SCENARIOS


CANONICAL_GRID = (500, 1000, 2000, 5000, 10000, 20000)
DEFAULT_N = 5000

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

# Sweepable scenario kwargs. Restricted to numerical dials so a typo can't
# silently end up as a benign-looking metadata field.
SWEEPABLE_KEYS = {"rho_star", "T_star", "sigma", "epsilon", "dt", "r_skin", "physics_steps"}

# Default regression threshold for --compare (% change in ns/atom-substep).
DEFAULT_REGRESSION_PCT = 5.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LJ benchmark suite")
    parser.add_argument(
        "--scenario", default="lj_liquid",
        choices=[*SCENARIOS.keys(), "all"],
        help="scenario to run, or 'all' (excludes lj_sweep) (default lj_liquid)",
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
    parser.add_argument("--sweep", type=str, default=None,
                        help="parametric sweep: KEY=v1,v2,v3 (e.g. r_skin=0.1,0.3,0.5). "
                             f"Allowed keys: {sorted(SWEEPABLE_KEYS)}")
    parser.add_argument("--json", type=str, default=None,
                        help="write all results to this JSON path")
    parser.add_argument("--baseline", action="store_true",
                        help=f"write {BASELINES_DIR}/<hostname>.json (overrides --json)")
    parser.add_argument("--compare", action="store_true",
                        help="diff a fresh run against the on-disk baseline; exit 1 on regression")
    parser.add_argument("--regression-pct", type=float, default=DEFAULT_REGRESSION_PCT,
                        help=f"--compare: %% increase in ns/atom-substep counted as a regression "
                             f"(default {DEFAULT_REGRESSION_PCT})")
    parser.add_argument("--breakdown", action="store_true",
                        help="component-level timing per scenario (check_displacement, "
                             "build_neighbor_list, build_atom_neighbor_csr, integrate_n_steps, "
                             "apply_thermostat, escape_filter)")
    parser.add_argument("--spatial-sort-once", action="store_true",
                        help="experimental: reorder atoms by cell after equilibration "
                             "(once, before priming) to test whether spatial sorting "
                             "improves LJ pair-loop cache locality")
    args = parser.parse_args(argv)

    if args.scenario == "all":
        scenario_names = [name for name in SCENARIOS if name != "lj_sweep"]
    else:
        scenario_names = [args.scenario]
    Ns = _parse_N(args.N)
    sweep_key, sweep_values = _parse_sweep(args.sweep) if args.sweep else (None, [None])

    print(f"# host={socket.gethostname()} python={platform.python_version()} "
          f"platform={platform.platform()}")
    print(f"# samples={args.samples} equilibration={args.equilibration} "
          f"physics_steps={args.physics_steps} seed={args.seed}")
    if sweep_key is not None:
        print(f"# sweep={sweep_key}={sweep_values}")

    all_results = []
    for scenario_name in scenario_names:
        builder = SCENARIOS[scenario_name]
        for N in Ns:
            for sweep_value in sweep_values:
                kwargs = {
                    "N": N,
                    "seed": args.seed,
                    "physics_steps": args.physics_steps,
                }
                if sweep_key is not None:
                    kwargs[sweep_key] = sweep_value

                sim, metadata = builder(**kwargs)
                if sweep_key is not None:
                    metadata["sweep_key"] = sweep_key
                    metadata["sweep_value"] = sweep_value

                print(
                    f"# {scenario_name}: rho*={metadata['rho_star']} "
                    f"T*={metadata['T_star']} world_size={metadata['world_size']:.4f} "
                    f"dt={metadata['dt']} r_skin={metadata['r_skin']} "
                    f"wall_damping={metadata['wall_damping']}"
                )
                # The harness reads physics_steps off the call args, not metadata.
                steps = metadata.get("physics_steps", args.physics_steps)
                if args.breakdown:
                    result = run_breakdown(
                        scenario_name=scenario_name,
                        sim=sim,
                        physics_steps=steps,
                        equilibration_substeps=args.equilibration,
                        samples=args.samples,
                        metadata=metadata,
                    )
                    print(format_breakdown(result))
                else:
                    result = run_bench(
                        scenario_name=scenario_name,
                        sim=sim,
                        physics_steps=steps,
                        equilibration_substeps=args.equilibration,
                        samples=args.samples,
                        metadata=metadata,
                        spatial_sort_once=args.spatial_sort_once,
                    )
                    print(format_result(result))
                all_results.append(result)

    if args.compare:
        return _do_compare(all_results, args.regression_pct)

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


def _parse_sweep(spec: str) -> tuple[str, list]:
    """Parse --sweep: KEY=v1,v2,v3. Coerces values to int if KEY is physics_steps."""
    if "=" not in spec:
        raise SystemExit(f"--sweep expects KEY=v1,v2,...; got {spec!r}")
    key, values_str = spec.split("=", 1)
    key = key.strip()
    if key not in SWEEPABLE_KEYS:
        raise SystemExit(f"--sweep key {key!r} not in {sorted(SWEEPABLE_KEYS)}")
    raw = [v.strip() for v in values_str.split(",") if v.strip()]
    if not raw:
        raise SystemExit(f"--sweep needs at least one value")
    coerce = int if key == "physics_steps" else float
    try:
        values = [coerce(v) for v in raw]
    except ValueError as exc:
        raise SystemExit(f"invalid --sweep value: {exc}")
    return key, values


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


def _do_compare(fresh_results, regression_pct: float) -> int:
    """Diff fresh results against the on-disk baseline. Exit code 1 if any
    (scenario, N) regresses by more than `regression_pct` on ns_per_atom_substep.
    Missing-from-baseline entries are reported but don't fail the run."""
    baseline_path = BASELINES_DIR / f"{socket.gethostname()}.json"
    if not baseline_path.exists():
        print(f"# no baseline at {baseline_path}; run with --baseline first")
        return 1

    with open(baseline_path) as f:
        baseline_doc = json.load(f)
    baseline_index = {
        (r["scenario"], r["N"]): r for r in baseline_doc.get("results", [])
    }

    print()
    print(f"# compare vs {baseline_path} (threshold {regression_pct:.1f}%)")
    print(f"# baseline timestamp: {baseline_doc.get('timestamp_utc', '?')}")
    header = f"  {'scenario':<14} {'N':>6}  {'baseline':>10}  {'fresh':>10}  {'delta':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    any_regression = False
    for r in fresh_results:
        key = (r.scenario, r.N)
        baseline_r = baseline_index.get(key)
        if baseline_r is None:
            print(f"  {r.scenario:<14} {r.N:>6}  {'(new)':>10}  "
                  f"{r.ns_per_atom_substep:>10.2f}  {'-':>8}")
            continue
        b_ns = baseline_r["ns_per_atom_substep"]
        f_ns = r.ns_per_atom_substep
        delta_pct = (f_ns - b_ns) / b_ns * 100.0
        marker = ""
        if delta_pct > regression_pct:
            marker = "  REGRESSION"
            any_regression = True
        elif delta_pct < -regression_pct:
            marker = "  improvement"
        print(f"  {r.scenario:<14} {r.N:>6}  {b_ns:>10.2f}  {f_ns:>10.2f}  "
              f"{delta_pct:>+7.2f}%{marker}")

    return 1 if any_regression else 0


if __name__ == "__main__":
    sys.exit(main())
