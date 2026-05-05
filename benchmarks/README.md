# Flow State Benchmarks

Performance regression suite for the free-LJ particle physics core.

This directory is **not** collected by pytest — benchmarks are slow and noisy
relative to unit tests, and they're meant to be run manually or in dedicated
perf workflows. `tests/` and `benchmarks/` are siblings.

## Scope (CR-116 EI-3)

Free Lennard-Jones atoms only. No tethers, no static atoms, no Sketch
geometry, no constraints. Pure `Simulation.step()` throughput.

## Running

```
cd flow-state
python -m benchmarks                                # lj_liquid at N=5000
python -m benchmarks --scenario all                 # all 3 scenarios at N=5000
python -m benchmarks -N 500,1000,5000               # comma-list of Ns
python -m benchmarks -N grid                        # canonical grid (500..20k)
python -m benchmarks --scenario all -N grid         # full matrix
python -m benchmarks --baseline                     # write baselines/<host>.json
python -m benchmarks --json out.json                # arbitrary JSON path
```

## What is measured

Three numbers per scenario, primary first:

1. **`ns_per_atom_substep`** — `(median_ms × 1e6) / (N × physics_steps)`.
   Cross-N comparable. Watch this for regressions.
2. **`median_ms` per `step()` call** — what the user feels per frame in the
   running app. `step(physics_steps=k)` runs `k` substeps internally.
3. **MASPS** — million-atom-substeps-per-second. Inverse of #1, headline-friendly.

Reported alongside: `min_ms`, `max_ms`, `p10_ms`, `p90_ms`, sample count.

## Methodology

| Variable | Default | Reason |
|---|---|---|
| Boundary | reflecting walls (`use_boundaries=True`) | Only stable BC for free LJ at constant N. Open-world bleeds atoms via the per-step escape filter; PBC isn't implemented. |
| Gravity | 0 | We're benchmarking LJ, not falling. |
| Thermostat | Berendsen, on, mix=0.1, target=T* | Pins temperature so kinetic energy doesn't drift between runs. |
| Wall damping | 1.0 (elastic) | Diverges from production default (0.99). Avoids a hidden energy sink that would fight the thermostat at the boundary. Documented divergence. |
| Initial layout | perturbed square lattice (5% jitter), fixed seed | Avoids LJ overlap on startup; deterministic. |
| Initial velocities | Maxwell-Boltzmann at T*, fixed seed, zero net momentum, rescaled to exact target KE | Deterministic, no spurious drift, lands at the target temperature on substep 0. |
| Equilibration | 200 substeps before timing | Discards the relaxation transient. |
| JIT priming | 3 untimed `step(physics_steps)` calls | Pays Numba compile and parallel-pool spin-up before timing starts. One call is empirically not enough. |
| Samples | 50 timed `step()` calls, median reported | Cheap, robust to one-off jitter. |

Scenario presets (reduced LJ units, σ=ε=mass=1):

| Scenario | ρ* | T* | Phase | What it stresses |
|---|---|---|---|---|
| `lj_gas` | 0.05 | 2.0 | Dilute gas | Few neighbours/atom — neighbour-list bookkeeping cost dominates |
| `lj_liquid` | 0.7 | 1.0 | Liquid | Realistic working point — production-like |
| `lj_dense` | 0.85 | 0.7 | Dense liquid (near triple point) | Many pairs/atom — LJ force loop dominates |

The canonical N grid is `(500, 1000, 2000, 5000, 10000, 20000)`, selectable via `-N grid`.

## Baselines

`--baseline` writes `benchmarks/baselines/<hostname>.json` for the current run.
The directory is `.gitignored` — baselines are per-machine and shouldn't pollute
the repo, since absolute timings aren't comparable across hardware. Workflow:

1. Establish a baseline before a perf-suspect change: `python -m benchmarks --scenario all -N grid --baseline`
2. Make the change, re-run with `--baseline` to overwrite.
3. (EI-3) `--compare` will diff a fresh run against the on-disk baseline.

## Known limitations

- **No PBC.** Reflecting walls are the only stable BC. Wall reflections are a
  non-physical interaction surface; their effect is bounded but real. When PBC
  ships, add periodic scenarios alongside the reflecting ones.
- **No component breakdown.** Only whole-`step()` time is measured. When a
  regression appears, diagnosis requires manual instrumentation. Add
  `--breakdown` in a follow-up if needed.
- **No CI gating.** Per-machine, manual-run only.
- **Dispersion at higher N.** p90/p10 ratio is ~2-3× because of intermittent
  neighbour-list rebuilds (rebuilds happen when an atom drifts > r_skin/2).
  The median is robust; if you need tighter regression detection consider
  `--samples 200` or report a trimmed mean.

## Layout

```
benchmarks/
├── __init__.py
├── __main__.py     # `python -m benchmarks` entry
├── README.md       # this file
├── harness.py      # timing + statistics
├── scenarios.py    # lj_gas, lj_liquid, lj_dense + SCENARIOS registry
├── run.py          # CLI
└── baselines/      # per-host JSON baselines (gitignored)
```
