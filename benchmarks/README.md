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
python -m benchmarks                    # lj_liquid at N=5000, default dials
python -m benchmarks -N 10000           # override atom count
python -m benchmarks --samples 100      # more samples for tighter percentiles
python -m benchmarks --json out.json    # also write JSON
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
| JIT priming | one untimed `step(physics_steps)` call | Pays Numba compile + first-rebuild costs outside the timing window. |
| Samples | 50 timed `step()` calls, median reported | Cheap, robust to one-off jitter. |

Scenario constants (reduced LJ units, σ=ε=mass=1):

| Scenario | ρ* | T* | Phase |
|---|---|---|---|
| `lj_liquid` | 0.7 | 1.0 | Liquid; realistic working point |

(`lj_gas` and `lj_dense` scenarios planned for EI-2.)

## Known limitations

- **No PBC.** Reflecting walls are the only stable BC. Wall reflections are a
  non-physical interaction surface; their effect is bounded but real. When PBC
  ships, add periodic scenarios alongside the reflecting ones.
- **No component breakdown.** Only whole-`step()` time is measured. When a
  regression appears, diagnosis requires manual instrumentation. Add
  `--breakdown` in a follow-up if needed.
- **Single-machine baselines.** No CI gating. Numbers are only comparable on
  the same machine across runs; we don't try to normalize across hardware.

## Layout

```
benchmarks/
├── __init__.py
├── README.md       # this file
├── harness.py      # timing + statistics
├── scenarios.py    # scenario builders → (Simulation, metadata)
└── run.py          # CLI entry point
```
