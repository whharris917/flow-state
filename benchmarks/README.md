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
python -m benchmarks                                       # lj_liquid at N=5000
python -m benchmarks --scenario all                        # 4 scenarios at N=5000
python -m benchmarks -N 500,1000,5000                      # comma-list of Ns
python -m benchmarks -N grid                               # canonical grid (500..20k)
python -m benchmarks --scenario all -N grid                # full matrix
python -m benchmarks --baseline                            # write baselines/<host>.json
python -m benchmarks --compare                             # diff fresh run vs baseline
python -m benchmarks --scenario lj_sweep --sweep r_skin=0.1,0.3,0.5
python -m benchmarks --json out.json                       # arbitrary JSON path
```

## What is measured

Three numbers per scenario, primary first:

1. **`ns_per_atom_substep`** — `(median_ms × 1e6) / (N × actual_substeps)`.
   Cross-N comparable. Watch this for regressions.
2. **`median_ms` per `step()` call** — what the user feels per frame in the
   running app.
3. **MASPS** — million atom-substeps-per-second. Inverse of #1, headline-friendly.

Reported alongside: `min_ms`, `max_ms`, `p10_ms`, `p90_ms`, 20%-trimmed mean,
and `median_actual_substeps` (see "actual vs requested substeps" below).

## Methodology

| Variable | Default | Reason |
|---|---|---|
| Boundary | reflecting walls (`use_boundaries=True`) | Only stable BC for free LJ at constant N. Open-world bleeds atoms via the per-step escape filter; PBC isn't implemented. |
| Gravity | 0 (controlled scenarios), `config.DEFAULT_GRAVITY` (lj_production) | Controlled scenarios isolate LJ throughput from gravitational drift; lj_production includes it. |
| Thermostat | Berendsen on, mix=0.1, target=T* (controlled); off (lj_production) | Pins T in controlled runs so kinetic energy doesn't drift between runs. lj_production omits it because production omits it. |
| Wall damping | 1.0 elastic (controlled); `config.DEFAULT_DAMPING` (lj_production) | Avoids a hidden energy sink fighting the thermostat in controlled runs. |
| Initial layout | perturbed square lattice (5% jitter), fixed seed | Avoids LJ overlap on startup; deterministic. |
| Initial velocities | Maxwell-Boltzmann at T*, fixed seed, zero net momentum, rescaled to exact target KE | Deterministic, no spurious drift, lands at the target temperature on substep 0. |
| Equilibration | 200 substeps before timing | Discards the relaxation transient. Increase for production scenarios (gravity not at steady state). |
| JIT priming | 3 untimed `step(physics_steps)` calls | Pays Numba compile and parallel-pool spin-up before timing starts. One call is empirically not enough. |
| Samples | 50 timed `step()` calls, median reported | Cheap, robust to one-off jitter. |
| Numerical defaults | `dt`, `r_skin` read from `shared.config` | Benchmarks track production tuning automatically. |

## Scenarios

Reduced LJ units throughout (σ=ε=mass=1):

| Scenario | ρ* | T* | Gravity | Thermostat | Wall damping | What it stresses |
|---|---|---|---|---|---|---|
| `lj_gas` | 0.05 | 2.0 | 0 | on | 1.0 | Few neighbours/atom — neighbour-list bookkeeping cost dominates |
| `lj_liquid` | 0.7 | 1.0 | 0 | on | 1.0 | Realistic working point in controlled conditions |
| `lj_dense` | 0.85 | 0.7 | 0 | on | 1.0 | Many pairs/atom — LJ force loop dominates |
| `lj_production` | 0.7 | 1.0 | `config.DEFAULT_GRAVITY` | off | `config.DEFAULT_DAMPING` | Mirrors actual production conditions |
| `lj_sweep` | configurable | configurable | configurable | configurable | configurable | Used by `--sweep` to vary one numerical dial |

The canonical N grid is `(500, 1000, 2000, 5000, 10000, 20000)`, selectable via `-N grid`.

## Sweeps and comparisons

`--sweep KEY=v1,v2,v3` iterates the chosen scenario over values of one
numerical dial. Allowed keys: `rho_star`, `T_star`, `sigma`, `epsilon`,
`dt`, `r_skin`, `physics_steps`. Use with `--scenario lj_sweep` if you want
to vary `rho_star` or `T_star` without the named-preset wrappers
overriding them.

`--compare` runs the configured benchmarks and diffs the fresh
`ns_per_atom_substep` against the on-disk baseline for the same
(scenario, N) pairs. Exits with code 1 if any pair regresses by more than
`--regression-pct` (default 5%). Run `--baseline` first to seed the file.

## Baselines

`--baseline` writes `benchmarks/baselines/<hostname>.json` for the current
run. The directory is `.gitignored` — baselines are per-machine and shouldn't
pollute the repo, since absolute timings aren't comparable across hardware.

## Actual vs requested substeps

`Simulation.step(physics_steps=k)` calls `integrate_n_steps(steps_to_run=k)`,
which **can return early** when its periodic displacement-safety check (every
`SUBSTEP_SAFETY_CHECK_FREQ`=20 substeps) detects atoms have drifted past
`r_skin/2`. The kernel returns the actual count via `sim.total_steps`.

The harness records actual substeps per call and reports throughput based on
those, not the requested count. When the two diverge, `format_result` shows
`actual/requested` so the early exit is visible.

This matters: at high `physics_steps` (50, 100), the kernel saturates at ~20
substeps per call regardless. A naive harness would report inflated throughput
and falsely suggest raising `DEFAULT_DRAW_M`.

## Component-level breakdown (`--breakdown`)

`--breakdown` runs a phase-instrumented harness that mirrors `Simulation.step()`
and times each kernel separately:

```
benchmarks/breakdown.py     → run_breakdown(scenario, sim, ...)
                            → format_breakdown(result)
```

Phases reported (means across all calls, not medians — so phases that only
run on a fraction of calls amortize correctly):

1. `check_displacement` — every call
2. `build_neighbor_list` — only on rebuild
3. `build_atom_neighbor_csr` — only on rebuild
4. `integrate_n_steps` — every call
5. `apply_thermostat` — only if thermostat is on
6. `escape_filter` — every call

Rebuild phases also report per-rebuild cost. Use this to diagnose
regressions: when whole-step time moves, breakdown tells you which kernel.

## Worked example: r_skin tuning

This is the suite's first real-world story.

Under `lj_liquid` controlled conditions at N=5000, an `r_skin` sweep showed:

| r_skin | ns/atom-substep | vs default 0.3 |
|---|---|---|
| 0.3 (current default) | 49.9 | — |
| 0.5 | 35.7 | -28% |
| 0.7 | 32.4 (best) | -35% |

Looked like a clear win. But running the same sweep under `lj_production`
(gravity on, no thermostat) showed:

| r_skin | ns/atom-substep | vs default 0.3 |
|---|---|---|
| 0.3 | 87.6 | — |
| 0.5 | 103.9 | **+19% worse** |

The controlled-conditions optimization didn't transfer. Gravity-driven motion
keeps atoms drifting fast enough that a larger r_skin doesn't reduce rebuild
frequency proportionally to the larger pair list. The current default 0.3 is
correct for production; do not change it.

This is what the suite is for: catching false-positive optimizations before
they ship.

## Known limitations

- **No PBC.** Reflecting walls are the only stable BC. Wall reflections are a
  non-physical interaction surface; their effect is bounded but real. When
  PBC ships, add periodic scenarios alongside the reflecting ones.
- **No component breakdown.** Only whole-`step()` time is measured. When a
  regression appears, diagnosis requires manual instrumentation.
- **No CI gating.** Per-machine, manual-run only.
- **Dispersion at higher N.** p90/p10 ratio is ~2-3× because of intermittent
  neighbour-list rebuilds (rebuilds happen when an atom drifts > r_skin/2).
  The median is robust; the trimmed mean is a useful corroboration. For
  tighter regression detection use `--samples 200` or larger.
- **Run-to-run variance ~5-10%** even on the same machine, depending on
  background load. The 5% default `--regression-pct` is borderline against
  this noise floor.

## Worked example #2: spatial sort

A second story showing the same shape. Classical MD wisdom says sorting
atoms by spatial cell improves LJ pair-loop cache locality. The suite
includes an experiment harness for this:
`benchmarks/experiments.py::external_spatial_sort(sim)` permutes all 16
per-atom arrays by cell key without touching engine code, and the CLI
exposes it as `--spatial-sort-once`.

Result at N=5000:

| Scenario | Baseline | Sorted | Delta |
|---|---|---|---|
| lj_gas | 15.91 | 15.79 | -0.7% (noise) |
| lj_liquid | 42.12 | 54.44 | **+29% worse** |
| lj_dense | 42.41 | 41.45 | -2% (noise) |
| lj_production | 68.83 | 72.15 | +5% (noise) |

Same result at N=10k and N=20k (deltas within noise floor). Spatial sort
is **not** a perf lever in this Flow State configuration. The Numba
parallel atom-centric pair loop already has good locality without explicit
reordering. The experiment harness stays in the suite to document the
negative result and let future engineers re-test if the kernel architecture
changes.

## Layout

```
benchmarks/
├── __init__.py
├── __main__.py     # `python -m benchmarks` entry
├── README.md       # this file
├── harness.py      # timing + statistics + actual-substep tracking
├── scenarios.py    # 5 scenario builders + SCENARIOS registry
├── breakdown.py    # phase-instrumented harness (--breakdown)
├── experiments.py  # spatial-sort and other engine-change experiments
├── run.py          # CLI: --scenario, -N, --sweep, --baseline, --compare,
│                   #      --breakdown, --spatial-sort-once
└── baselines/      # per-host JSON baselines (gitignored)
```
