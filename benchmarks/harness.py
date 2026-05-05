"""
Benchmark harness — timing and statistics for Simulation.step() calls.

Reports the canonical metric ns/atom-substep, plus ms/step() and MASPS, with
median, p10, p90, min, max over k samples. Equilibrates and primes the Numba
JIT before the timed loop so first-call compile time never enters the sample.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from statistics import median


@dataclass
class BenchResult:
    scenario: str
    N: int
    physics_steps: int
    samples: int
    times_ms: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def median_ms(self) -> float:
        return median(self.times_ms)

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def p10_ms(self) -> float:
        return _percentile(self.times_ms, 10)

    @property
    def p90_ms(self) -> float:
        return _percentile(self.times_ms, 90)

    @property
    def ns_per_atom_substep(self) -> float:
        return self.median_ms * 1e6 / (self.N * self.physics_steps)

    @property
    def masps(self) -> float:
        return (self.N * self.physics_steps) / (self.median_ms * 1e3)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["median_ms"] = self.median_ms
        d["min_ms"] = self.min_ms
        d["max_ms"] = self.max_ms
        d["p10_ms"] = self.p10_ms
        d["p90_ms"] = self.p90_ms
        d["ns_per_atom_substep"] = self.ns_per_atom_substep
        d["masps"] = self.masps
        return d


def run_bench(
    scenario_name: str,
    sim,
    physics_steps: int,
    *,
    equilibration_substeps: int = 200,
    samples: int = 50,
    metadata: dict | None = None,
) -> BenchResult:
    """
    Run a benchmark on a pre-built Simulation.

    Calling convention:
      1. Run equilibration_substeps of physics with the thermostat on so the
         system relaxes to the target temperature. Discarded.
      2. Run 3 priming step() calls equal in size to the timed call. Discarded.
         The first call pays Numba's JIT compile; the next two cover the
         parallel-pool spin-up (thread creation is lazy and not always done in
         one call). One priming call is empirically not enough — first-sample
         outliers of ~500 ms have been observed without this triple-prime.
      3. Run `samples` timed step(physics_steps) calls. Recorded.

    Returns a BenchResult; caller decides whether to print, persist, or compare.
    """
    _run_substeps(sim, equilibration_substeps)
    for _ in range(3):
        sim.step(steps_to_run=physics_steps)

    times_ms: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        sim.step(steps_to_run=physics_steps)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    return BenchResult(
        scenario=scenario_name,
        N=sim.count,
        physics_steps=physics_steps,
        samples=samples,
        times_ms=times_ms,
        metadata=metadata or {},
    )


def _run_substeps(sim, total_substeps: int, chunk: int = 50) -> None:
    """Step a sim through total_substeps in chunks of `chunk` substeps per call."""
    remaining = total_substeps
    while remaining > 0:
        n = min(chunk, remaining)
        sim.step(steps_to_run=n)
        remaining -= n


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def format_result(r: BenchResult) -> str:
    """Single-line human-readable summary of a result.

    Scenario metadata (rho*, T*, dt, r_skin, ...) is intentionally not printed
    here — the caller owns that header so it's printed once, not once per row.
    """
    return (
        f"{r.scenario:<14} N={r.N:>6}  steps={r.physics_steps:>3}  "
        f"ns/atom-substep={r.ns_per_atom_substep:>7.2f}  "
        f"ms/step={r.median_ms:>7.3f} (p10={r.p10_ms:.3f}, p90={r.p90_ms:.3f})  "
        f"MASPS={r.masps:>6.2f}"
    )
