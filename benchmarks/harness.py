"""
Benchmark harness — timing and statistics for Simulation.step() calls.

Reports the canonical metric ns/atom-substep, plus ms/step() and MASPS, with
median, p10, p90, min, max over k samples. Equilibrates and primes the Numba
JIT before the timed loop so first-call compile time never enters the sample.

Important correctness note: Simulation.step(physics_steps=k) internally calls
integrate_n_steps(steps_to_run=k), which can RETURN EARLY when the displacement
safety check (every SUBSTEP_SAFETY_CHECK_FREQ substeps) detects atoms have
drifted past the skin limit. The kernel's return value is the number of
substeps that actually ran. We track sim.total_steps before/after each timed
call and use the *actual* substep count, not the requested one — otherwise
ns/atom-substep is artificially low at high physics_steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from statistics import median


@dataclass
class BenchResult:
    scenario: str
    N: int
    physics_steps_requested: int
    samples: int
    times_ms: list[float] = field(default_factory=list)
    actual_substeps_per_call: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Backwards-compatible alias for code/tests that reference physics_steps.
    @property
    def physics_steps(self) -> int:
        return self.physics_steps_requested

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
    def trimmed_mean_ms(self) -> float:
        """Mean of the middle 60% of samples — robust to rebuild-induced outliers."""
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        n = len(s)
        k = max(1, int(n * 0.2))  # drop top and bottom 20%
        trimmed = s[k:n - k] if n - k > k else s
        return sum(trimmed) / len(trimmed)

    @property
    def median_actual_substeps(self) -> float:
        """Median of how many substeps integrate_n_steps actually ran per call.
        When this is less than physics_steps_requested, the kernel was exiting
        early via the displacement safety check."""
        if not self.actual_substeps_per_call:
            return float(self.physics_steps_requested)
        return median(self.actual_substeps_per_call)

    @property
    def ns_per_atom_substep(self) -> float:
        """Nanoseconds per atom-substep, using actual substeps run (not requested)."""
        substeps = self.median_actual_substeps
        if substeps <= 0:
            return float("inf")
        return self.median_ms * 1e6 / (self.N * substeps)

    @property
    def masps(self) -> float:
        """Million atom-substeps per second, using actual substeps."""
        substeps = self.median_actual_substeps
        if substeps <= 0:
            return 0.0
        return (self.N * substeps) / (self.median_ms * 1e3)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["median_ms"] = self.median_ms
        d["min_ms"] = self.min_ms
        d["max_ms"] = self.max_ms
        d["p10_ms"] = self.p10_ms
        d["p90_ms"] = self.p90_ms
        d["trimmed_mean_ms"] = self.trimmed_mean_ms
        d["median_actual_substeps"] = self.median_actual_substeps
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
    spatial_sort_once: bool = False,
) -> BenchResult:
    """
    Run a benchmark on a pre-built Simulation.

    Calling convention:
      1. Run equilibration_substeps of physics with the thermostat on so the
         system relaxes to the target temperature. Discarded.
      2. (Optional) If spatial_sort_once is True, reorder all per-atom arrays
         by spatial cell key so the LJ pair loop has improved cache locality.
         This is an experimental knob — the benchmark measures whether
         sorting is worthwhile before justifying an engine change.
      3. Run 3 priming step() calls equal in size to the timed call. Discarded.
         The first call pays Numba's JIT compile; the next two cover the
         parallel-pool spin-up (thread creation is lazy and not always done in
         one call). One priming call is empirically not enough — first-sample
         outliers of ~500 ms have been observed without this triple-prime.
      4. Run `samples` timed step(physics_steps) calls. Each call's wall time
         AND its actual substep count (from sim.total_steps delta) are recorded.

    Returns a BenchResult; caller decides whether to print, persist, or compare.
    """
    _run_substeps(sim, equilibration_substeps)
    if spatial_sort_once:
        from benchmarks.experiments import external_spatial_sort
        external_spatial_sort(sim)
    for _ in range(3):
        sim.step(steps_to_run=physics_steps)

    times_ms: list[float] = []
    actual_substeps: list[int] = []
    for _ in range(samples):
        before_total = sim.total_steps
        t0 = time.perf_counter()
        sim.step(steps_to_run=physics_steps)
        t1 = time.perf_counter()
        after_total = sim.total_steps
        times_ms.append((t1 - t0) * 1e3)
        actual_substeps.append(after_total - before_total)

    return BenchResult(
        scenario=scenario_name,
        N=sim.count,
        physics_steps_requested=physics_steps,
        samples=samples,
        times_ms=times_ms,
        actual_substeps_per_call=actual_substeps,
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

    Reports actual median substeps if it differs from requested — surfaces the
    early-exit safety check that would otherwise be invisible.
    """
    actual = r.median_actual_substeps
    requested = r.physics_steps_requested
    if abs(actual - requested) > 0.5:
        steps_field = f"{int(actual)}/{requested}"
    else:
        steps_field = f"{requested}"
    return (
        f"{r.scenario:<14} N={r.N:>6}  steps={steps_field:>5}  "
        f"ns/atom-substep={r.ns_per_atom_substep:>7.2f}  "
        f"ms/step={r.median_ms:>7.3f} (p10={r.p10_ms:.3f}, p90={r.p90_ms:.3f})  "
        f"trim_mean={r.trimmed_mean_ms:>6.3f}  MASPS={r.masps:>6.2f}"
    )
