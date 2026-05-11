"""
Component-level breakdown harness.

Mirrors Simulation.step() phase by phase, with time.perf_counter() around each
kernel. Lets us see where the wall time actually goes — which is the
diagnostic the suite needs whenever a regression appears at the whole-step
level. Not normally invoked; runs via `python -m benchmarks --breakdown`.

The phases timed correspond to Simulation.step()'s flow:

  1. check_displacement  — every call, decides whether a rebuild is needed
  2. build_neighbor_list — only when rebuild needed (cell-list construction)
  3. build_atom_neighbor_csr — converts half-pair list to CSR (only on rebuild)
  4. integrate_n_steps   — the substep integration loop with the LJ pair loop
  5. apply_thermostat    — only if sim.use_thermostat
  6. escape_filter       — numpy ops removing out-of-bounds dynamic atoms

Phases that don't run on a given call (e.g., rebuild phases when no rebuild is
needed) get a zero entry, which keeps the medians honest. The reported
"per-call median" of a never-running phase will be 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median

import numpy as np

import core.config as config
from engine.physics_core import (
    integrate_n_steps, build_neighbor_list, check_displacement,
    apply_thermostat, build_atom_neighbor_csr,
)


PHASE_NAMES = (
    "check_displacement",
    "build_neighbor_list",
    "build_atom_neighbor_csr",
    "integrate_n_steps",
    "apply_thermostat",
    "escape_filter",
)


@dataclass
class BreakdownResult:
    scenario: str
    N: int
    physics_steps_requested: int
    samples: int
    actual_substeps_per_call: list[int] = field(default_factory=list)
    rebuilds_per_call: list[int] = field(default_factory=list)
    times_ms: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def median_actual_substeps(self) -> float:
        if not self.actual_substeps_per_call:
            return float(self.physics_steps_requested)
        return median(self.actual_substeps_per_call)

    @property
    def total_median_ms(self) -> float:
        return sum(median(self.times_ms[k]) for k in PHASE_NAMES)

    def per_phase_medians_ms(self) -> dict[str, float]:
        return {k: median(self.times_ms[k]) for k in PHASE_NAMES}

    def rebuild_rate(self) -> float:
        """Fraction of timed calls that triggered a rebuild."""
        if not self.rebuilds_per_call:
            return 0.0
        return sum(self.rebuilds_per_call) / len(self.rebuilds_per_call)


def run_breakdown(
    scenario_name: str,
    sim,
    physics_steps: int,
    *,
    equilibration_substeps: int = 200,
    samples: int = 50,
    metadata: dict | None = None,
) -> BreakdownResult:
    """Run a phase-instrumented benchmark on a pre-built Simulation."""
    _run_substeps(sim, equilibration_substeps)
    for _ in range(3):
        _instrumented_step(sim, physics_steps, record=None)

    times_ms: dict[str, list[float]] = {k: [] for k in PHASE_NAMES}
    actual_substeps: list[int] = []
    rebuilds: list[int] = []

    for _ in range(samples):
        before_total = sim.total_steps
        record = {k: 0.0 for k in PHASE_NAMES}
        rebuilt = _instrumented_step(sim, physics_steps, record=record)
        after_total = sim.total_steps

        for k in PHASE_NAMES:
            times_ms[k].append(record[k])
        actual_substeps.append(after_total - before_total)
        rebuilds.append(1 if rebuilt else 0)

    return BreakdownResult(
        scenario=scenario_name,
        N=sim.count,
        physics_steps_requested=physics_steps,
        samples=samples,
        actual_substeps_per_call=actual_substeps,
        rebuilds_per_call=rebuilds,
        times_ms=times_ms,
        metadata=metadata or {},
    )


def _instrumented_step(sim, physics_steps: int, *, record: dict | None) -> bool:
    """Run one step() worth of work with optional per-phase timing.

    Returns True if a neighbour-list rebuild was performed this call. When
    record is None the phases still run but timings aren't captured (priming).
    """
    rebuilt = False

    # Phase 1: displacement check
    should_rebuild = sim.rebuild_next
    if not should_rebuild and sim.count > 0:
        t0 = time.perf_counter() if record is not None else None
        should_rebuild = check_displacement(
            sim.pos_x[:sim.count], sim.pos_y[:sim.count],
            sim.last_x[:sim.count], sim.last_y[:sim.count],
            sim.r_skin_sq_limit,
        )
        if record is not None:
            record["check_displacement"] = (time.perf_counter() - t0) * 1e3

    # Phase 2 + 3: rebuild path
    if should_rebuild and sim.count > 0:
        rebuilt = True
        t0 = time.perf_counter() if record is not None else None
        while True:
            count = build_neighbor_list(
                sim.pos_x[:sim.count], sim.pos_y[:sim.count],
                sim.r_list2, sim.cell_size, sim.world_size,
                sim.pair_i, sim.pair_j,
                sim.boundary_mode,
            )
            if count >= sim.max_pairs:
                sim.max_pairs *= 2
                sim.pair_i = np.zeros(sim.max_pairs, dtype=np.int32)
                sim.pair_j = np.zeros(sim.max_pairs, dtype=np.int32)
                sim.nbr_idx = np.zeros(2 * sim.max_pairs, dtype=np.int32)
                continue
            sim.pair_count = count
            break
        if record is not None:
            record["build_neighbor_list"] = (time.perf_counter() - t0) * 1e3

        sim.last_x[:sim.count] = sim.pos_x[:sim.count]
        sim.last_y[:sim.count] = sim.pos_y[:sim.count]
        sim.rebuild_next = False

        t0 = time.perf_counter() if record is not None else None
        build_atom_neighbor_csr(
            sim.count, sim.pair_i, sim.pair_j, sim.pair_count,
            sim.nbr_start[:sim.count + 1], sim.nbr_idx,
        )
        if record is not None:
            record["build_atom_neighbor_csr"] = (time.perf_counter() - t0) * 1e3

    # Phase 4: integrate
    if sim.count > 0:
        t0 = time.perf_counter() if record is not None else None
        bc = sim.bond_count
        steps_done = integrate_n_steps(
            physics_steps,
            sim.pos_x[:sim.count], sim.pos_y[:sim.count],
            sim.vel_x[:sim.count], sim.vel_y[:sim.count],
            sim.force_x[:sim.count], sim.force_y[:sim.count],
            sim.last_x[:sim.count], sim.last_y[:sim.count],
            sim.is_static[:sim.count],
            sim.atom_sigma[:sim.count], sim.atom_eps_sqrt[:sim.count],
            sim.atom_mass[:sim.count],
            sim.nbr_start[:sim.count + 1], sim.nbr_idx,
            sim.tether_entity_idx[:sim.count],
            sim.joint_ids[:sim.count],
            sim.bond_i[:bc], sim.bond_j[:bc],
            sim.bond_k[:bc], sim.bond_r_eq[:bc],
            np.float32(sim.dt), np.float32(sim.gravity),
            np.float32(sim.r_cut_base**2), np.float32(sim.r_skin_sq_limit),
            np.float32(sim.world_size), sim.boundary_mode,
            np.float32(sim.damping),
        )
        if record is not None:
            record["integrate_n_steps"] = (time.perf_counter() - t0) * 1e3

        sim.total_steps += steps_done
        if steps_done < physics_steps:
            sim.rebuild_next = True

        # Phase 5: thermostat
        if sim.use_thermostat:
            t0 = time.perf_counter() if record is not None else None
            apply_thermostat(
                sim.vel_x[:sim.count], sim.vel_y[:sim.count],
                np.float32(config.ATOM_MASS), sim.is_static[:sim.count],
                np.float32(sim.target_temp), np.float32(0.1),
            )
            if record is not None:
                record["apply_thermostat"] = (time.perf_counter() - t0) * 1e3

        # Phase 6: escape filter (matches Simulation.step body verbatim)
        t0 = time.perf_counter() if record is not None else None
        active_x = sim.pos_x[:sim.count]
        active_y = sim.pos_y[:sim.count]
        active_static = sim.is_static[:sim.count]
        w = sim.world_size
        is_inside = (active_x >= 0) & (active_x <= w) & (active_y >= 0) & (active_y <= w)
        keep = is_inside | (active_static != 0)
        if not np.all(keep):
            keep_indices = np.where(keep)[0]
            sim.compact_arrays(keep_indices)
            sim.rebuild_next = True
        if record is not None:
            record["escape_filter"] = (time.perf_counter() - t0) * 1e3

    return rebuilt


def _run_substeps(sim, total_substeps: int, chunk: int = 50) -> None:
    """Equilibration via the real sim.step() — same code path as production."""
    remaining = total_substeps
    while remaining > 0:
        n = min(chunk, remaining)
        sim.step(steps_to_run=n)
        remaining -= n


def format_breakdown(r: BreakdownResult) -> str:
    """Multi-line breakdown summary.

    For phases that only run on a fraction of calls (build_neighbor_list,
    build_atom_neighbor_csr — only on rebuild), we report the **mean** across
    all calls, not the median. Mean correctly amortizes the rebuild cost over
    every call. For phases that run every call, mean and median are usually
    close; we still use mean here so all rows compose to a meaningful total.
    """
    rebuild_rate_pct = r.rebuild_rate() * 100.0

    means = {phase: (sum(r.times_ms[phase]) / len(r.times_ms[phase]))
             if r.times_ms[phase] else 0.0
             for phase in PHASE_NAMES}
    total = sum(means.values())

    lines = [
        f"  {r.scenario:<14} N={r.N:>6}  steps={int(r.median_actual_substeps)}/"
        f"{r.physics_steps_requested}  rebuild_rate={rebuild_rate_pct:.0f}%  "
        f"total_mean={total:.3f} ms"
    ]
    for phase in PHASE_NAMES:
        ms = means[phase]
        pct = (ms / total * 100) if total > 0 else 0.0
        suffix = ""
        if phase in ("build_neighbor_list", "build_atom_neighbor_csr") and rebuild_rate_pct > 0:
            # Show what each rebuild costs, in addition to the amortized mean.
            per_rebuild = ms / r.rebuild_rate() if r.rebuild_rate() > 0 else 0.0
            suffix = f"  [per-rebuild: {per_rebuild:.4f} ms]"
        lines.append(f"    {phase:<26} = {ms:>7.4f} ms   ({pct:>4.1f}%){suffix}")
    return "\n".join(lines)
