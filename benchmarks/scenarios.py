"""
Benchmark scenarios — free LJ atoms only.

Each scenario constructs a Simulation directly (no Scene, no Sketch, no
Compiler) and returns it ready for the harness to equilibrate and time.

Two families of scenarios:

- *Controlled* (lj_gas, lj_liquid, lj_dense): isothermal at a chosen (rho*, T*)
  via the Berendsen thermostat, gravity zero, wall damping 1.0. These are the
  regression-friendly scenarios — same physics every time.

- *Production-like* (lj_production): no thermostat, gravity = config.DEFAULT_GRAVITY,
  wall damping = config.DEFAULT_DAMPING. Mirrors the actual conditions Flow State
  runs in. Higher dispersion; use to check that controlled-scenario findings
  hold under realistic load.

All scenarios share:
- Reduced LJ units (sigma = epsilon = mass = 1) by default.
- Reflecting walls (use_boundaries=True). Open-world is unstable for free LJ
  because Simulation.step() removes any dynamic atom that leaves the box; with
  no PBC available, reflecting walls is the only constant-N option.
- Numerical defaults (dt, r_skin) read from shared.config so the benchmarks
  automatically track production tuning.
"""

from __future__ import annotations

import math
import numpy as np

from engine.simulation import Simulation
from shared import config


# Numerical-dial defaults inherit from production config so a tuning change in
# config.py automatically flows into the benchmark scenarios.
_DEFAULT_DT = config.DEFAULT_DT
_DEFAULT_R_SKIN = config.DEFAULT_SKIN_DISTANCE


# ---------------------------------------------------------------------------
# Named scenario presets — controlled (isothermal, gravity off)
# ---------------------------------------------------------------------------


def lj_gas(N: int, **overrides) -> tuple[Simulation, dict]:
    """Dilute gas point: rho* = 0.05, T* = 2.0. Few neighbours per atom; the
    neighbour-list bookkeeping cost is the dominant term."""
    return _build_lj_scenario(
        name="lj_gas", N=N, rho_star=0.05, T_star=2.0, **overrides
    )


def lj_liquid(N: int, **overrides) -> tuple[Simulation, dict]:
    """Liquid point: rho* = 0.7, T* = 1.0. Realistic working point — the
    canonical 'production-like' benchmark."""
    return _build_lj_scenario(
        name="lj_liquid", N=N, rho_star=0.7, T_star=1.0, **overrides
    )


def lj_dense(N: int, **overrides) -> tuple[Simulation, dict]:
    """Dense liquid near the triple point: rho* = 0.85, T* = 0.7. Many pairs
    per atom; the LJ force loop dominates."""
    return _build_lj_scenario(
        name="lj_dense", N=N, rho_star=0.85, T_star=0.7, **overrides
    )


def lj_sweep(
    N: int,
    *,
    rho_star: float = 0.7,
    T_star: float = 1.0,
    **overrides,
) -> tuple[Simulation, dict]:
    """Parametric scenario for sweeps. Defaults are the lj_liquid point but
    every dial — rho*, T*, sigma, epsilon, dt, r_skin, physics_steps — is
    individually overrideable. The CLI's --sweep flag iterates over values of
    a single keyword through this builder."""
    return _build_lj_scenario(
        name="lj_sweep", N=N, rho_star=rho_star, T_star=T_star, **overrides
    )


# ---------------------------------------------------------------------------
# Production-like scenario — gravity on, no thermostat, prod wall damping
# ---------------------------------------------------------------------------


def lj_production(N: int, **overrides) -> tuple[Simulation, dict]:
    """Production-conditions scenario: gravity ON, thermostat OFF, wall damping
    matches config.DEFAULT_DAMPING. Initial layout matches lj_liquid (rho*=0.7,
    T*=1.0 at substep 0) but the system evolves freely under gravity from there.

    Used to verify that tuning decisions made under controlled scenarios still
    hold when production conditions kick in. Expect higher dispersion — the
    system isn't in steady state and gravity continuously injects energy that
    wall damping bleeds off.
    """
    return _build_lj_scenario(
        name="lj_production",
        N=N,
        rho_star=0.7,
        T_star=1.0,
        thermostat=False,
        gravity=config.DEFAULT_GRAVITY,
        wall_damping=config.DEFAULT_DAMPING,
        **overrides,
    )


# Registry exposed for the CLI. Insertion order is the recommended display
# order. lj_sweep is registered separately because --scenario all skips it.
SCENARIOS = {
    "lj_gas": lj_gas,
    "lj_liquid": lj_liquid,
    "lj_dense": lj_dense,
    "lj_production": lj_production,
    "lj_sweep": lj_sweep,
}


# ---------------------------------------------------------------------------
# Underlying builder
# ---------------------------------------------------------------------------


def _build_lj_scenario(
    *,
    name: str,
    N: int,
    rho_star: float,
    T_star: float,
    seed: int = 0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    dt: float = _DEFAULT_DT,
    r_skin: float = _DEFAULT_R_SKIN,
    physics_steps: int = 10,
    thermostat: bool = True,
    gravity: float = 0.0,
    wall_damping: float = 1.0,
) -> tuple[Simulation, dict]:
    """Build a Simulation at the given (rho*, T*) point.

    `name` is propagated into metadata.scenario so the harness reports it
    consistently regardless of which preset wrapper invoked the builder.
    """
    L = math.sqrt(N / rho_star) * sigma
    sim = _build_lj_sim(
        N=N,
        seed=seed,
        L=L,
        T_star=T_star,
        sigma=sigma,
        epsilon=epsilon,
        dt=dt,
        r_skin=r_skin,
        thermostat=thermostat,
        gravity=gravity,
        wall_damping=wall_damping,
    )
    metadata = {
        "scenario": name,
        "rho_star": rho_star,
        "T_star": T_star,
        "world_size": L,
        "sigma": sigma,
        "epsilon": epsilon,
        "dt": dt,
        "r_skin": r_skin,
        "thermostat": thermostat,
        "gravity": gravity,
        "wall_damping": wall_damping,
        "seed": seed,
        "physics_steps": physics_steps,
    }
    return sim, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_lj_sim(
    *,
    N: int,
    seed: int,
    L: float,
    T_star: float,
    sigma: float,
    epsilon: float,
    dt: float,
    r_skin: float,
    thermostat: bool,
    gravity: float,
    wall_damping: float,
) -> Simulation:
    """Construct a Simulation with N free LJ atoms ready for equilibration."""
    sim = Simulation(skip_warmup=True)

    sim.world_size = float(L)
    sim.use_boundaries = True
    sim.gravity = float(gravity)
    sim.dt = float(dt)
    sim.skin_distance = float(r_skin)
    sim.sigma = float(sigma)
    sim.epsilon = float(epsilon)
    sim.damping = float(wall_damping)
    sim.target_temp = float(T_star)
    sim.use_thermostat = bool(thermostat)
    sim._update_derived_params()

    if N > sim.capacity:
        while sim.capacity < N:
            sim.capacity *= 2
        sim._resize_arrays()

    rng = np.random.default_rng(seed)
    pos = _lattice_positions(N, L, sigma, rng)
    vel = _maxwell_boltzmann_velocities(N, T_star, mass=1.0, rng=rng)

    sim.count = N
    sim.pos_x[:N] = pos[:, 0].astype(np.float32)
    sim.pos_y[:N] = pos[:, 1].astype(np.float32)
    sim.vel_x[:N] = vel[:, 0].astype(np.float32)
    sim.vel_y[:N] = vel[:, 1].astype(np.float32)
    sim.force_x[:N] = 0.0
    sim.force_y[:N] = 0.0
    sim.is_static[:N] = 0
    sim.atom_sigma[:N] = sigma
    sim.atom_eps_sqrt[:N] = math.sqrt(epsilon)
    sim.tether_entity_idx[:N] = -1
    sim.joint_ids[:N] = 0
    sim.last_x[:N] = sim.pos_x[:N]
    sim.last_y[:N] = sim.pos_y[:N]
    sim.rebuild_next = True

    return sim


def _lattice_positions(N: int, L: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Place N atoms on a perturbed square lattice fitting inside [0, L] x [0, L].

    Lattice spacing is L / ceil(sqrt(N)) so all sites sit strictly inside the
    box. Each position is jittered by up to 5% of the lattice spacing in each
    axis to break exact symmetry without risking LJ overlap.
    """
    side = math.ceil(math.sqrt(N))
    a = L / side
    jitter_amp = 0.05 * a

    coords = np.empty((N, 2), dtype=np.float64)
    for k in range(N):
        i = k % side
        j = k // side
        coords[k, 0] = (i + 0.5) * a
        coords[k, 1] = (j + 0.5) * a

    coords += rng.uniform(-jitter_amp, jitter_amp, size=coords.shape)
    return coords


def _maxwell_boltzmann_velocities(
    N: int, T: float, mass: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample 2D MB velocities at temperature T, then zero net momentum and
    rescale to land exactly at <KE> = T per DOF (2D: <KE_total> = N * T).
    """
    sigma_v = math.sqrt(T / mass)
    v = rng.normal(0.0, sigma_v, size=(N, 2))

    v -= v.mean(axis=0)

    ke = 0.5 * mass * np.sum(v * v)
    target_ke = N * T
    if ke > 0:
        v *= math.sqrt(target_ke / ke)
    return v
