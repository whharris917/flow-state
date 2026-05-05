"""
Benchmark scenarios — free LJ atoms only.

Each scenario constructs a Simulation directly (no Scene, no Sketch, no
Compiler) and returns it ready for the harness to equilibrate and time.

Conventions:
- Reduced LJ units: sigma=epsilon=mass=1 by default.
- Reflecting walls (use_boundaries=True). Open-world is unstable for free LJ
  because Simulation.step() removes any dynamic atom that leaves the box; with
  no PBC available, reflecting walls is the only constant-N option.
- Gravity = 0.
- Wall damping = 1.0 (elastic). Diverges from production default 0.99 to avoid
  a hidden energy sink that would fight the thermostat.
- Berendsen thermostat ON, mix=0.1 (the value hardcoded in Simulation.step).
"""

from __future__ import annotations

import math
import numpy as np

from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Named scenario presets
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


# Registry exposed for the CLI. Insertion order is the recommended display
# order (gas → liquid → dense reads as increasing density / decreasing T).
SCENARIOS = {
    "lj_gas": lj_gas,
    "lj_liquid": lj_liquid,
    "lj_dense": lj_dense,
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
    dt: float = 0.002,
    r_skin: float = 0.3,
    physics_steps: int = 10,
) -> tuple[Simulation, dict]:
    """Build a Simulation at the given (rho*, T*) point with the documented
    benchmark conventions. Returns (sim, metadata).

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
        "wall_damping": 1.0,
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
) -> Simulation:
    """Construct a Simulation with N free LJ atoms ready for equilibration."""
    sim = Simulation(skip_warmup=True)

    sim.world_size = float(L)
    sim.use_boundaries = True
    sim.gravity = 0.0
    sim.dt = float(dt)
    sim.skin_distance = float(r_skin)
    sim.sigma = float(sigma)
    sim.epsilon = float(epsilon)
    sim.damping = 1.0
    sim.target_temp = float(T_star)
    sim.use_thermostat = True
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
