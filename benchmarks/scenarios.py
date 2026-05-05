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


def lj_liquid(
    N: int,
    *,
    seed: int = 0,
    rho_star: float = 0.7,
    T_star: float = 1.0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    dt: float = 0.002,
    r_skin: float = 0.3,
    physics_steps: int = 10,
) -> tuple[Simulation, dict]:
    """
    Liquid-phase LJ point: rho* = 0.7, T* = 1.0.

    Returns (Simulation, metadata). Metadata records every controlled variable
    so a benchmark report can reproduce the run exactly.
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
