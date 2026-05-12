"""Demo presets — one-call setup of canonical emergent-phenomena scenarios.

Each preset clears the Simulation and populates it with an initial state
that exhibits a specific physical phenomenon under the LJ kernel + the R2
seeded cross-ε overrides. The user just hits Play and watches.

Presets are deterministic when given a `seed`: same seed, same initial
positions / velocities (subject to NumPy / Python random determinism).

These presets DO clear the scene's dynamic particles. They do NOT touch
Sketch geometry — if you want a wall-bounded world, draw walls first,
then run a preset; the static atoms persist.

Available presets:
- preset_demixing(scene, ...)      — 50:50 Polar/Nonpolar atomic mixture
- preset_micelles(scene, ...)      — Polar solvent + Surfactant molecules
- preset_crystal_anneal(scene, ...) — High-T fluid for cooling demos

All three rebuild the eps_ij_matrix via scene.rebuild() at the end so any
sketch.lj_cross_overrides edits made just before the preset call land in
the kernel.
"""

import math
import random
from typing import Optional

import numpy as np

from core.molecule_commands import AddMoleculeCommand


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _seed_rngs(seed: Optional[int]) -> None:
    """Apply seed to both stdlib `random` and `numpy.random` so position
    sampling and any downstream Maxwell-Boltzmann draws are deterministic."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _clear_dynamic_and_compiled(scene) -> None:
    """Drop ALL particles from the simulation — dynamic, static, tethered —
    and reset bond / angle counts. Presets construct their own initial
    state and don't need anything carried over from previous runs.

    Calls `sim.clear(snapshot=True)` so the action is undoable: hitting
    Ctrl+Z after running a preset restores the previous particle state.
    """
    scene.simulation.clear(snapshot=True)


def _grid_positions(n: int, world_size: float, spacing: float,
                    jitter: float = 0.05) -> np.ndarray:
    """Return up to n (x, y) positions on a centred regular grid with mild
    jitter. The grid sidelength is chosen so the requested count fits;
    actual count may be slightly less if n isn't a perfect square. Caller
    should slice the result to enforce a precise count.

    Jitter (fraction of spacing) breaks the perfectly-periodic initial
    state so the system can equilibrate without crystallising to the grid.
    Default 0.05 = 5% of spacing — enough to break degeneracy, small
    enough to keep atoms safely apart.
    """
    side = int(math.ceil(math.sqrt(n)))
    if side < 1:
        return np.zeros((0, 2), dtype=np.float32)
    total_extent = side * spacing
    x0 = world_size * 0.5 - 0.5 * total_extent
    positions = np.zeros((side * side, 2), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            jx = (np.random.random() - 0.5) * 2.0 * jitter * spacing
            jy = (np.random.random() - 0.5) * 2.0 * jitter * spacing
            positions[i * side + j, 0] = x0 + (i + 0.5) * spacing + jx
            positions[i * side + j, 1] = x0 + (j + 0.5) * spacing + jy
    # Shuffle so callers slicing the first N atoms don't get a perfectly
    # ordered row-major layout (would bias species placement otherwise).
    np.random.shuffle(positions)
    return positions[:n]


def _maxwell_boltzmann_velocity(target_temp: float, mass: float) -> tuple:
    """Sample (vx, vy) from a Maxwell-Boltzmann distribution at target_temp
    for a particle of given mass. Stdlib random.gauss for parity with the
    Source._sample_velocity pattern used elsewhere."""
    if target_temp <= 0.0 or mass <= 0.0:
        return (0.0, 0.0)
    sigma_v = math.sqrt(target_temp / mass)
    return (random.gauss(0.0, sigma_v), random.gauss(0.0, sigma_v))


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

def preset_demixing(scene, n_per_species: int = 80, target_temp: float = 0.5,
                    spacing: float = 1.3, seed: Optional[int] = None) -> dict:
    """50:50 atomic mixture of Polar and Nonpolar — the workhorse demixing
    demo. With the R2 seeded ε_AB=0.25 override, like atoms attract (ε=1.0)
    and cross-pairs barely interact, so the mixture phase-separates into
    domains over a few seconds of simulation time.

    Args:
        scene: target Scene (cleared in place)
        n_per_species: atoms per species (total = 2 * n_per_species)
        target_temp: thermostat setpoint. ~0.5 is in the LJ liquid range;
            lower → tighter clusters; higher → more thermal noise / breaks up.
        spacing: initial grid spacing in σ units. Default 1.3σ is past
            r_min (2^(1/6)σ ≈ 1.122) so atoms don't initially repel.
        seed: RNG seed for reproducibility.

    Returns dict with keys:
        - 'n_polar', 'n_nonpolar': atoms placed of each species
        - 'world_size': the scene's world size (unchanged)
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)

    # Physics setup: gentle drag, no gravity, reflecting walls, thermostat on.
    sim.gravity = 0.0
    sim.damping = 0.995  # very mild — lets atoms diffuse but bleeds runaway KE
    sim.target_temp = target_temp
    sim.use_thermostat = True
    sim.boundary_mode = 1  # REFLECTING

    polar_id = sketch.get_material_index("Polar")
    nonpolar_id = sketch.get_material_index("Nonpolar")
    polar_mat = sketch.get_material("Polar")
    nonpolar_mat = sketch.get_material("Nonpolar")

    total = 2 * n_per_species
    positions = _grid_positions(total, sim.world_size, spacing)
    # Half the shuffled grid → Polar, the other half → Nonpolar.
    n_polar_placed = 0
    n_nonpolar_placed = 0
    for k, (x, y) in enumerate(positions):
        if k < n_per_species:
            vx, vy = _maxwell_boltzmann_velocity(target_temp, polar_mat.mass)
            sim._add_particle(
                float(x), float(y), vx=vx, vy=vy, is_static=0,
                sigma=polar_mat.sigma, epsilon=polar_mat.epsilon,
                mass=polar_mat.mass, color=polar_mat.color,
                material_id=polar_id,
            )
            n_polar_placed += 1
        else:
            vx, vy = _maxwell_boltzmann_velocity(target_temp, nonpolar_mat.mass)
            sim._add_particle(
                float(x), float(y), vx=vx, vy=vy, is_static=0,
                sigma=nonpolar_mat.sigma, epsilon=nonpolar_mat.epsilon,
                mass=nonpolar_mat.mass, color=nonpolar_mat.color,
                material_id=nonpolar_id,
            )
            n_nonpolar_placed += 1

    # Push the current eps_ij_matrix to the kernel (covers any
    # set_lj_cross_override calls the user made before invoking the preset).
    sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())

    return {
        'n_polar': n_polar_placed,
        'n_nonpolar': n_nonpolar_placed,
        'world_size': sim.world_size,
    }


def preset_micelles(scene, n_solvent: int = 100, n_surfactant: int = 12,
                    target_temp: float = 0.5, solvent_spacing: float = 1.3,
                    seed: Optional[int] = None) -> dict:
    """Polar solvent + Surfactant amphiphiles. With the seeded ε defaults
    and a moderate concentration of surfactants (~10-15% by molecule count),
    the surfactants self-organise into micelles: tails (Nonpolar) cluster
    in the interior, heads (Polar) point outward into the solvent.

    Args:
        scene: target Scene (cleared in place)
        n_solvent: solvent (Polar) atom count
        n_surfactant: surfactant molecule count (each molecule is 5 atoms)
        target_temp: thermostat setpoint
        solvent_spacing: initial grid spacing for solvent atoms
        seed: RNG seed

    Returns dict with 'n_solvent', 'n_surfactant_molecules',
    'n_surfactant_atoms', 'world_size'.
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)

    sim.gravity = 0.0
    sim.damping = 0.995
    sim.target_temp = target_temp
    sim.use_thermostat = True
    sim.boundary_mode = 1

    polar_id = sketch.get_material_index("Polar")
    polar_mat = sketch.get_material("Polar")

    # Place solvent atoms on a grid first; molecules go into vacant patches.
    positions = _grid_positions(n_solvent, sim.world_size, solvent_spacing)
    for (x, y) in positions:
        vx, vy = _maxwell_boltzmann_velocity(target_temp, polar_mat.mass)
        sim._add_particle(
            float(x), float(y), vx=vx, vy=vy, is_static=0,
            sigma=polar_mat.sigma, epsilon=polar_mat.epsilon,
            mass=polar_mat.mass, color=polar_mat.color,
            material_id=polar_id,
        )
    n_solvent_placed = sim.count

    # Place surfactant molecules at random positions inside a margin so the
    # full molecule fits within bounds. The Surfactant template spans about
    # 5σ along its long axis; allow 4σ margin.
    surfactant_tpl = sketch.molecules.get('Surfactant')
    if surfactant_tpl is None:
        return {
            'n_solvent': n_solvent_placed,
            'n_surfactant_molecules': 0,
            'n_surfactant_atoms': 0,
            'world_size': sim.world_size,
        }

    margin = 4.0
    placed = 0
    for _ in range(n_surfactant):
        cx = random.uniform(margin, sim.world_size - margin)
        cy = random.uniform(margin, sim.world_size - margin)
        rot = random.uniform(0.0, 2.0 * math.pi)
        # Use the command's execute path so the molecule's atoms get the
        # right material_ids + bonds + angles. Don't push to scene.commands —
        # presets are setup, not user actions.
        cmd = AddMoleculeCommand(scene, surfactant_tpl,
                                 world_pos=(cx, cy), rotation=rot)
        if cmd.execute():
            placed += 1

    n_surfactant_atoms = sim.count - n_solvent_placed
    sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())

    return {
        'n_solvent': n_solvent_placed,
        'n_surfactant_molecules': placed,
        'n_surfactant_atoms': n_surfactant_atoms,
        'world_size': sim.world_size,
    }


def preset_crystal_anneal(scene, n_atoms: int = 120,
                          target_temp: float = 1.5,
                          material: str = "Polar",
                          spacing: float = 1.4,
                          seed: Optional[int] = None) -> dict:
    """Single-species LJ fluid at high T — designed to be cooled. As the
    user dials target_temp down (manually or via a future scheduler), the
    fluid passes through the LJ phase diagram: gas → liquid → 2D
    hexagonal crystal. Default starts at T=1.5 (well into the gas/liquid
    regime); cooling to T~0.3 produces a visible solid.

    Args:
        scene: target Scene (cleared in place)
        n_atoms: total atom count
        target_temp: starting thermostat setpoint (high; expected to be lowered)
        material: which species to use. Default "Polar"; "Heavy" gives a
            denser fluid that crystallises at higher T due to deeper LJ wells.
        spacing: initial grid spacing in σ units
        seed: RNG seed

    Returns dict with 'n_atoms', 'world_size', 'material'.
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)

    sim.gravity = 0.0
    sim.damping = 0.999  # almost-elastic — atoms keep moving so the fluid
                         # has time to thermalise across cooling sweeps
    sim.target_temp = target_temp
    sim.use_thermostat = True
    sim.boundary_mode = 1

    mat_id = sketch.get_material_index(material)
    mat = sketch.get_material(material)

    positions = _grid_positions(n_atoms, sim.world_size, spacing)
    for (x, y) in positions:
        vx, vy = _maxwell_boltzmann_velocity(target_temp, mat.mass)
        sim._add_particle(
            float(x), float(y), vx=vx, vy=vy, is_static=0,
            sigma=mat.sigma, epsilon=mat.epsilon, mass=mat.mass,
            color=mat.color, material_id=mat_id,
        )

    sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())

    return {
        'n_atoms': sim.count,
        'world_size': sim.world_size,
        'material': material,
    }
