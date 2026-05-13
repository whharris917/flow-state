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
- preset_pbc_liquid(scene, density, ...) — Uniform dense LJ liquid under
  periodic boundaries; parameterized by reduced density ρ* = N σ² / L².

The PBC-liquid preset is the only one that requests BOUNDARY_PERIODIC; the
others use REFLECTING walls. Controller-level wrappers call the PBC preset
with several canonical densities for a quick density-spectrum demo.

All presets rebuild the eps_ij_matrix via scene.rebuild() at the end so any
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


def _grid_positions_pbc_fill(side: int, world_size: float,
                             jitter: float = 0.05) -> np.ndarray:
    """Return `side * side` positions on a uniform grid that tiles [0, L)²
    exactly under periodic boundaries. Grid pitch is `world_size / side` so
    no atom ever lands outside the wrap interval, and the layout is
    translationally invariant (no edge bias toward the box centre, unlike
    `_grid_positions` which centres a sub-tile and leaves bordering
    vacuum). Jitter is a fraction of pitch — small enough to keep atoms
    well-separated, large enough to break the perfectly-periodic initial
    state so the LJ kernel doesn't sit on a stationary lattice point.
    Output is shuffled so callers slicing a prefix don't get a row-major
    ordering.
    """
    if side < 1:
        return np.zeros((0, 2), dtype=np.float32)
    pitch = world_size / side
    positions = np.zeros((side * side, 2), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            jx = (np.random.random() - 0.5) * 2.0 * jitter * pitch
            jy = (np.random.random() - 0.5) * 2.0 * jitter * pitch
            # Wrap into [0, L) explicitly — at default jitter the centred
            # cell positions stay clear of the wrap edges, but the modulo
            # is cheap insurance against floating-point drift at the seam.
            x = ((i + 0.5) * pitch + jx) % world_size
            y = ((j + 0.5) * pitch + jy) % world_size
            positions[i * side + j, 0] = x
            positions[i * side + j, 1] = y
    np.random.shuffle(positions)
    return positions


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


def _seed_pbc_demo_state(sim, target_temp: float,
                         damping: float = 0.999) -> bool:
    """Apply the canonical PBC-demo physics setup to `sim`.

    These four presets all want the same thing: no gravity, near-elastic
    damping (let the thermostat do the work), thermostat on at the demo's
    target temperature, and periodic boundaries so the bulk physics isn't
    contaminated by walls. PBC needs the cell list to tile [0, L) at
    least 3 cells deep — `_pbc_safe` checks that. At the default
    world_size=50, cell_size=2.8 this gives 17 cells, so PBC is always
    safe at default settings; only fails if the user has shrunk the
    world below ~8.4 sim units, in which case we fall back to REFLECTING
    so the demo still runs (just with wall artifacts).

    Returns True iff PBC was successfully applied; False on fallback.
    """
    sim.gravity = 0.0
    sim.damping = damping
    sim.target_temp = target_temp
    sim.use_thermostat = True
    pbc_set = bool(sim._pbc_safe())
    sim.boundary_mode = 2 if pbc_set else 1  # 2=PERIODIC, 1=REFLECTING
    return pbc_set


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

def preset_demixing(scene, n_per_species: int = 600, target_temp: float = 0.7,
                    seed: Optional[int] = None) -> dict:
    """50:50 atomic mixture of Polar and Nonpolar — the workhorse demixing
    demo. With the R2 seeded ε_AB=0.25 override, like atoms attract (ε=1.0)
    and cross-pairs barely interact, so the mixture phase-separates into
    domains over a few seconds of simulation time.

    Initial state fills the periodic box uniformly: 2·n_per_species atoms
    placed on a `side²` grid with shuffled species assignment, where
    side = ceil(√(2·n_per_species)). Any grid slots beyond 2·n_per_species
    are dropped so the requested per-species count is exact. Under PBC
    domains form in the bulk without wall-induced nucleation, giving a
    cleaner separation phenomenology than the old REFLECTING setup.

    Args:
        scene: target Scene (cleared in place)
        n_per_species: atoms per species (total = 2 · n_per_species).
            Default 600 → 1200 atoms total, ρ* ≈ 0.48 at the default
            world_size — well into the LJ liquid range so domains are
            dense and visually clear.
        target_temp: thermostat setpoint. 0.7 keeps the system fluid above
            the 2D vapor-liquid coexistence ceiling (avoids splitting into
            vapor + liquid in addition to demixing by species).
        seed: RNG seed for reproducibility.

    Returns dict with keys:
        - 'n_polar', 'n_nonpolar': atoms placed of each species (each
          equals n_per_species)
        - 'world_size': the scene's world size (unchanged)
        - 'pbc_set': True if PERIODIC was applied; False on fallback
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)
    pbc_set = _seed_pbc_demo_state(sim, target_temp)

    polar_id = sketch.get_material_index("Polar")
    nonpolar_id = sketch.get_material_index("Nonpolar")
    polar_mat = sketch.get_material("Polar")
    nonpolar_mat = sketch.get_material("Nonpolar")

    total = 2 * n_per_species
    side = max(1, int(math.ceil(math.sqrt(total))))
    positions = _grid_positions_pbc_fill(side, sim.world_size)
    # Helper returns side² (>= total); slice to exact requested count so
    # the per-species split is clean.
    positions = positions[:total]
    # Half the shuffled fill grid → Polar, the other half → Nonpolar.
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
        'pbc_set': pbc_set,
    }


def _compute_micelle_layout(world_size: float, n_surfactant: int,
                            surfactants_per_micelle: int,
                            R_micelle: float, head_half_length: float):
    """Distribute n_surfactant surfactants across ceil(n / sm) micelles laid
    out on a uniform grid in [margin, L-margin]². Within each micelle,
    surfactants are placed on a ring of radius R_micelle around the micelle
    centre at evenly-spaced angles, with rotation = θ + π so the head
    (local -x end of the surfactant) points OUTWARD from the micelle
    centre.

    Returns (centers, sizes, placements) where:
      - centers: list of (cx, cy) for each micelle
      - sizes:   list of surfactant counts per micelle (sums to n_surfactant)
      - placements: list of (sx, sy, rotation) for every surfactant, ordered
        micelle-by-micelle
    """
    n_micelles = max(1, math.ceil(n_surfactant / max(1, surfactants_per_micelle)))
    # Even distribution: first `extra` micelles get one more surfactant
    base = n_surfactant // n_micelles
    extra = n_surfactant % n_micelles
    sizes = [base + (1 if i < extra else 0) for i in range(n_micelles)]

    # Grid of micelle centres. Margin = head ring radius + 1σ cushion so a
    # micelle's outer atoms stay strictly inside [0, L). Under PBC the
    # margin is a placement convenience, not a physics requirement, but it
    # keeps the initial layout symmetric and the cell-list build trivial.
    head_ring_r = R_micelle + head_half_length
    margin = head_ring_r + 1.0
    cols = max(1, int(math.ceil(math.sqrt(n_micelles))))
    rows = max(1, int(math.ceil(n_micelles / cols)))
    span_x = max(0.0, world_size - 2 * margin)
    span_y = max(0.0, world_size - 2 * margin)

    centers = []
    placements = []
    for i in range(n_micelles):
        r = i // cols
        c = i % cols
        cx = margin + (c + 0.5) * span_x / cols
        cy = margin + (r + 0.5) * span_y / rows
        centers.append((cx, cy))

        size = sizes[i]
        for k in range(size):
            theta = 2.0 * math.pi * k / size if size > 0 else 0.0
            sx = cx + R_micelle * math.cos(theta)
            sy = cy + R_micelle * math.sin(theta)
            # head_outward: local -x end is the head; want it to point AWAY
            # from the micelle centre. The placement-time rotation that
            # maps local +x → (-cos θ, -sin θ) is θ + π.
            rot = theta + math.pi
            placements.append((sx, sy, rot))

    return centers, sizes, placements


def _surfactant_atom_positions(placements, head_half_length: float,
                               r_eq: float, n_atoms_per_molecule: int):
    """Compute every atom's world position for every planned surfactant
    placement. Used to exclude solvent grid cells that would otherwise
    overlap surfactant atoms at t=0.

    Local frame: 5 atoms along the +x axis at evenly-spaced positions
    centred on the molecule centre. Atom 0 (head) at local x = -head_half_length;
    atom k at local x = -head_half_length + k * r_eq.
    """
    if not placements:
        return np.zeros((0, 2), dtype=np.float64)
    local_x = np.array(
        [-head_half_length + k * r_eq for k in range(n_atoms_per_molecule)],
        dtype=np.float64,
    )
    out = np.empty((len(placements) * n_atoms_per_molecule, 2), dtype=np.float64)
    for i, (cx, cy, rot) in enumerate(placements):
        cos_r = math.cos(rot)
        sin_r = math.sin(rot)
        # Rotate (local_x, 0) by `rot` and translate by (cx, cy).
        wx = cx + cos_r * local_x
        wy = cy + sin_r * local_x
        out[i * n_atoms_per_molecule:(i + 1) * n_atoms_per_molecule, 0] = wx
        out[i * n_atoms_per_molecule:(i + 1) * n_atoms_per_molecule, 1] = wy
    return out


def preset_micelles(scene, n_solvent: int = 1000, n_surfactant: int = 24,
                    surfactants_per_micelle: int = 6,
                    target_temp: float = 0.6,
                    seed: Optional[int] = None) -> dict:
    """Polar solvent + Surfactant amphiphiles in pre-organised proto-micelles
    under periodic boundaries.

    Equilibrium self-assembly from a random initial state is too slow to
    show on demo timescales (random-placement → micelle takes ~10⁶ substeps
    at our concentration because surfactant diffusion across the box is the
    rate-limiting step). Instead, this preset places surfactants in
    ceil(n_surfactant / surfactants_per_micelle) pre-arranged radial
    clusters, each with surfactants pointing tails-inward and heads-outward.
    The simulation then tightens these proto-micelles into proper micellar
    structure under the seeded R2 hydrophobic mismatch (ε_PN=0.25) within a
    few hundred substeps; the user watches them stabilise, breathe, and
    occasionally exchange surfactants with the bulk.

    The status text accordingly describes "pre-arranged micelles stabilised
    by the hydrophobic effect" — not equilibrium self-assembly. Empirical
    diagnostic showed that the default-parameter version of the previous
    preset (random placement, n_surfactant=20) produced no aggregation
    at all on demo timescales.

    Initial state:
      - n_solvent target solvent atoms on a PBC fill grid, with grid cells
        within 1σ of any planned surfactant atom skipped so the LJ kernel
        doesn't see overlapping atoms at t=0 (which would otherwise blast
        the micelle structure apart in the first substep). Actual placed
        count is `info['n_solvent']`, ≤ requested.
      - Surfactants in `n_micelles` radial clusters of `surfactants_per_micelle`
        each. Within a cluster: K surfactants on a ring of radius R_micelle
        around the cluster centre at evenly-spaced angles, rotated so each
        surfactant's head points outward.

    Geometry (R_micelle=3.3, head_half_length=2.0 for the 5-atom Surfactant
    template): tail-tip ring at radius 1.3σ (tangential spacing 1.3σ at K=6,
    clear of LJ r_min ≈ 1.122σ); head ring at radius 5.3σ; inter-cluster
    centre distance L/cols ≥ 12σ at default settings.

    Args:
        scene: target Scene (cleared in place)
        n_solvent: solvent (Polar) target count. Default 1000 → after
            exclusion-zone filtering, ~970 placed at the default geometry.
        n_surfactant: surfactant molecule count. Default 24 = 4 clusters
            of 6 = clean 2×2 grid of micelles at the default world_size.
        surfactants_per_micelle: cluster size. Default 6 — gives a head
            ring of 6 surfactants spaced ~5.5σ apart around a tight tail
            interior. Smaller (K=4) gives looser micelles; larger (K=8)
            tightens the tail-tip ring uncomfortably for R_micelle=3.3.
        target_temp: thermostat setpoint. 0.6 — warm enough for surfactant
            mobility within the micelle, cool enough for the hydrophobic
            attraction to dominate dissociation. The micelle structure
            survives at this temperature; lower T tightens it further.
        seed: RNG seed (controls Maxwell-Boltzmann velocity draws only;
            layout is deterministic).

    Returns dict with keys:
        - 'n_solvent', 'n_surfactant_molecules', 'n_surfactant_atoms'
        - 'n_micelles', 'cluster_sizes', 'micelle_centers'
        - 'world_size', 'pbc_set'
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)
    pbc_set = _seed_pbc_demo_state(sim, target_temp)

    polar_id = sketch.get_material_index("Polar")
    polar_mat = sketch.get_material("Polar")
    L = float(sim.world_size)

    surfactant_tpl = sketch.molecules.get('Surfactant')

    if surfactant_tpl is None or n_surfactant <= 0:
        # No surfactants — fall back to a pure-solvent fill (preserves
        # back-compat for tests that pass n_surfactant=0 and for systems
        # whose Sketch was loaded without the Surfactant template).
        side = max(1, int(math.ceil(math.sqrt(max(1, n_solvent)))))
        positions = _grid_positions_pbc_fill(side, L)
        positions = positions[:n_solvent]
        for (x, y) in positions:
            vx, vy = _maxwell_boltzmann_velocity(target_temp, polar_mat.mass)
            sim._add_particle(
                float(x), float(y), vx=vx, vy=vy, is_static=0,
                sigma=polar_mat.sigma, epsilon=polar_mat.epsilon,
                mass=polar_mat.mass, color=polar_mat.color,
                material_id=polar_id,
            )
        sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())
        return {
            'n_solvent': sim.count,
            'n_surfactant_molecules': 0,
            'n_surfactant_atoms': 0,
            'n_micelles': 0,
            'cluster_sizes': [],
            'micelle_centers': [],
            'world_size': L,
            'pbc_set': pbc_set,
        }

    # ----- Plan the micelle layout (deterministic, no RNG) -----
    R_MICELLE = 3.3            # surfactant-centre ring radius
    HEAD_HALF_LENGTH = 2.0     # head atom is at local x = -2 in make_surfactant()
    R_EQ = 1.0                 # bond r_eq in make_surfactant()
    N_ATOMS_PER_MOLECULE = 5   # 1 head + 4 tails
    EXCLUSION_R2 = 1.0         # σ² — skip solvent grid cells with any
                               # planned surfactant atom within 1σ

    centers, sizes, placements = _compute_micelle_layout(
        world_size=L,
        n_surfactant=n_surfactant,
        surfactants_per_micelle=surfactants_per_micelle,
        R_micelle=R_MICELLE,
        head_half_length=HEAD_HALF_LENGTH,
    )
    surfactant_atoms = _surfactant_atom_positions(
        placements, head_half_length=HEAD_HALF_LENGTH, r_eq=R_EQ,
        n_atoms_per_molecule=N_ATOMS_PER_MOLECULE,
    )
    # Wrap surfactant-atom positions into [0, L) for PBC-correct exclusion
    # distance computation. Solvent grid positions are already in [0, L).
    sa_x = surfactant_atoms[:, 0] % L
    sa_y = surfactant_atoms[:, 1] % L

    # ----- Solvent fill with exclusion-zone filter -----
    side = max(1, int(math.ceil(math.sqrt(max(1, n_solvent)))))
    solvent_positions = _grid_positions_pbc_fill(side, L)
    placed_solvent = 0
    for (x, y) in solvent_positions:
        if placed_solvent >= n_solvent:
            break
        # PBC minimum-image distance from this grid cell to every surfactant atom.
        dx = np.abs(sa_x - float(x))
        dy = np.abs(sa_y - float(y))
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        d2_min = float((dx * dx + dy * dy).min())
        if d2_min < EXCLUSION_R2:
            continue
        vx, vy = _maxwell_boltzmann_velocity(target_temp, polar_mat.mass)
        sim._add_particle(
            float(x), float(y), vx=vx, vy=vy, is_static=0,
            sigma=polar_mat.sigma, epsilon=polar_mat.epsilon,
            mass=polar_mat.mass, color=polar_mat.color,
            material_id=polar_id,
        )
        placed_solvent += 1

    # ----- Place the pre-organised surfactants -----
    placed_mols = 0
    for (sx, sy, rot) in placements:
        cmd = AddMoleculeCommand(
            scene, surfactant_tpl, world_pos=(sx, sy), rotation=rot
        )
        if cmd.execute():
            placed_mols += 1

    n_surfactant_atoms = sim.count - placed_solvent
    sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())

    return {
        'n_solvent': placed_solvent,
        'n_surfactant_molecules': placed_mols,
        'n_surfactant_atoms': n_surfactant_atoms,
        'n_micelles': len(centers),
        'cluster_sizes': sizes,
        'micelle_centers': centers,
        'world_size': L,
        'pbc_set': pbc_set,
    }


def preset_crystal_anneal(scene, n_atoms: int = 1600,
                          target_temp: float = 1.5,
                          material: str = "Polar",
                          seed: Optional[int] = None) -> dict:
    """Single-species LJ fluid at high T under periodic boundaries —
    designed to be cooled. As the user dials target_temp down (manually
    or via a future scheduler), the fluid passes through the LJ phase
    diagram: gas → liquid → 2D hexagonal crystal. Default starts at
    T=1.5 (well into the supercritical regime); cooling to T~0.3 over
    several seconds produces a visible 2D HCP-like solid filling the
    periodic box.

    Initial state: side² PBC fill grid (side = ceil(√n_atoms)), sliced
    to exactly n_atoms positions. Default n_atoms=1600 (40²=1600 fits
    exactly, no slicing loss) → ρ* ≈ 0.64 at default world_size, dense
    enough that cooling traces a clean fluid → solid line without
    passing through a vapor-liquid coexistence gap.

    Args:
        scene: target Scene (cleared in place)
        n_atoms: total atom count. Default 1600 (= 40²) ρ* ≈ 0.64.
        target_temp: starting thermostat setpoint (high; expected to be
            lowered by the user)
        material: which species to use. Default "Polar"; "Heavy" gives a
            denser fluid that crystallises at higher T due to deeper LJ
            wells.
        seed: RNG seed

    Returns dict with 'n_atoms', 'world_size', 'material', 'pbc_set'.
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)
    pbc_set = _seed_pbc_demo_state(sim, target_temp)

    mat_id = sketch.get_material_index(material)
    mat = sketch.get_material(material)

    side = max(1, int(math.ceil(math.sqrt(max(1, n_atoms)))))
    positions = _grid_positions_pbc_fill(side, sim.world_size)
    positions = positions[:n_atoms]
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
        'pbc_set': pbc_set,
    }


def preset_pbc_liquid(scene, density: float = 0.7, target_temp: float = 0.7,
                      material: str = "Polar", seed: Optional[int] = None,
                      jitter: float = 0.05) -> dict:
    """Uniform LJ fluid filling the entire periodic simulation box at the
    requested reduced density ρ* = N σ² / L². No walls, no clusters, no
    gradient — every column / row of the box is initialised identically up
    to the seed-controlled jitter, so the run is purely about the bulk
    physics that emerges at this density.

    The number of atoms is chosen as side² where side = round(L √ρ / σ),
    giving an exact square grid that tiles [0, L)² under PBC. The achieved
    density is reported back in the return dict and may differ from the
    request by a few percent because side must be integer.

    Args:
        scene: target Scene (cleared in place)
        density: reduced number density ρ* = N σ² / L². Physically
            interesting ranges (2D LJ):
              0.05–0.20  vapor / dilute gas
              0.40–0.70  liquid (typical LJ range, well above the 2D critical
                         density ~0.34)
              0.80–0.95  dense liquid near the freezing line; cools toward
                         a 2D hexagonal solid at low T
              ≥ 1.00     overcompressed — needs higher T to stay fluid, else
                         crystallises immediately on a near-HCP lattice
                         (2D close-packing max ρ ≈ 1.155)
        target_temp: thermostat setpoint. 0.7 is a comfortable LJ liquid
            temperature for ρ* in [0.4, 0.85]. Bump to ~1.2 for ρ* ≥ 1.0
            to keep the system fluid rather than freezing on contact.
        material: which species to use. Default "Polar" gives clean reduced
            LJ units (σ=ε=mass=1).
        seed: RNG seed for reproducibility (controls both grid jitter and
            Maxwell-Boltzmann velocity draws).
        jitter: fractional displacement applied to each grid position, in
            units of grid pitch. 0.05 = 5% — breaks lattice degeneracy
            without risking initial overlaps.

    Returns dict with keys:
        - 'n_atoms': actual atom count placed (= side²)
        - 'density_requested': the ρ* argument
        - 'density_achieved': the realised ρ* = n_atoms σ² / L²
        - 'side': grid side length (side² == n_atoms)
        - 'spacing': grid pitch (= L / side)
        - 'world_size': scene's world size (unchanged)
        - 'material': material name
        - 'pbc_set': True if PERIODIC was applied; False if the world is
            too small for a PBC cell list (caller asked but `_pbc_safe`
            refused — falls back to REFLECTING and reports it)
    """
    _seed_rngs(seed)
    sketch = scene.sketch
    sim = scene.simulation

    _clear_dynamic_and_compiled(scene)
    pbc_set = _seed_pbc_demo_state(sim, target_temp)

    mat_id = sketch.get_material_index(material)
    mat = sketch.get_material(material)

    L = float(sim.world_size)
    sigma = float(mat.sigma)
    # N from ρ*: ρ = N σ² / L²  →  N = ρ (L/σ)². Round to the nearest
    # perfect square so the grid tiles [0, L)² uniformly under PBC.
    target_n = max(1, int(round(density * (L / sigma) ** 2)))
    side = max(1, int(round(math.sqrt(target_n))))
    n_atoms = side * side
    achieved_density = n_atoms * sigma * sigma / (L * L)
    spacing = L / side

    positions = _grid_positions_pbc_fill(side, L, jitter=jitter)
    for (x, y) in positions:
        vx, vy = _maxwell_boltzmann_velocity(target_temp, mat.mass)
        sim._add_particle(
            float(x), float(y), vx=vx, vy=vy, is_static=0,
            sigma=mat.sigma, epsilon=mat.epsilon, mass=mat.mass,
            color=mat.color, material_id=mat_id,
        )

    sim.set_eps_ij_matrix(sketch.build_eps_ij_matrix())

    return {
        'n_atoms': n_atoms,
        'density_requested': float(density),
        'density_achieved': float(achieved_density),
        'side': side,
        'spacing': float(spacing),
        'world_size': L,
        'material': material,
        'pbc_set': pbc_set,
    }
