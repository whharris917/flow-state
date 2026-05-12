"""Tests for the R3 demo presets.

Three layers:
1. Each preset sets up the simulation state correctly (atom counts,
   material_ids, physics params).
2. Determinism — same seed → same initial positions.
3. End-to-end phenomenology: with the preset's initial conditions and a
   plausible relaxation run, the simulation produces the expected emergent
   behaviour (demixing for the demixing preset).
"""

import math

import numpy as np
import pytest

from core.demo_presets import (
    preset_demixing, preset_micelles, preset_crystal_anneal,
)
from core.scene import Scene


# =============================================================================
# preset_demixing
# =============================================================================

class TestPresetDemixing:
    def test_atom_count_matches_request(self):
        scene = Scene(skip_warmup=True)
        info = preset_demixing(scene, n_per_species=40, seed=0)
        assert info['n_polar'] == 40
        assert info['n_nonpolar'] == 40
        assert scene.simulation.count == 80

    def test_material_ids_correct(self):
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=30, seed=0)
        sim = scene.simulation
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        n_polar = int(np.sum(sim.atom_material_id[:sim.count] == polar_id))
        n_nonpolar = int(np.sum(sim.atom_material_id[:sim.count] == nonpolar_id))
        assert n_polar == 30
        assert n_nonpolar == 30

    def test_physics_params_set(self):
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, target_temp=0.7, seed=0)
        sim = scene.simulation
        assert sim.target_temp == pytest.approx(0.7)
        assert sim.use_thermostat is True
        assert sim.gravity == 0.0
        assert sim.boundary_mode == 1  # REFLECTING

    def test_atoms_have_nonzero_initial_velocity(self):
        """Maxwell-Boltzmann sampling gives the system thermal kinetic
        energy from t=0 so the thermostat has something to work with."""
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=20, target_temp=0.5, seed=0)
        sim = scene.simulation
        speeds = np.sqrt(sim.vel_x[:sim.count]**2 + sim.vel_y[:sim.count]**2)
        assert np.mean(speeds) > 0.0
        # Sanity: not all zero, not all huge
        assert speeds.max() < 10.0

    def test_atoms_within_world_bounds(self):
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=50, seed=0)
        sim = scene.simulation
        N = sim.count
        assert np.all(sim.pos_x[:N] >= 0)
        assert np.all(sim.pos_x[:N] <= sim.world_size)
        assert np.all(sim.pos_y[:N] >= 0)
        assert np.all(sim.pos_y[:N] <= sim.world_size)

    def test_deterministic_with_seed(self):
        scene_a = Scene(skip_warmup=True)
        scene_b = Scene(skip_warmup=True)
        preset_demixing(scene_a, n_per_species=20, seed=42)
        preset_demixing(scene_b, n_per_species=20, seed=42)
        N = scene_a.simulation.count
        # Same positions to within float-roundoff
        assert np.allclose(scene_a.simulation.pos_x[:N],
                           scene_b.simulation.pos_x[:N])
        assert np.allclose(scene_a.simulation.pos_y[:N],
                           scene_b.simulation.pos_y[:N])

    def test_clears_previous_state(self):
        """A second call to preset_demixing replaces the prior state."""
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=10, seed=0)
        first_count = scene.simulation.count
        preset_demixing(scene, n_per_species=10, seed=1)
        assert scene.simulation.count == first_count

    def test_eps_matrix_carries_seeded_override(self):
        """After the preset runs, the kernel should see ε_Polar-Nonpolar = 0.25
        (the R2 seed). End-to-end smoke: R1 + R2 + R3 all integrate."""
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=10, seed=0)
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        m = scene.simulation.eps_ij_matrix
        assert m[polar_id, nonpolar_id] == pytest.approx(0.25)


# =============================================================================
# preset_micelles
# =============================================================================

class TestPresetMicelles:
    def test_solvent_atom_count(self):
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, n_solvent=60, n_surfactant=8, seed=0)
        assert info['n_solvent'] == 60

    def test_surfactant_molecules_placed(self):
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, n_solvent=60, n_surfactant=8, seed=0)
        assert info['n_surfactant_molecules'] == 8
        # 5 atoms per surfactant template (1 head + 4 tails)
        assert info['n_surfactant_atoms'] == 8 * 5

    def test_total_count_matches(self):
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, n_solvent=50, n_surfactant=6, seed=0)
        sim = scene.simulation
        assert sim.count == info['n_solvent'] + info['n_surfactant_atoms']

    def test_surfactant_atoms_carry_polar_and_nonpolar_ids(self):
        scene = Scene(skip_warmup=True)
        preset_micelles(scene, n_solvent=20, n_surfactant=4, seed=0)
        sim = scene.simulation
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        # Surfactant atoms: 4 × 5 = 20 atoms, of which 4 are Polar (heads)
        # and 16 are Nonpolar (tails). Plus 20 solvent (all Polar).
        # Total Polar = 20 + 4 = 24; total Nonpolar = 16.
        n_polar = int(np.sum(sim.atom_material_id[:sim.count] == polar_id))
        n_nonpolar = int(np.sum(sim.atom_material_id[:sim.count] == nonpolar_id))
        assert n_polar == 24
        assert n_nonpolar == 16

    def test_surfactant_molecule_creates_bonds_and_angles(self):
        """Each surfactant has 4 bonds + 3 angles (n_tail=4 → 5 atoms in chain)."""
        scene = Scene(skip_warmup=True)
        preset_micelles(scene, n_solvent=0, n_surfactant=3, seed=0)
        sim = scene.simulation
        assert sim.bond_count == 3 * 4   # 3 surfactants × 4 bonds each
        assert sim.angle_count == 3 * 3  # 3 surfactants × 3 angles each

    def test_physics_params_set(self):
        scene = Scene(skip_warmup=True)
        preset_micelles(scene, target_temp=0.4, seed=0)
        sim = scene.simulation
        assert sim.target_temp == pytest.approx(0.4)
        assert sim.use_thermostat is True
        assert sim.boundary_mode == 1


# =============================================================================
# preset_crystal_anneal
# =============================================================================

class TestPresetCrystalAnneal:
    def test_atom_count_matches_request(self):
        scene = Scene(skip_warmup=True)
        info = preset_crystal_anneal(scene, n_atoms=80, seed=0)
        # _grid_positions returns up to n; actual count == ceil(sqrt(n))^2 sliced to n
        assert info['n_atoms'] == 80
        assert scene.simulation.count == 80

    def test_all_atoms_share_material(self):
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, n_atoms=60, material="Polar", seed=0)
        sim = scene.simulation
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        assert np.all(sim.atom_material_id[:sim.count] == polar_id)

    def test_custom_material_heavy(self):
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, n_atoms=40, material="Heavy", seed=0)
        sim = scene.simulation
        sketch = scene.sketch
        heavy_id = sketch.get_material_index("Heavy")
        assert np.all(sim.atom_material_id[:sim.count] == heavy_id)
        # Sigma matches Heavy's value (1.2)
        assert np.allclose(sim.atom_sigma[:sim.count], 1.2)

    def test_starts_at_high_target_temp(self):
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, target_temp=1.8, seed=0)
        assert scene.simulation.target_temp == pytest.approx(1.8)

    def test_thermostat_enabled(self):
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, seed=0)
        assert scene.simulation.use_thermostat is True


# =============================================================================
# Phenomenology: demixing actually demixes
# =============================================================================

class TestDemixingPhenomenology:
    """End-to-end: the preset_demixing initial state should evolve toward
    phase separation under the LJ kernel with seeded ε_AB=0.25. Metric:
    after relaxation, an atom's mean distance to its 4 nearest LIKE
    neighbours should be smaller than the mean distance to its 4 nearest
    UNLIKE neighbours — i.e., like atoms cluster more tightly than they
    mingle with unlike ones.

    With the R1+R2 setup this should be a robust signal — Polar and
    Nonpolar share σ so the metric isn't confounded by size mismatch,
    and the like-pair ε is 4× the cross-pair so cohesion strongly
    favours like clustering.
    """

    def _mean_dist_to_k_nearest_like_unlike(self, sim, sketch, k_neighbors=4):
        """For each atom, find the k nearest LIKE atoms and the k nearest
        UNLIKE atoms; return (mean_like_dist, mean_unlike_dist). Smaller
        like-distance indicates like-clustering."""
        N = sim.count
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        px = np.asarray(sim.pos_x[:N], dtype=np.float64)
        py = np.asarray(sim.pos_y[:N], dtype=np.float64)
        mids = np.asarray(sim.atom_material_id[:N])
        # Pairwise distance matrix
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)

        like_dists = []
        unlike_dists = []
        for i in range(N):
            same = mids == mids[i]
            same[i] = False
            other = ~same & (mids != -1) & np.arange(N) != i
            same_dists = np.sort(dist[i, same])[:k_neighbors]
            other_dists = np.sort(dist[i, ~same & (np.arange(N) != i)])[:k_neighbors]
            if len(same_dists) > 0:
                like_dists.append(np.mean(same_dists))
            if len(other_dists) > 0:
                unlike_dists.append(np.mean(other_dists))
        return np.mean(like_dists), np.mean(unlike_dists)

    def test_demixing_produces_like_clustering(self):
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=60, target_temp=0.4, seed=42)
        sim = scene.simulation
        sketch = scene.sketch

        # Initial state: random grid placement → like and unlike distances
        # should be similar (no spatial bias yet).
        like0, unlike0 = self._mean_dist_to_k_nearest_like_unlike(sim, sketch)
        # Run a few seconds of simulation time so domains can form.
        for _ in range(25):
            sim.step(steps_to_run=200)

        like1, unlike1 = self._mean_dist_to_k_nearest_like_unlike(sim, sketch)

        # After relaxation, like-distance should be CLEARLY smaller than
        # unlike-distance — atoms have aggregated by species.
        assert like1 < unlike1, (
            f"Expected like-clustering after relaxation; "
            f"got like_dist={like1:.3f} unlike_dist={unlike1:.3f}"
        )
        # And the gap should have GROWN from the initial random-ish state.
        # Margin chosen permissively to absorb thermal fluctuations.
        gap_initial = unlike0 - like0
        gap_final = unlike1 - like1
        assert gap_final > gap_initial + 0.05, (
            f"Demixing didn't strengthen over time; "
            f"initial gap={gap_initial:.3f}, final gap={gap_final:.3f}"
        )
