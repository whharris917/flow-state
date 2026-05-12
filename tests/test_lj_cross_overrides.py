"""Tests for the LJ per-pair ε cross-override mechanism (R1).

Breaks Berthelot's geometric-mean rule on a per-material-pair basis. This is
the load-bearing engine addition that unlocks emergent demixing, micelle
formation, and wetting demos. Layered tests:

1. Sketch palette: store/retrieve/round-trip, order-independence, L-B fallback
2. Simulation arrays: atom_material_id storage, eps_ij_matrix shape/coerce
3. Kernel: cross-pair force matches the override (not L-B)
4. End-to-end: A-B species with strong ε_AB cohesion segregate vs B-only L-B
"""

import math

import numpy as np
import pytest

from core.scene import Scene
from engine.simulation import Simulation
from model.properties import Material
from model.sketch import Sketch


# =============================================================================
# Sketch API
# =============================================================================

class TestSketchLjCrossOverrides:
    def test_defaults_seeded_for_r2_species(self):
        """A fresh Sketch ships with the R2 emergent-phenomena cross-ε
        overrides seeded by _seed_default_lj_overrides. Pre-R2 saves
        restore without them (covered by TestSketchSerialization below)."""
        s = Sketch()
        # The four seeded R2 pairs (order-independent lookup)
        assert s.get_lj_epsilon("Polar", "Nonpolar") == pytest.approx(0.25)
        assert s.get_lj_epsilon("Polar", "LightGas") == pytest.approx(0.15)
        assert s.get_lj_epsilon("Heavy", "LightGas") == pytest.approx(0.20)
        assert s.get_lj_epsilon("Nonpolar", "Heavy") == pytest.approx(1.5)
        # Legacy pairs untouched — fall back to L-B
        # Water-Oil: √(1.0 · 0.8)
        assert s.get_lj_epsilon("Water", "Oil") == pytest.approx(math.sqrt(0.8), rel=1e-5)

    def test_get_lj_epsilon_falls_back_to_lorentz_berthelot(self):
        """No override → geometric-mean of the materials' own ε."""
        s = Sketch()
        # Defaults: Water ε=1.0, Oil ε=0.8.
        expected = math.sqrt(1.0 * 0.8)
        got = s.get_lj_epsilon("Water", "Oil")
        assert abs(got - expected) < 1e-6

    def test_set_override_changes_lookup(self):
        s = Sketch()
        s.set_lj_cross_override("Water", "Oil", 0.2)
        assert s.get_lj_epsilon("Water", "Oil") == pytest.approx(0.2)

    def test_lookup_is_order_independent(self):
        s = Sketch()
        s.set_lj_cross_override("Water", "Oil", 0.2)
        # frozenset key → A-B vs B-A return same value
        assert s.get_lj_epsilon("Oil", "Water") == pytest.approx(0.2)

    def test_remove_override_restores_lb(self):
        s = Sketch()
        s.set_lj_cross_override("Water", "Oil", 0.2)
        s.remove_lj_cross_override("Water", "Oil")
        expected = math.sqrt(1.0 * 0.8)
        assert s.get_lj_epsilon("Water", "Oil") == pytest.approx(expected)

    def test_remove_nonexistent_override_is_noop(self):
        s = Sketch()
        # No KeyError when there's nothing to remove for this pair
        before = dict(s.lj_cross_overrides)
        s.remove_lj_cross_override("Water", "Mercury")
        # Water-Mercury wasn't a seeded R2 override → no change to the dict
        assert s.lj_cross_overrides == before

    def test_unknown_material_falls_back_to_water_defaults(self):
        """get_material returns Water-as-default for unknown names; the L-B
        path therefore uses Water's ε for both. Important so the API never
        raises for unknown names (matches the existing get_material contract)."""
        s = Sketch()
        # "Nope" doesn't exist → resolved as Water; "Water" is Water.
        # L-B then is sqrt(1.0 * 1.0) = 1.0.
        assert s.get_lj_epsilon("Nope", "Water") == pytest.approx(1.0)


class TestSketchEpsMatrixBuild:
    def test_matrix_shape_matches_materials(self):
        s = Sketch()
        matrix = s.build_eps_ij_matrix()
        n = len(s.materials)
        assert matrix.shape == (n, n)
        assert matrix.dtype == np.float32

    def test_matrix_diagonal_is_pure_epsilon(self):
        """Diagonal entry (i, i) is geometric mean of the material with
        itself = the material's own ε (no override needed)."""
        s = Sketch()
        names = list(s.materials.keys())
        matrix = s.build_eps_ij_matrix()
        for i, name in enumerate(names):
            assert matrix[i, i] == pytest.approx(s.materials[name].epsilon, rel=1e-5)

    def test_matrix_is_symmetric_without_overrides(self):
        s = Sketch()
        matrix = s.build_eps_ij_matrix()
        # L-B is symmetric: √(ε_i·ε_j) == √(ε_j·ε_i)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                assert matrix[i, j] == pytest.approx(matrix[j, i], rel=1e-6)

    def test_matrix_reflects_override(self):
        s = Sketch()
        s.set_lj_cross_override("Water", "Oil", 0.15)
        names = list(s.materials.keys())
        i = names.index("Water")
        j = names.index("Oil")
        matrix = s.build_eps_ij_matrix()
        assert matrix[i, j] == pytest.approx(0.15)
        assert matrix[j, i] == pytest.approx(0.15)


class TestSketchSerialization:
    def test_lj_overrides_round_trip(self):
        s = Sketch()
        s.set_lj_cross_override("Water", "Oil", 0.18)
        s.set_lj_cross_override("Mercury", "Water", 1.4)
        data = s.to_dict()

        s2 = Sketch()
        s2.restore(data)
        # Both overrides restored, order-independent
        assert s2.get_lj_epsilon("Water", "Oil") == pytest.approx(0.18)
        assert s2.get_lj_epsilon("Mercury", "Water") == pytest.approx(1.4)
        # No spurious override introduced
        assert s2.get_lj_epsilon("Oil", "Mercury") == pytest.approx(
            math.sqrt(0.8 * 2.0), rel=1e-5
        )

    def test_pre_r1_save_loads_with_empty_overrides(self):
        """Saves authored before R1 have no 'lj_cross_overrides' key.
        restore() must not crash; defaults to empty (L-B everywhere)."""
        s = Sketch()
        data = s.to_dict()
        del data['lj_cross_overrides']
        s2 = Sketch()
        s2.restore(data)
        assert s2.lj_cross_overrides == {}


class TestSketchGetMaterialIndex:
    def test_returns_insertion_order_position(self):
        s = Sketch()
        names = list(s.materials.keys())
        for i, name in enumerate(names):
            assert s.get_material_index(name) == i

    def test_unknown_returns_minus_one(self):
        s = Sketch()
        assert s.get_material_index("Nonexistent") == -1


# =============================================================================
# Simulation arrays + matrix plumbing
# =============================================================================

class TestSimulationMaterialIdArray:
    def test_defaults_to_minus_one(self):
        sim = Simulation(skip_warmup=True)
        # Fresh array — every slot is -1
        assert np.all(sim.atom_material_id == -1)

    def test_add_particle_stores_material_id(self):
        sim = Simulation(skip_warmup=True)
        idx = sim._add_particle(10.0, 10.0, material_id=2)
        assert sim.atom_material_id[idx] == 2

    def test_add_particle_default_material_id_is_minus_one(self):
        """Bare-Sim test path without a material → -1 → kernel falls back to
        per-atom ε_sqrt. Preserves legacy behavior."""
        sim = Simulation(skip_warmup=True)
        idx = sim._add_particle(10.0, 10.0, sigma=1.0, epsilon=1.0)
        assert sim.atom_material_id[idx] == -1


class TestSimulationEpsMatrix:
    def test_default_shape_is_empty(self):
        sim = Simulation(skip_warmup=True)
        assert sim.eps_ij_matrix.shape == (0, 0)

    def test_set_eps_ij_matrix_coerces_to_float32(self):
        sim = Simulation(skip_warmup=True)
        sim.set_eps_ij_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
        assert sim.eps_ij_matrix.dtype == np.float32
        assert sim.eps_ij_matrix.shape == (2, 2)

    def test_set_eps_ij_matrix_none_resets_to_empty(self):
        sim = Simulation(skip_warmup=True)
        sim.set_eps_ij_matrix(np.eye(3))
        sim.set_eps_ij_matrix(None)
        assert sim.eps_ij_matrix.shape == (0, 0)

    def test_set_eps_ij_matrix_rejects_non_square(self):
        sim = Simulation(skip_warmup=True)
        with pytest.raises(ValueError):
            sim.set_eps_ij_matrix(np.zeros((2, 3)))


class TestSimulationMaterialIdLifecycle:
    def test_compact_arrays_remaps_material_id(self):
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, material_id=0)
        sim._add_particle(20.0, 20.0, material_id=1)
        sim._add_particle(30.0, 30.0, material_id=2)
        # Keep indices 0 and 2 (drop 1)
        sim.compact_arrays([0, 2])
        assert sim.atom_material_id[0] == 0
        assert sim.atom_material_id[1] == 2

    def test_snapshot_restore_preserves_material_id(self):
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, material_id=3)
        sim.snapshot()
        sim._add_particle(20.0, 20.0, material_id=5)
        sim.undo()
        assert sim.count == 1
        assert sim.atom_material_id[0] == 3

    def test_to_dict_restore_round_trip(self):
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, material_id=2)
        sim._add_particle(20.0, 20.0, material_id=4)
        data = sim.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(data)
        assert sim2.atom_material_id[0] == 2
        assert sim2.atom_material_id[1] == 4

    def test_pre_r1_restore_defaults_to_minus_one(self):
        """Saves predating R1 have no 'atom_material_id' key — restored
        atoms get -1 and the kernel uses the per-atom ε_sqrt path."""
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, material_id=2)
        data = sim.to_dict()
        del data['atom_material_id']
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(data)
        assert sim2.atom_material_id[0] == -1


# =============================================================================
# Kernel behaviour (force matches override, not L-B)
# =============================================================================

class TestKernelUsesOverrideEpsilon:
    """End-to-end through Simulation.step(): two atoms of different materials
    with a strongly attractive cross-ε override should experience LJ forces
    that pull them together more (or less) than the L-B baseline.

    We don't need a microbenchmark — just verify the displacement direction
    matches the override.
    """

    def _make_two_atom_sim(self, eps_ij_matrix):
        sim = Simulation(skip_warmup=True)
        # Disable gravity and damping so the only force is LJ pair.
        sim.gravity = 0.0
        sim.damping = 1.0
        sim.use_thermostat = False
        # Atoms placed slightly inside the LJ cutoff — feels the force from t=0
        sim._add_particle(10.0, 10.0, sigma=1.0, epsilon=1.0, material_id=0)
        sim._add_particle(11.5, 10.0, sigma=1.0, epsilon=1.0, material_id=1)
        sim.set_eps_ij_matrix(eps_ij_matrix)
        return sim

    def test_override_attractive_pair_pulls_atoms_closer_than_lb(self):
        """Build two sims with same setup but different ε_AB:
        - sim_lb: ε_AB = 1.0 (the L-B value of √(1·1))
        - sim_attractive: ε_AB = 4.0 (much stronger cohesion)
        The attractive sim's atoms should end up closer after the same number of substeps.
        """
        # L-B baseline
        eps_lb = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        sim_lb = self._make_two_atom_sim(eps_lb)
        sim_lb.step(steps_to_run=200)
        gap_lb = abs(sim_lb.pos_x[1] - sim_lb.pos_x[0])

        # Strong cross-pair attraction
        eps_attractive = np.array([[1.0, 4.0], [4.0, 1.0]], dtype=np.float32)
        sim_attr = self._make_two_atom_sim(eps_attractive)
        sim_attr.step(steps_to_run=200)
        gap_attr = abs(sim_attr.pos_x[1] - sim_attr.pos_x[0])

        # Stronger ε ⇒ deeper well ⇒ atoms held closer (around the same r_min
        # of 2^(1/6)σ ≈ 1.122 in 1D, but the SAME r_min — *displacement*
        # behaviour after impulse differs). Since both start at r=1.5 (>r_min)
        # they're attracted; the attractive sim attracts harder.
        assert gap_attr < gap_lb, f"attractive gap {gap_attr} should be < L-B gap {gap_lb}"

    def test_override_repulsive_pair_pushes_atoms_apart(self):
        """With ε_AB very small (~zero), the LJ well is shallow — the
        repulsive r^-12 core dominates within the cutoff, so the atoms
        separate rather than attract. Use a starting distance slightly less
        than r_min so the repulsive branch is active.
        """
        # Start very close (inside the well minimum) so r^-12 dominates.
        sim = Simulation(skip_warmup=True)
        sim.gravity = 0.0
        sim.damping = 1.0
        sim.use_thermostat = False
        sim._add_particle(10.0, 10.0, sigma=1.0, epsilon=1.0, material_id=0)
        sim._add_particle(10.9, 10.0, sigma=1.0, epsilon=1.0, material_id=1)
        eps = np.array([[1.0, 0.01], [0.01, 1.0]], dtype=np.float32)
        sim.set_eps_ij_matrix(eps)
        initial_gap = abs(sim.pos_x[1] - sim.pos_x[0])
        sim.step(steps_to_run=200)
        final_gap = abs(sim.pos_x[1] - sim.pos_x[0])
        # Even a weak ε still has a non-zero LJ force; atoms should move
        # *outward* from their sub-r_min start. Verify they moved away rather
        # than collapsed.
        assert final_gap > initial_gap

    def test_minus_one_material_id_uses_per_atom_eps_sqrt(self):
        """Atoms with material_id = -1 fall back to per-atom √(ε_i·ε_j),
        regardless of what's in the matrix. Verify by setting an exotic
        matrix value that would NOT match L-B if used: atoms still behave
        per their per-atom ε_sqrt = 1.0."""
        sim = Simulation(skip_warmup=True)
        sim.gravity = 0.0
        sim.damping = 1.0
        sim.use_thermostat = False
        sim._add_particle(10.0, 10.0, sigma=1.0, epsilon=1.0, material_id=-1)
        sim._add_particle(11.5, 10.0, sigma=1.0, epsilon=1.0, material_id=-1)
        # Matrix would say ε=100 (absurd) but it's bypassed by -1 material_id
        eps = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
        sim.set_eps_ij_matrix(eps)

        # Reference run with same particles, no matrix at all (shape (0,0))
        sim_ref = Simulation(skip_warmup=True)
        sim_ref.gravity = 0.0
        sim_ref.damping = 1.0
        sim_ref.use_thermostat = False
        sim_ref._add_particle(10.0, 10.0, sigma=1.0, epsilon=1.0, material_id=-1)
        sim_ref._add_particle(11.5, 10.0, sigma=1.0, epsilon=1.0, material_id=-1)

        sim.step(steps_to_run=100)
        sim_ref.step(steps_to_run=100)
        # Both runs use per-atom ε_sqrt → identical trajectories
        assert abs(sim.pos_x[0] - sim_ref.pos_x[0]) < 1e-4
        assert abs(sim.pos_x[1] - sim_ref.pos_x[1]) < 1e-4


# =============================================================================
# Scene integration — rebuild pushes matrix automatically
# =============================================================================

class TestSceneIntegration:
    def test_scene_init_pushes_matrix(self):
        """Constructing a Scene must populate sim.eps_ij_matrix with the
        L-B values for all currently-registered materials."""
        scene = Scene(skip_warmup=True)
        n = len(scene.sketch.materials)
        assert scene.simulation.eps_ij_matrix.shape == (n, n)

    def test_scene_rebuild_picks_up_new_override(self):
        scene = Scene(skip_warmup=True)
        scene.sketch.set_lj_cross_override("Water", "Oil", 0.05)
        scene.rebuild()
        names = list(scene.sketch.materials.keys())
        i = names.index("Water")
        j = names.index("Oil")
        assert scene.simulation.eps_ij_matrix[i, j] == pytest.approx(0.05)


# Note on phenomenology: a full demixing / micelle-formation end-to-end test
# belongs in R3 once we have a tuned palette (e.g. "Polar" + "Nonpolar" with
# matched σ but contrasting ε, surfactant molecules, etc). R1's scope is the
# mechanism — verified by TestKernelUsesOverrideEpsilon above. With the
# current Water/Oil defaults (ε=1.0 vs 0.8), tuning a robust phenomenological
# assertion within the existing parameter range requires more setup than R1
# warrants; the standalone-pair tests already prove the kernel reads ε from
# the matrix, which is the actual contract under test.
