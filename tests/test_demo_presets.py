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
    preset_pbc_liquid,
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
        assert sim.boundary_mode == 2  # PERIODIC — demixing under PBC has
                                       # no wall-induced nucleation bias

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
        # PBC fill places atoms strictly in [0, L) — no edge atoms.
        assert np.all(sim.pos_x[:N] >= 0)
        assert np.all(sim.pos_x[:N] < sim.world_size)
        assert np.all(sim.pos_y[:N] >= 0)
        assert np.all(sim.pos_y[:N] < sim.world_size)

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
        assert sim.gravity == 0.0
        assert sim.boundary_mode == 2  # PERIODIC


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

    def test_periodic_boundary_set(self):
        """Crystal annealing under PBC produces a cleaner solid phase —
        no wall-induced nucleation biasing the crystal orientation."""
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, seed=0)
        assert scene.simulation.boundary_mode == 2

    def test_no_gravity(self):
        scene = Scene(skip_warmup=True)
        preset_crystal_anneal(scene, seed=0)
        assert scene.simulation.gravity == 0.0


# =============================================================================
# preset_pbc_liquid
# =============================================================================

class TestPresetPbcLiquid:
    def test_density_close_to_request_at_typical_target(self):
        """At default L=50 the side rounding error is tiny: ρ*=0.5 →
        side=round(50 * √0.5) = round(35.36) = 35 → n=1225 → achieved
        ρ* = 1225/2500 = 0.49. Within 2% of request."""
        scene = Scene(skip_warmup=True)
        info = preset_pbc_liquid(scene, density=0.5, seed=0)
        assert info['n_atoms'] == 35 * 35
        assert abs(info['density_achieved'] - 0.5) < 0.02
        assert info['density_requested'] == pytest.approx(0.5)

    def test_count_matches_side_squared(self):
        scene = Scene(skip_warmup=True)
        info = preset_pbc_liquid(scene, density=0.7, seed=0)
        assert info['n_atoms'] == info['side'] ** 2
        assert scene.simulation.count == info['n_atoms']

    def test_atom_count_monotonic_in_density(self):
        """Higher ρ* → more atoms (assuming same world_size). Sanity check
        that nothing in the rounding chain inverts the relationship."""
        densities = [0.2, 0.5, 0.8, 1.0]
        counts = []
        for d in densities:
            scene = Scene(skip_warmup=True)
            info = preset_pbc_liquid(scene, density=d, seed=0)
            counts.append(info['n_atoms'])
        assert counts == sorted(counts)
        # And distinct (no plateau): 0.2 → 100, 0.5 → 1225, 0.8 → 2025, 1.0 → 2500.
        assert len(set(counts)) == len(counts)

    def test_pbc_mode_set(self):
        """The whole point of these presets — boundary_mode should be
        PERIODIC (==2) after the preset runs."""
        scene = Scene(skip_warmup=True)
        info = preset_pbc_liquid(scene, density=0.5, seed=0)
        assert info['pbc_set'] is True
        assert scene.simulation.boundary_mode == 2  # BOUNDARY_PERIODIC

    def test_physics_params_set(self):
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.5, target_temp=0.6, seed=0)
        sim = scene.simulation
        assert sim.target_temp == pytest.approx(0.6)
        assert sim.use_thermostat is True
        assert sim.gravity == 0.0

    def test_all_atoms_share_material(self):
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.4, material="Polar", seed=0)
        sim = scene.simulation
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        assert np.all(sim.atom_material_id[:sim.count] == polar_id)

    def test_atoms_within_world_bounds(self):
        """Under PBC atoms can wrap, but the initial placement must be in
        [0, L)² before any integration runs."""
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.7, seed=0)
        sim = scene.simulation
        N = sim.count
        assert np.all(sim.pos_x[:N] >= 0)
        assert np.all(sim.pos_x[:N] < sim.world_size)
        assert np.all(sim.pos_y[:N] >= 0)
        assert np.all(sim.pos_y[:N] < sim.world_size)

    def test_grid_spans_full_box(self):
        """Verify the placement actually fills the box (not centred with
        margins like _grid_positions does). Check the min/max coordinates
        span most of [0, L).

        Pitch = L/side; with jitter ±0.05*pitch the atom range is
        roughly [0.5*pitch - 0.05*pitch, L - 0.5*pitch + 0.05*pitch].
        For ρ*=0.5, L=50, side=35: pitch ≈ 1.43, so range ≈ [0.65, 49.35].
        """
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.5, seed=0)
        sim = scene.simulation
        N = sim.count
        L = sim.world_size
        # Min is well inside the lower half; max in the upper half.
        # Used a permissive margin (10% of L on each end) so the assertion
        # is robust to jitter and to the side-rounding spacing.
        assert np.min(sim.pos_x[:N]) < 0.1 * L
        assert np.max(sim.pos_x[:N]) > 0.9 * L
        assert np.min(sim.pos_y[:N]) < 0.1 * L
        assert np.max(sim.pos_y[:N]) > 0.9 * L

    def test_atoms_have_nonzero_initial_velocity(self):
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.4, target_temp=0.7, seed=0)
        sim = scene.simulation
        speeds = np.sqrt(sim.vel_x[:sim.count]**2 + sim.vel_y[:sim.count]**2)
        assert np.mean(speeds) > 0.0
        assert speeds.max() < 20.0  # No runaway MB samples

    def test_deterministic_with_seed(self):
        scene_a = Scene(skip_warmup=True)
        scene_b = Scene(skip_warmup=True)
        preset_pbc_liquid(scene_a, density=0.4, seed=42)
        preset_pbc_liquid(scene_b, density=0.4, seed=42)
        N = scene_a.simulation.count
        assert np.allclose(scene_a.simulation.pos_x[:N],
                           scene_b.simulation.pos_x[:N])
        assert np.allclose(scene_a.simulation.pos_y[:N],
                           scene_b.simulation.pos_y[:N])

    def test_clears_previous_state(self):
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.3, seed=0)
        first_count = scene.simulation.count
        preset_pbc_liquid(scene, density=0.3, seed=1)
        # Same density on a fresh seed → same atom count (deterministic in N)
        assert scene.simulation.count == first_count

    def test_runs_stably_for_short_integration(self):
        """End-to-end smoke: run a moderate number of substeps and confirm
        no NaNs, no escapes outside [0, L) (atoms stay wrapped), and total
        speed remains finite. Catches kernel-integration regressions
        specific to the PBC + dense fluid combo."""
        scene = Scene(skip_warmup=True)
        preset_pbc_liquid(scene, density=0.5, target_temp=0.7, seed=0)
        sim = scene.simulation
        for _ in range(5):
            sim.step(steps_to_run=200)
        N = sim.count
        assert not np.any(np.isnan(sim.pos_x[:N]))
        assert not np.any(np.isnan(sim.pos_y[:N]))
        # PBC keeps atoms in [0, L) after each step
        assert np.all(sim.pos_x[:N] >= 0)
        assert np.all(sim.pos_x[:N] < sim.world_size)
        assert np.all(sim.pos_y[:N] >= 0)
        assert np.all(sim.pos_y[:N] < sim.world_size)
        speeds = np.sqrt(sim.vel_x[:N]**2 + sim.vel_y[:N]**2)
        # Thermostat keeps the mean speed in a sensible band
        assert speeds.max() < 50.0


# =============================================================================
# Phenomenology: micelles preset maintains head-out / tail-in structure
# =============================================================================

class TestMicellesPhenomenology:
    """The micelles preset uses a PRE-ORGANIZED initial state (random-IC
    self-assembly takes ~10⁶ substeps at our concentration — confirmed
    empirically). These tests verify that the pre-arranged proto-micelles
    stay stable under the seeded ε_PN=0.25 hydrophobic mismatch — they don't
    dissolve, and the head-out / tail-in structure is preserved or tightens
    over a several-thousand-substep relaxation.

    The metric used is the cluster-size distribution of tail atoms under
    a 1.5σ adjacency threshold (excluding same-molecule pairs): if K
    pre-organized surfactants per micelle hold together, the tail-tip
    ring shows up as a cluster of size K, and there should be `n_micelles`
    such clusters.
    """

    def _tail_cluster_sizes(self, sim, n_solvent: int, n_surf_mols: int,
                             threshold: float = 1.5):
        """Return sorted-descending list of connected-component sizes
        among different-molecule tail atoms within `threshold` σ (PBC).
        Each surfactant is laid out as 1 head + 4 tails."""
        N = sim.count
        L = float(sim.world_size)
        px = np.asarray(sim.pos_x[:N], dtype=np.float64)
        py = np.asarray(sim.pos_y[:N], dtype=np.float64)
        s0 = n_solvent
        tail_idx = np.array(
            [s0 + 5 * i + j for i in range(n_surf_mols) for j in (1, 2, 3, 4)],
            dtype=np.int64,
        )
        tail_mol = np.array(
            [i for i in range(n_surf_mols) for _ in range(4)],
            dtype=np.int64,
        )
        ax = px[tail_idx][:, None]; ay = py[tail_idx][:, None]
        bx = px[tail_idx][None, :]; by = py[tail_idx][None, :]
        dx = np.minimum(np.abs(ax - bx), L - np.abs(ax - bx))
        dy = np.minimum(np.abs(ay - by), L - np.abs(ay - by))
        dist = np.sqrt(dx * dx + dy * dy)
        same_mol = tail_mol[:, None] == tail_mol[None, :]
        adj = (dist < threshold) & (dist > 0.0) & (~same_mol)

        visited = np.zeros(len(tail_idx), dtype=bool)
        sizes = []
        for start in range(len(tail_idx)):
            if visited[start]:
                continue
            comp = 0
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp += 1
                for nb in np.where(adj[node])[0]:
                    if not visited[nb]:
                        stack.append(int(nb))
            sizes.append(comp)
        return sorted(sizes, reverse=True)

    def _head_solvent_vs_head_tail(self, sim, n_solvent, n_surf_mols):
        """Mean nearest-neighbour distance for head atoms: to solvent vs to
        other-molecule tail atoms. Returns (hs_nn, ht_nn). Micelle structure
        (heads out) implies hs_nn < ht_nn by a wide margin."""
        N = sim.count
        L = float(sim.world_size)
        px = np.asarray(sim.pos_x[:N], dtype=np.float64)
        py = np.asarray(sim.pos_y[:N], dtype=np.float64)
        s0 = n_solvent
        head_idx = np.array([s0 + 5 * i for i in range(n_surf_mols)], dtype=np.int64)
        tail_idx = np.array(
            [s0 + 5 * i + j for i in range(n_surf_mols) for j in (1, 2, 3, 4)],
            dtype=np.int64,
        )
        tail_mol = np.array(
            [i for i in range(n_surf_mols) for _ in range(4)],
            dtype=np.int64,
        )
        solvent_idx = np.arange(0, n_solvent, dtype=np.int64)

        def _pbc_dist(a_idx, b_idx):
            ax = px[a_idx][:, None]; ay = py[a_idx][:, None]
            bx = px[b_idx][None, :]; by = py[b_idx][None, :]
            dx = np.minimum(np.abs(ax - bx), L - np.abs(ax - bx))
            dy = np.minimum(np.abs(ay - by), L - np.abs(ay - by))
            return np.sqrt(dx * dx + dy * dy)

        hs = _pbc_dist(head_idx, solvent_idx)
        ht = _pbc_dist(head_idx, tail_idx)
        for i in range(n_surf_mols):
            ht[i, tail_mol == i] = np.inf
        return float(np.min(hs, axis=1).mean()), float(np.min(ht, axis=1).mean())

    def test_initial_state_has_n_micelles_distinct_clusters(self):
        """Right after the preset runs, the pre-organized layout should give
        exactly `n_micelles` clusters whose sizes match `cluster_sizes`."""
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, seed=42)
        assert info['n_micelles'] == 4
        assert info['cluster_sizes'] == [6, 6, 6, 6]
        sizes = self._tail_cluster_sizes(
            scene.simulation, info['n_solvent'], info['n_surfactant_molecules']
        )
        # Top 4 clusters should each be size 6 — the tail-tip ring of each
        # micelle. The remaining tail atoms are singletons (outer tails
        # sitting outside the 1.5σ adjacency threshold from each other).
        assert sizes[:4] == [6, 6, 6, 6], (
            f"Expected 4 size-6 clusters at t=0; got {sizes[:8]}"
        )

    def test_micelles_remain_stable_after_relaxation(self):
        """After integration, the 4 pre-organized micelles should still be
        4 distinct clusters — they don't fuse into a single blob, and they
        don't dissociate into singletons. The interior may consolidate
        (cluster sizes grow as outer-tail-atoms come within 1.5σ of other
        molecules' tails), but the head count of distinct clusters of
        size ≥ 4 stays at 4."""
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, seed=42)
        sim = scene.simulation
        for _ in range(20):
            sim.step(steps_to_run=500)  # 10k substeps total
        sizes = self._tail_cluster_sizes(
            sim, info['n_solvent'], info['n_surfactant_molecules']
        )
        big = [s for s in sizes if s >= 4]
        assert len(big) == 4, (
            f"Expected 4 stable micelles (clusters of size >= 4); got {sizes[:12]}"
        )
        # And no single dominant blob: largest cluster should be << total
        # tail count (24 surfactants × 4 = 96 tails). If a 50+ cluster
        # appears, the micelles have fused into a phase-separated blob.
        assert sizes[0] < 50, (
            f"Largest cluster too big — micelles have merged into a blob: {sizes[:6]}"
        )

    def test_micelles_head_out_structure_preserved(self):
        """The defining feature of a micelle: heads point outward (toward
        solvent), tails point inward (toward other tails). Metric: a head's
        nearest-neighbour distance to solvent should be much smaller than
        its nearest-neighbour distance to other-molecule tail atoms."""
        scene = Scene(skip_warmup=True)
        info = preset_micelles(scene, seed=42)
        sim = scene.simulation
        for _ in range(20):
            sim.step(steps_to_run=500)
        hs, ht = self._head_solvent_vs_head_tail(
            sim, info['n_solvent'], info['n_surfactant_molecules']
        )
        assert hs < ht, (
            f"Head-out structure lost: head-solvent NN={hs:.2f}, "
            f"head-tail NN={ht:.2f}"
        )
        # Sanity bands: heads in solvent are at LJ equilibrium ~1-1.5σ;
        # heads far from other-molecule tails should be ~3σ or more if
        # surfactants are in tight clusters (not dispersed).
        assert hs < 1.8, f"Heads too far from solvent (hs-NN={hs:.2f}σ)"
        assert ht > 2.5, f"Heads too close to other molecules' tails (ht-NN={ht:.2f}σ)"


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
        # Under PBC the system lacks wall-induced nucleation, so domains
        # only form if the density is high enough for LJ attraction to
        # dominate thermal motion. n_per_species=300 (total 600, ρ*≈0.24)
        # puts the nearest-neighbour distance near 2σ — solidly in LJ's
        # well-of-attraction range. T=0.6 keeps diffusion fast enough that
        # 5000 substeps reach a visibly demixed state.
        scene = Scene(skip_warmup=True)
        preset_demixing(scene, n_per_species=300, target_temp=0.6, seed=42)
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
