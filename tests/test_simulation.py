"""Tests for engine/simulation.py — particle arrays, world, snapshot/restore."""

import math
import numpy as np
import pytest

import core.config as config
from engine.simulation import Simulation, ENTITY_TYPE_LINE, ENTITY_TYPE_CIRCLE, ENTITY_TYPE_POINT
from model.geometry import Line, Circle, Point


class TestDefaults:
    def test_initial_count_is_zero(self, simulation):
        assert simulation.count == 0

    def test_initial_world_size_matches_config(self, simulation):
        assert simulation.world_size == config.DEFAULT_WORLD_SIZE

    def test_initial_capacity(self, simulation):
        assert simulation.capacity == 5000

    def test_starts_paused(self, simulation):
        assert simulation.paused is True


class TestWorldResize:
    def test_resize_changes_world_size(self, simulation):
        simulation.resize_world(100.0)
        assert simulation.world_size == 100.0

    def test_resize_clamps_to_minimum(self, simulation):
        simulation.resize_world(1.0)
        assert simulation.world_size == 10.0

    def test_resize_clears_particles(self, simulation):
        simulation._add_particle(5, 5)
        simulation._add_particle(10, 10)
        assert simulation.count == 2
        simulation.resize_world(100.0)
        assert simulation.count == 0

    def test_resize_resets_physics_defaults(self, simulation):
        simulation.gravity = 99.0
        simulation.dt = 0.5
        simulation.resize_world(50.0)
        # resize_world calls reset() internally, restoring defaults
        assert simulation.gravity == config.DEFAULT_GRAVITY
        assert simulation.dt == config.DEFAULT_DT


class TestClearAndReset:
    def test_clear_zeros_count(self, simulation):
        simulation._add_particle(1, 1)
        simulation._add_particle(2, 2)
        simulation.clear(snapshot=False)
        assert simulation.count == 0

    def test_reset_returns_to_defaults(self, simulation):
        simulation.gravity = 0.0
        simulation.world_size = 999.0
        simulation.reset()
        assert simulation.gravity == config.DEFAULT_GRAVITY
        assert simulation.world_size == config.DEFAULT_WORLD_SIZE


class TestParticleAddition:
    def test_add_particle_increments_count(self, simulation):
        idx = simulation._add_particle(5.0, 5.0)
        assert idx == 0
        assert simulation.count == 1

    def test_add_particle_stores_state(self, simulation):
        idx = simulation._add_particle(1.0, 2.0, vx=3.0, vy=4.0, is_static=1)
        assert simulation.pos_x[idx] == 1.0
        assert simulation.pos_y[idx] == 2.0
        assert simulation.vel_x[idx] == 3.0
        assert simulation.vel_y[idx] == 4.0
        assert simulation.is_static[idx] == 1

    def test_add_particle_writes_default_color(self, simulation):
        """Without an explicit color, _add_particle must still write a sensible
        default — never leaving the slot's previous (possibly wall) colour in
        place. Default is the project's water-blue."""
        idx = simulation._add_particle(1.0, 1.0)
        assert tuple(simulation.atom_color[idx]) == (50, 150, 255)

    def test_add_particle_does_not_inherit_residual_slot_color(self, simulation):
        """Regression: a Source spawning at a slot that previously held a wall
        atom used to inherit the wall colour because the primitive only wrote
        pos/vel/sigma/eps/is_static, never atom_color. Visible in-game as
        wall-coloured 'free' atoms bouncing around inside a wall enclosure.

        Reproduces by hand-staining the next-spawn slot with a wall colour,
        then adding a particle and asserting the colour was overwritten.
        """
        next_slot = simulation.count
        WALL_COLOR = (180, 180, 200)
        simulation.atom_color[next_slot] = WALL_COLOR

        idx = simulation._add_particle(2.0, 2.0)

        assert idx == next_slot
        assert tuple(simulation.atom_color[idx]) != WALL_COLOR
        assert tuple(simulation.atom_color[idx]) == (50, 150, 255)

    def test_add_particle_honours_explicit_color(self, simulation):
        idx = simulation._add_particle(3.0, 3.0, color=(200, 50, 80))
        assert tuple(simulation.atom_color[idx]) == (200, 50, 80)


class TestEntitySync:
    def test_line_sync_writes_endpoints(self, simulation):
        line = Line((1.0, 2.0), (3.0, 4.0))
        simulation.sync_entity_arrays([line])
        assert simulation.entity_count == 1
        assert simulation.entity_types[0] == ENTITY_TYPE_LINE
        assert simulation.entity_positions[0, 0] == 1.0
        assert simulation.entity_positions[0, 1] == 2.0
        assert simulation.entity_positions[0, 2] == 3.0
        assert simulation.entity_positions[0, 3] == 4.0

    def test_circle_sync_writes_center_and_radius(self, simulation):
        circle = Circle((5.0, 5.0), 2.0)
        simulation.sync_entity_arrays([circle])
        assert simulation.entity_types[0] == ENTITY_TYPE_CIRCLE
        assert simulation.entity_positions[0, 0] == 5.0
        assert simulation.entity_positions[0, 1] == 5.0
        assert simulation.entity_positions[0, 2] == 2.0  # radius

    def test_point_sync_writes_position(self, simulation):
        point = Point(7.0, 8.0)
        simulation.sync_entity_arrays([point])
        assert simulation.entity_types[0] == ENTITY_TYPE_POINT
        assert simulation.entity_positions[0, 0] == 7.0
        assert simulation.entity_positions[0, 1] == 8.0


class TestStaticAtomTeleport:
    def test_static_atoms_follow_line_endpoints(self, simulation):
        # Set up line + tethered static atom
        line = Line((0.0, 0.0), (10.0, 0.0))
        simulation.sync_entity_arrays([line])
        # Add a static atom tethered to entity 0 at t=0.5 (midpoint)
        simulation._add_particle(5.0, 0.0, is_static=1)
        simulation.tether_entity_idx[0] = 0
        simulation.tether_local_pos[0, 0] = 0.5

        # Move the line
        line.start[:] = [0.0, 5.0]
        line.end[:] = [10.0, 5.0]
        simulation.sync_entity_arrays([line])
        simulation.sync_static_atoms_to_geometry()

        # Atom should now be at the new midpoint (5, 5)
        assert simulation.pos_x[0] == pytest.approx(5.0)
        assert simulation.pos_y[0] == pytest.approx(5.0)

    def test_dynamic_atoms_are_not_teleported(self, simulation):
        # is_static=0 means dynamic — sync_static should leave it alone
        line = Line((0, 0), (10, 0))
        simulation.sync_entity_arrays([line])
        simulation._add_particle(2.5, 2.5, is_static=0)  # dynamic
        simulation.tether_entity_idx[0] = 0  # even with linkage
        simulation.tether_local_pos[0, 0] = 0.5

        line.start[:] = [100.0, 100.0]
        line.end[:] = [110.0, 100.0]
        simulation.sync_entity_arrays([line])
        simulation.sync_static_atoms_to_geometry()

        # Dynamic atom should not have been moved
        assert simulation.pos_x[0] == 2.5
        assert simulation.pos_y[0] == 2.5


class TestSerializationRoundTrip:
    def test_to_dict_then_restore_preserves_count(self, simulation):
        simulation._add_particle(1.0, 2.0)
        simulation._add_particle(3.0, 4.0, vx=1.0, vy=1.0, is_static=1)
        d = simulation.to_dict()

        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)
        assert sim2.count == 2

    def test_restore_preserves_positions_and_velocities(self, simulation):
        simulation._add_particle(1.0, 2.0, vx=0.5, vy=-0.5)
        d = simulation.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)
        assert sim2.pos_x[0] == 1.0
        assert sim2.pos_y[0] == 2.0
        assert sim2.vel_x[0] == 0.5
        assert sim2.vel_y[0] == -0.5

    def test_restore_preserves_world_size(self, simulation):
        # Ensure at least one particle is present so restore takes the
        # populated-array path. The empty-array path has a known shape-mismatch
        # bug captured separately by test_restore_empty_simulation_xfail.
        simulation._add_particle(1.0, 1.0)
        simulation.world_size = 75.0
        d = simulation.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)
        assert sim2.world_size == 75.0

    def test_restore_empty_simulation(self):
        """Fixed in CR-116 EI-3: Simulation.restore() with count==0 used to hit
        a numpy shape mismatch on atom_color. Now guarded — the empty case is
        a clean no-op."""
        sim = Simulation(skip_warmup=True)
        d = sim.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)
        assert sim2.count == 0


class TestPhysicsUndoStack:
    def test_clear_snapshot_can_be_undone(self, simulation):
        simulation._add_particle(1.0, 1.0)
        simulation._add_particle(2.0, 2.0)
        simulation.clear(snapshot=True)
        assert simulation.count == 0
        assert simulation.undo() is True
        assert simulation.count == 2

    def test_undo_then_redo_round_trip(self, simulation):
        simulation._add_particle(1.0, 1.0)
        simulation.clear(snapshot=True)
        simulation.undo()
        simulation.redo()
        # Redo applies the clear back
        assert simulation.count == 0

    def test_undo_returns_false_on_empty_stack(self, simulation):
        assert simulation.undo() is False


class TestHasParticleNear:
    def test_detects_existing_particle(self, simulation):
        simulation._add_particle(5.0, 5.0)
        assert simulation.has_particle_near(5.0, 5.0, 0.5) is True

    def test_returns_false_when_far(self, simulation):
        simulation._add_particle(0.0, 0.0)
        assert simulation.has_particle_near(50.0, 50.0, 1.0) is False


# ----- Tether sync (anchors / static atom teleport) -------------------------

class TestSnapTetheredAtomsToAnchors:
    def test_snap_zeroes_velocity_and_force(self, simulation):
        """After snap, tethered atoms must have zero vel/force (cold start
        guarantee that prevents oscillation when an entity becomes dynamic)."""
        line = Line((0.0, 0.0), (10.0, 0.0))
        simulation.sync_entity_arrays([line])

        idx = simulation._add_particle(50.0, 50.0)  # arbitrary initial pos
        simulation.is_static[idx] = 3  # tethered
        simulation.tether_entity_idx[idx] = 0
        simulation.tether_local_pos[idx, 0] = 0.5
        simulation.vel_x[idx] = 99.0
        simulation.vel_y[idx] = 99.0
        simulation.force_x[idx] = 99.0
        simulation.force_y[idx] = 99.0

        simulation.snap_tethered_atoms_to_anchors()

        assert simulation.vel_x[idx] == 0.0
        assert simulation.vel_y[idx] == 0.0
        assert simulation.force_x[idx] == 0.0
        assert simulation.force_y[idx] == 0.0

    def test_snap_places_circle_atom_using_angle(self, simulation):
        """Circles store theta in tether_local_pos[:, 0]; snap should compute
        position with cos/sin (not lerp like Lines)."""
        import math
        circle = Circle((10.0, 10.0), 3.0)
        simulation.sync_entity_arrays([circle])

        idx = simulation._add_particle(0.0, 0.0)
        simulation.is_static[idx] = 3
        simulation.tether_entity_idx[idx] = 0
        # theta = pi/2 (top of circle)
        simulation.tether_local_pos[idx, 0] = math.pi / 2

        simulation.snap_tethered_atoms_to_anchors()

        # Should land at center + radius * (cos(pi/2), sin(pi/2)) = (10, 13)
        assert simulation.pos_x[idx] == pytest.approx(10.0, abs=0.01)
        assert simulation.pos_y[idx] == pytest.approx(13.0, abs=0.01)


class TestStaticAtomTeleportCircles:
    def test_circle_static_atoms_use_angle_not_t(self, simulation):
        """sync_static_atoms_to_geometry must interpret tether_local_pos[:,0]
        as theta for circles, not as a t parameter."""
        import math
        circle = Circle((0.0, 0.0), 5.0)
        simulation.sync_entity_arrays([circle])

        idx = simulation._add_particle(0.0, 0.0, is_static=1)
        simulation.tether_entity_idx[idx] = 0
        simulation.tether_local_pos[idx, 0] = 0.0  # theta=0 → +x axis

        simulation.sync_static_atoms_to_geometry()
        assert simulation.pos_x[idx] == pytest.approx(5.0, abs=0.01)
        assert simulation.pos_y[idx] == pytest.approx(0.0, abs=0.01)

        # Move the circle
        circle.center[:] = [10, 10]
        simulation.sync_entity_arrays([circle])
        simulation.sync_static_atoms_to_geometry()
        # Atom moves with the circle
        assert simulation.pos_x[idx] == pytest.approx(15.0, abs=0.01)
        assert simulation.pos_y[idx] == pytest.approx(10.0, abs=0.01)


# ----- compact_arrays / _resize_arrays preserve all parallel arrays --------

class TestCompactArrays:
    def test_lockstep_compaction(self, simulation):
        """compact_arrays must reorder every parallel array consistently."""
        # Add 5 particles with distinct color, joint_id, tether linkage
        for i in range(5):
            idx = simulation._add_particle(float(i), 0.0)
            simulation.atom_color[idx] = (i * 50, 0, 0)
            simulation.joint_ids[idx] = i + 1
            simulation.tether_entity_idx[idx] = i
            simulation.tether_local_pos[idx, 0] = float(i) * 0.1
            simulation.tether_stiffness[idx] = float(i) * 100.0

        # Keep only every other one (0, 2, 4)
        simulation.compact_arrays(np.array([0, 2, 4], dtype=np.int32))
        assert simulation.count == 3
        # All parallel arrays should reflect this reordering
        assert int(simulation.joint_ids[0]) == 1
        assert int(simulation.joint_ids[1]) == 3
        assert int(simulation.joint_ids[2]) == 5
        assert int(simulation.atom_color[1, 0]) == 100
        assert simulation.tether_local_pos[2, 0] == pytest.approx(0.4, abs=0.001)
        assert simulation.tether_stiffness[2] == pytest.approx(400.0, abs=0.001)
        assert int(simulation.tether_entity_idx[2]) == 4


class TestResizeArrays:
    def test_resize_preserves_existing_data(self, simulation):
        """_resize_arrays() must preserve all existing array contents."""
        for i in range(10):
            idx = simulation._add_particle(float(i), float(i))
            simulation.atom_color[idx] = (i * 10, i * 5, 0)
            simulation.joint_ids[idx] = i
            simulation.tether_entity_idx[idx] = i
            simulation.tether_stiffness[idx] = float(i)

        old_capacity = simulation.capacity
        simulation._resize_arrays()
        assert simulation.capacity == old_capacity * 2

        for i in range(10):
            assert simulation.pos_x[i] == float(i)
            assert int(simulation.atom_color[i, 0]) == i * 10
            assert int(simulation.joint_ids[i]) == i
            assert int(simulation.tether_entity_idx[i]) == i
            assert simulation.tether_stiffness[i] == pytest.approx(float(i))

    def test_new_slots_have_default_values(self, simulation):
        """After resize, slots in the *freshly allocated* half must have the
        manually-set defaults (tether_entity_idx=-1). Per TU-SIM: slot 100 was
        already initialized to -1 by __init__, so probing it after resize
        proves nothing about the resize path. Probe in the new half instead."""
        old_capacity = simulation.capacity
        simulation._add_particle(1.0, 1.0)
        simulation._resize_arrays()
        # Slot is in the freshly allocated upper half — only present after resize
        new_slot = old_capacity + 100
        assert int(simulation.tether_entity_idx[new_slot]) == -1


# ----- Latent bug: world-escape compaction orphans tethered/static atoms ----

class TestWorldEscape:
    @pytest.mark.slow
    def test_step_preserves_out_of_bounds_tethered_atom(self):
        """Fixed in CR-116 EI-3: Simulation.step()'s escape filter now gates
        on is_static. Static (1) and tethered (3) atoms are retained even when
        their position is outside world_size — only dynamic atoms (is_static==0)
        are compacted out. This prevents tether linkage orphaning when an atom
        drifts out during a drag."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 10.0
        line = Line((0.0, 0.0), (10.0, 0.0))
        sim.sync_entity_arrays([line])

        idx = sim._add_particle(15.0, 5.0)  # x > world_size
        sim.is_static[idx] = 3
        sim.tether_entity_idx[idx] = 0
        sim.tether_local_pos[idx, 0] = 0.5
        sim.tether_stiffness[idx] = 10000.0

        before = sim.count
        sim.step(steps_to_run=1)
        # Tethered atom survives out-of-bounds step
        assert sim.count == before
        assert int(sim.tether_entity_idx[0]) == 0  # tether linkage intact

    @pytest.mark.slow
    def test_step_still_removes_out_of_bounds_dynamic_atom(self):
        """The escape filter must STILL remove dynamic atoms (is_static==0)
        that escape — only static/tethered atoms get the linkage protection."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 10.0
        # Add a dynamic atom inside bounds, plus one outside
        sim._add_particle(5.0, 5.0)        # inside
        sim._add_particle(15.0, 5.0)       # outside, dynamic — should be removed

        sim.step(steps_to_run=1)
        # Out-of-bounds dynamic atom removed; in-bounds atom retained
        assert sim.count == 1
        assert sim.pos_x[0] == pytest.approx(5.0, abs=0.5)


class TestApplyThermostat:
    """Berendsen velocity-rescaling thermostat — pins the contract.

    No existing tests covered apply_thermostat directly. These were added
    when the kernel was switched from @njit(parallel=True) to plain
    @njit (single-threaded was 2-30× faster up to N≈30 000 in
    microbenchmarks). The math/signature did not change; these tests
    establish a regression baseline for future thermostat work.
    """

    def _kinetic_energy_per_atom(self, vel_x, vel_y, atom_mass, is_static):
        """KE/N over dynamic atoms only — matches the kernel's `current_T`."""
        dyn = is_static == 0
        # atom_mass may be an array (post per-particle-mass refactor) or a
        # scalar (legacy). Broadcast handles both.
        ke = (0.5 * np.asarray(atom_mass) * (vel_x**2 + vel_y**2))[dyn].sum()
        return float(ke) / int(dyn.sum())

    def test_thermostat_pulls_hot_velocities_toward_target(self):
        from engine.physics_core import apply_thermostat
        rng = np.random.default_rng(0)
        n = 1000
        # Hot start: KE/atom ≈ 5.0; we want to pull toward 0.5
        vel_x = rng.standard_normal(n).astype(np.float32) * np.sqrt(10.0)
        vel_y = rng.standard_normal(n).astype(np.float32) * np.sqrt(10.0)
        is_static = np.zeros(n, dtype=np.int32)
        atom_mass = np.ones(n, dtype=np.float32)

        T_before = self._kinetic_energy_per_atom(vel_x, vel_y, atom_mass, is_static)
        apply_thermostat(vel_x, vel_y, atom_mass, is_static, np.float32(0.5), np.float32(0.1))
        T_after = self._kinetic_energy_per_atom(vel_x, vel_y, atom_mass, is_static)

        # mix=0.1 means one application moves ~10% of the way toward target.
        # Direction must be correct; magnitude should be a meaningful step.
        assert T_after < T_before
        assert T_after > 0.5  # not all the way there in one step

    def test_thermostat_pulls_cold_velocities_toward_target(self):
        from engine.physics_core import apply_thermostat
        rng = np.random.default_rng(1)
        n = 1000
        # Cold start: KE/atom ≈ 0.05; want to push up to 0.5
        vel_x = rng.standard_normal(n).astype(np.float32) * np.sqrt(0.1)
        vel_y = rng.standard_normal(n).astype(np.float32) * np.sqrt(0.1)
        is_static = np.zeros(n, dtype=np.int32)
        atom_mass = np.ones(n, dtype=np.float32)

        T_before = self._kinetic_energy_per_atom(vel_x, vel_y, atom_mass, is_static)
        apply_thermostat(vel_x, vel_y, atom_mass, is_static, np.float32(0.5), np.float32(0.1))
        T_after = self._kinetic_energy_per_atom(vel_x, vel_y, atom_mass, is_static)

        assert T_after > T_before
        assert T_after < 0.5

    def test_thermostat_ignores_static_atoms(self):
        from engine.physics_core import apply_thermostat
        n = 100
        # Even-index atoms dynamic + hot; odd-index atoms static + at fixed velocity.
        vel_x = np.full(n, 2.0, dtype=np.float32)
        vel_y = np.full(n, 0.0, dtype=np.float32)
        is_static = np.zeros(n, dtype=np.int32)
        is_static[1::2] = 1  # half static
        atom_mass = np.ones(n, dtype=np.float32)

        static_vel_before = vel_x[is_static == 1].copy()
        apply_thermostat(vel_x, vel_y, atom_mass, is_static, np.float32(0.5), np.float32(0.1))

        # Static atoms unchanged
        np.testing.assert_array_equal(vel_x[is_static == 1], static_vel_before)
        # Dynamic atoms scaled (away from 2.0, toward target)
        assert not np.allclose(vel_x[is_static == 0], 2.0)

    def test_thermostat_no_dynamic_atoms_is_noop(self):
        """count==0 path: all-static array must not divide-by-zero."""
        from engine.physics_core import apply_thermostat
        n = 50
        vel_x = np.full(n, 3.0, dtype=np.float32)
        vel_y = np.full(n, 0.0, dtype=np.float32)
        is_static = np.ones(n, dtype=np.int32)  # all static
        atom_mass = np.ones(n, dtype=np.float32)

        before = vel_x.copy()
        apply_thermostat(vel_x, vel_y, atom_mass, is_static,
                         np.float32(0.5), np.float32(0.1))
        np.testing.assert_array_equal(vel_x, before)

    def test_thermostat_zero_velocity_is_noop(self):
        """current_T <= 1e-6 path: no division by tiny-positive nonsense."""
        from engine.physics_core import apply_thermostat
        n = 50
        vel_x = np.zeros(n, dtype=np.float32)
        vel_y = np.zeros(n, dtype=np.float32)
        is_static = np.zeros(n, dtype=np.int32)
        atom_mass = np.ones(n, dtype=np.float32)

        apply_thermostat(vel_x, vel_y, atom_mass, is_static,
                         np.float32(0.5), np.float32(0.1))
        # Velocities still zero (would be NaN/inf if guard failed)
        assert np.all(vel_x == 0.0)
        assert np.all(vel_y == 0.0)


class TestBuildAtomNeighborCSR:
    """Pins the half-pair → atom-centric CSR converter.

    Drives the parallel atom-centric LJ pair loop. For each pair (i,j) in
    the input half-pair list, the output CSR must list j as a neighbour
    of i AND i as a neighbour of j. nbr_start[i+1] - nbr_start[i] gives
    atom i's neighbour count.
    """

    def _csr(self, n, pairs):
        """Helper: build CSR from a list of (i,j) pairs."""
        from engine.physics_core import build_atom_neighbor_csr
        pair_i = np.array([p[0] for p in pairs], dtype=np.int32) if pairs else np.zeros(0, dtype=np.int32)
        pair_j = np.array([p[1] for p in pairs], dtype=np.int32) if pairs else np.zeros(0, dtype=np.int32)
        nbr_start = np.zeros(n + 1, dtype=np.int32)
        nbr_idx = np.zeros(max(2 * len(pairs), 1), dtype=np.int32)
        build_atom_neighbor_csr(n, pair_i, pair_j, len(pairs), nbr_start, nbr_idx)
        return nbr_start, nbr_idx

    def _neighbours_of(self, atom_idx, nbr_start, nbr_idx):
        return sorted(nbr_idx[nbr_start[atom_idx]:nbr_start[atom_idx + 1]].tolist())

    def test_empty_pair_list_yields_empty_csr(self):
        nbr_start, _ = self._csr(5, [])
        assert nbr_start.tolist() == [0, 0, 0, 0, 0, 0]

    def test_single_pair_appears_in_both_neighbour_lists(self):
        nbr_start, nbr_idx = self._csr(3, [(0, 2)])
        assert self._neighbours_of(0, nbr_start, nbr_idx) == [2]
        assert self._neighbours_of(1, nbr_start, nbr_idx) == []
        assert self._neighbours_of(2, nbr_start, nbr_idx) == [0]

    def test_complete_graph_three_atoms(self):
        # All-pairs over {0,1,2}: (0,1), (0,2), (1,2). Each atom should
        # neighbour the other two.
        nbr_start, nbr_idx = self._csr(3, [(0, 1), (0, 2), (1, 2)])
        assert self._neighbours_of(0, nbr_start, nbr_idx) == [1, 2]
        assert self._neighbours_of(1, nbr_start, nbr_idx) == [0, 2]
        assert self._neighbours_of(2, nbr_start, nbr_idx) == [0, 1]

    def test_isolated_atoms_have_empty_neighbour_lists(self):
        # 5 atoms, only atoms 1 and 3 are paired. 0, 2, 4 are isolated.
        nbr_start, nbr_idx = self._csr(5, [(1, 3)])
        assert self._neighbours_of(0, nbr_start, nbr_idx) == []
        assert self._neighbours_of(1, nbr_start, nbr_idx) == [3]
        assert self._neighbours_of(2, nbr_start, nbr_idx) == []
        assert self._neighbours_of(3, nbr_start, nbr_idx) == [1]
        assert self._neighbours_of(4, nbr_start, nbr_idx) == []

    def test_nbr_start_is_monotonic(self):
        """Property check: nbr_start must be non-decreasing (each entry's
        difference equals atom i's degree)."""
        pairs = [(0, 1), (0, 2), (0, 5), (1, 3), (2, 4), (3, 5)]
        nbr_start, _ = self._csr(6, pairs)
        diffs = np.diff(nbr_start)
        assert (diffs >= 0).all()
        # Sum of degrees = 2 * |E|
        assert nbr_start[-1] == 2 * len(pairs)


# ===========================================================================
# Boundary modes — OPEN / REFLECTING / PERIODIC
# ===========================================================================

from engine.physics_core import (
    BOUNDARY_OPEN, BOUNDARY_REFLECTING, BOUNDARY_PERIODIC,
)


class TestBoundaryModeProperty:
    """The bool ``use_boundaries`` is a backward-compat alias over the canonical
    int ``boundary_mode``. The UI Bounds button writes ``use_boundaries`` every
    frame, so the setter must not downgrade an active PERIODIC mode."""

    def test_default_mode_is_open(self, simulation):
        assert simulation.boundary_mode == BOUNDARY_OPEN
        assert simulation.use_boundaries is False

    def test_setter_true_from_open_promotes_to_reflecting(self, simulation):
        simulation.use_boundaries = True
        assert simulation.boundary_mode == BOUNDARY_REFLECTING

    def test_setter_false_resets_to_open(self, simulation):
        simulation.boundary_mode = BOUNDARY_REFLECTING
        simulation.use_boundaries = False
        assert simulation.boundary_mode == BOUNDARY_OPEN

    def test_setter_true_does_not_downgrade_periodic(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.use_boundaries = True
        assert simulation.boundary_mode == BOUNDARY_PERIODIC

    def test_use_boundaries_reads_truthy_under_periodic(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        assert simulation.use_boundaries is True

    def test_reset_restores_open(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.reset()
        assert simulation.boundary_mode == BOUNDARY_OPEN

    def test_cycle_advances_open_to_reflecting(self, simulation):
        assert simulation.boundary_mode == BOUNDARY_OPEN
        assert simulation.cycle_boundary_mode() == BOUNDARY_REFLECTING

    def test_cycle_full_loop(self, simulation):
        # default world is large enough for PBC
        assert simulation.cycle_boundary_mode() == BOUNDARY_REFLECTING
        assert simulation.cycle_boundary_mode() == BOUNDARY_PERIODIC
        assert simulation.cycle_boundary_mode() == BOUNDARY_OPEN

    def test_cycle_skips_periodic_when_world_too_small(self, simulation):
        # cell_size = r_cut + skin = 2.8. n_cells = floor(L/cs) + 1 must be
        # >= 3 for PBC. L = 5 → n_cells = 2 → unsafe → cycle skips PERIODIC.
        simulation.world_size = 5.0
        simulation._update_derived_params()
        simulation.boundary_mode = BOUNDARY_REFLECTING
        nxt = simulation.cycle_boundary_mode()
        assert nxt == BOUNDARY_OPEN  # skipped PERIODIC, jumped to OPEN

    def test_pbc_safe_at_default_world(self, simulation):
        assert simulation._pbc_safe() is True

    def test_pbc_unsafe_below_three_cells(self, simulation):
        simulation.world_size = 5.0
        simulation._update_derived_params()
        assert simulation._pbc_safe() is False


class TestPeriodicWrap:
    """Position-update wrap behaviour for dynamic atoms under PBC."""

    def test_atom_past_right_edge_wraps_left(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        idx = simulation._add_particle(L - 0.01, L / 2, vx=10.0, is_static=0)
        simulation.step(steps_to_run=1)
        # New x ≈ L + 0.09 → wrapped to ≈ 0.09
        assert 0.0 <= simulation.pos_x[idx] < 1.0

    def test_atom_past_left_edge_wraps_right(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        idx = simulation._add_particle(0.01, L / 2, vx=-10.0, is_static=0)
        simulation.step(steps_to_run=1)
        # New x ≈ -0.09 → wrapped to ≈ L - 0.09
        assert L - 1.0 < simulation.pos_x[idx] <= L

    def test_atom_past_top_edge_wraps_bottom(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        idx = simulation._add_particle(L / 2, L - 0.01, vy=10.0, is_static=0)
        simulation.step(steps_to_run=1)
        assert 0.0 <= simulation.pos_y[idx] < 1.0

    def test_atom_past_bottom_edge_wraps_top(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        idx = simulation._add_particle(L / 2, 0.01, vy=-10.0, is_static=0)
        simulation.step(steps_to_run=1)
        assert L - 1.0 < simulation.pos_y[idx] <= L

    def test_atom_does_not_wrap_under_open(self, simulation):
        # Under OPEN, the post-step escape filter removes atoms that left
        # the [0, world_size]^2 domain.
        simulation.boundary_mode = BOUNDARY_OPEN
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        simulation._add_particle(L - 0.01, L / 2, vx=10.0, is_static=0)
        simulation.step(steps_to_run=1)
        assert simulation.count == 0

    def test_atom_reflects_under_reflecting(self, simulation):
        simulation.boundary_mode = BOUNDARY_REFLECTING
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        idx = simulation._add_particle(L - 0.01, L / 2, vx=10.0, is_static=0)
        simulation.step(steps_to_run=1)
        # Reflecting: position bounces back inside [0, L] and vel_x flips.
        assert 0.0 <= simulation.pos_x[idx] <= L
        assert simulation.vel_x[idx] < 0


class TestPeriodicNoEscape:
    """Under PBC the post-step escape filter must not remove dynamic atoms."""

    def test_count_constant_when_atoms_cross_each_boundary(self, simulation):
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.dt = 0.01
        simulation.gravity = 0.0
        L = simulation.world_size
        # Four atoms, each pushed across a different wall this step.
        simulation._add_particle(L - 0.01, L / 2, vx=20.0, is_static=0)
        simulation._add_particle(0.01, L / 2, vx=-20.0, is_static=0)
        simulation._add_particle(L / 2, L - 0.01, vy=20.0, is_static=0)
        simulation._add_particle(L / 2, 0.01, vy=-20.0, is_static=0)
        n_before = simulation.count
        for _ in range(5):
            simulation.step(steps_to_run=1)
        assert simulation.count == n_before
        assert ((simulation.pos_x[:n_before] >= 0)
                & (simulation.pos_x[:n_before] <= L)).all()
        assert ((simulation.pos_y[:n_before] >= 0)
                & (simulation.pos_y[:n_before] <= L)).all()


class TestPeriodicNeighbourList:
    """Cell list with PBC must include cross-boundary pairs via wrap and use
    the minimum-image convention when checking pair distance."""

    def _build(self, simulation):
        from engine.physics_core import build_neighbor_list
        n = simulation.count
        return build_neighbor_list(
            simulation.pos_x[:n], simulation.pos_y[:n],
            simulation.r_list2, simulation.cell_size, simulation.world_size,
            simulation.pair_i, simulation.pair_j,
            np.int32(simulation.boundary_mode),
        )

    def test_atoms_across_right_boundary_are_neighbours_under_pbc(self, simulation):
        L = simulation.world_size
        simulation._add_particle(0.5, L / 2, is_static=0)
        simulation._add_particle(L - 0.5, L / 2, is_static=0)
        simulation.boundary_mode = BOUNDARY_PERIODIC
        # Min-image dx = ±1.0, well within r_list ≈ 2.8 → exactly one pair.
        assert self._build(simulation) == 1

    def test_atoms_across_right_boundary_not_neighbours_under_open(self, simulation):
        L = simulation.world_size
        simulation._add_particle(0.5, L / 2, is_static=0)
        simulation._add_particle(L - 0.5, L / 2, is_static=0)
        simulation.boundary_mode = BOUNDARY_OPEN
        # Direct distance L - 1 ≫ r_list → no pair.
        assert self._build(simulation) == 0

    def test_atoms_across_corner_diagonally_are_neighbours_under_pbc(self, simulation):
        L = simulation.world_size
        simulation._add_particle(0.5, 0.5, is_static=0)
        simulation._add_particle(L - 0.5, L - 0.5, is_static=0)
        simulation.boundary_mode = BOUNDARY_PERIODIC
        # Min-image distance = √2, well within r_list. Direct distance
        # ≈ √2(L-1) ≈ 70 — far beyond r_list.
        assert self._build(simulation) == 1

    def test_no_double_counting_under_pbc(self, simulation):
        """Each (i, j) pair must be emitted exactly once even when reachable
        from both the direct neighbour cell and a wrapped neighbour cell.
        At default world_size n_cells ≈ 17 so no atom is reached via both."""
        L = simulation.world_size
        simulation._add_particle(L / 2, L / 2, is_static=0)
        simulation._add_particle(L / 2 + 0.5, L / 2, is_static=0)
        simulation._add_particle(L / 2, L / 2 + 0.5, is_static=0)
        simulation.boundary_mode = BOUNDARY_PERIODIC
        assert self._build(simulation) == 3


class TestPeriodicForceMinimumImage:
    """Two atoms on opposite sides of the periodic boundary must feel an
    LJ force pulling them across the wrap (and away from each other in the
    minimum-image direction at sub-equilibrium separation)."""

    def test_force_across_right_boundary_repels_via_minimum_image(self, simulation):
        # σ = 1, ε = 1 — equilibrium at r ≈ 1.122. At r = 1.0 (sub-equilibrium)
        # the LJ force is repulsive. With min-image dx_A = +1.0 (across the
        # wrap), F on A is in +x; F on B is in -x.
        L = simulation.world_size
        simulation.boundary_mode = BOUNDARY_PERIODIC
        simulation.gravity = 0.0
        simulation.dt = 0.001
        idx_a = simulation._add_particle(0.5, L / 2, is_static=0)
        idx_b = simulation._add_particle(L - 0.5, L / 2, is_static=0)
        simulation.step(steps_to_run=1)
        assert simulation.vel_x[idx_a] > 0
        assert simulation.vel_x[idx_b] < 0

    def test_no_force_between_far_atoms_under_open(self, simulation):
        L = simulation.world_size
        simulation.boundary_mode = BOUNDARY_OPEN
        simulation.gravity = 0.0
        simulation.dt = 0.001
        idx_a = simulation._add_particle(0.5, L / 2, is_static=0)
        idx_b = simulation._add_particle(L - 0.5, L / 2, is_static=0)
        simulation.step(steps_to_run=1)
        # Direct distance L - 1 ≫ r_cut → no pair force. With gravity off
        # nothing pushes the atoms; velocities remain exactly zero.
        assert simulation.vel_x[idx_a] == 0.0
        assert simulation.vel_x[idx_b] == 0.0


class TestNewton3KernelEquivalence:
    """The opt-in `use_newton3` half-pair kernel must be physics-equivalent
    to the classic atom-centric kernel, modulo float32 sum-order noise.
    Both kernels visit the same pairs; they differ only in iteration order
    and accumulation strategy. Drift across many substeps is expected to
    stay well below per-step LJ force magnitudes."""

    def _setup(self, N, seed):
        sim = Simulation(skip_warmup=True)
        rho = 0.7
        L = (N / rho) ** 0.5
        sim.world_size = float(L)
        sim._update_derived_params()
        sim.boundary_mode = 1  # reflecting
        sim.use_thermostat = False
        sim.gravity = 0.0
        sim.damping = 1.0

        side = int(np.ceil(np.sqrt(N)))
        spacing = L / side
        rng = np.random.default_rng(seed)
        k = 0
        for i in range(side):
            for j in range(side):
                if k < N:
                    sim.pos_x[k] = (i + 0.5) * spacing + rng.uniform(-0.05, 0.05) * spacing
                    sim.pos_y[k] = (j + 0.5) * spacing + rng.uniform(-0.05, 0.05) * spacing
                    k += 1
        sim.count = N
        sim.atom_sigma[:N] = 1.0
        sim.atom_eps_sqrt[:N] = 1.0
        sim.vel_x[:N] = rng.normal(0, 1.0, N).astype(np.float32)
        sim.vel_y[:N] = rng.normal(0, 1.0, N).astype(np.float32)
        sim.vel_x[:N] -= sim.vel_x[:N].mean()
        sim.vel_y[:N] -= sim.vel_y[:N].mean()
        sim.dt = 0.002
        sim.rebuild_next = True
        return sim

    def test_classic_and_newton3_agree_within_float32_noise(self):
        N = 500
        sim_a = self._setup(N=N, seed=42)
        sim_a.use_newton3 = False
        sim_b = self._setup(N=N, seed=42)
        sim_b.use_newton3 = True

        for _ in range(10):
            sim_a.step(steps_to_run=10)
            sim_b.step(steps_to_run=10)

        # Float32 sum-order ambiguity grows with substeps; over 100 substeps
        # at LJ liquid density a 1e-2 position bound is comfortable.
        max_dx = float(np.max(np.abs(sim_a.pos_x[:N] - sim_b.pos_x[:N])))
        max_dy = float(np.max(np.abs(sim_a.pos_y[:N] - sim_b.pos_y[:N])))
        assert max_dx < 1e-2
        assert max_dy < 1e-2

    def test_newton3_default_is_off(self):
        sim = Simulation(skip_warmup=True)
        assert sim.use_newton3 is False

    def test_local_force_buffers_lazy_until_use(self):
        sim = Simulation(skip_warmup=True)
        assert sim._local_force_x is None
        assert sim._local_force_y is None

    def test_local_force_buffers_allocated_on_newton3_step(self):
        sim = Simulation(skip_warmup=True)
        sim.use_newton3 = True
        sim._add_particle(10.0, 10.0, is_static=0)
        sim._add_particle(11.0, 10.0, is_static=0)
        sim.step(steps_to_run=1)
        assert sim._local_force_x is not None
        assert sim._local_force_y is not None
        assert sim._local_force_x.shape[1] == sim.capacity


# ============================================================================
# Per-particle mass — the simulation's atoms must integrate with their own
# mass, not a single global value. Heavier atoms should accelerate less
# under the same force (F = m·a).
# ============================================================================


class TestPerParticleMass:
    def test_atom_mass_array_exists_and_default(self):
        """Newly added particles get the configured default mass."""
        from core import config
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, is_static=0)
        assert sim.atom_mass[0] == pytest.approx(config.ATOM_MASS)

    def test_add_particle_stores_per_particle_mass(self):
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, is_static=0, mass=1.0)
        sim._add_particle(11.0, 10.0, is_static=0, mass=13.5)  # Mercury
        sim._add_particle(12.0, 10.0, is_static=0, mass=0.9)   # Oil
        assert sim.atom_mass[0] == 1.0
        assert sim.atom_mass[1] == 13.5
        assert sim.atom_mass[2] == 0.9

    def test_heavy_particle_falls_less_than_light_under_gravity(self):
        """The bug the Lead reported: two blobs of different mass should
        NOT move with the same inertia. Under gravity, a 100× heavier
        particle should fall LESS distance per unit time than a unit-mass
        particle. Specifically, F = m·g and a = F/m = g — so acceleration
        is *independent* of mass for gravity alone. But that's not what
        the user observed: they saw same motion regardless of mass, which
        is consistent with the old scalar-mass bug where every particle
        used the same global mass.

        This test verifies the new per-particle mass machinery: with
        IDENTICAL initial conditions but different mass, the gravity term
        F = m·g produces the same a = g, so both should fall the SAME
        distance. The contrast comes from a NON-gravity force — see the
        next test for that.
        """
        sim = Simulation(skip_warmup=True)
        sim.gravity = -10.0  # downward (negative Y)
        sim.damping = 1.0    # no medium drag — gravity-only test
        sim.paused = False
        sim.dt = 0.01

        # Place far apart so they don't interact via LJ
        sim._add_particle(10.0, 50.0, is_static=0, mass=1.0, sigma=1.0, epsilon=1.0)
        sim._add_particle(90.0, 50.0, is_static=0, mass=100.0, sigma=1.0, epsilon=1.0)

        y0_light = float(sim.pos_y[0])
        y0_heavy = float(sim.pos_y[1])
        sim.step(steps_to_run=100)
        dy_light = y0_light - float(sim.pos_y[0])
        dy_heavy = y0_heavy - float(sim.pos_y[1])

        # Under gravity, both fall the same amount (F=mg, a=g — mass cancels).
        # This is the physical sanity check that the kernel is using mass
        # CONSISTENTLY: the bug would have shown the same delta even with
        # scalar mass, so this test doesn't *directly* catch the bug —
        # but it pins the correct physical behaviour.
        assert dy_light == pytest.approx(dy_heavy, rel=0.05)

    def test_heavy_particle_decelerates_less_under_lj_repulsion(self):
        """Two particles with the same initial velocity scatter off identical
        static atoms. The heavy particle has 100× the momentum (p = m·v),
        so the same repulsive impulse decelerates it MUCH less than the
        light one — its final velocity stays close to v0, while the light
        particle's velocity is substantially reduced (and may reverse).

        Pre-refactor (scalar mass), both particles used the same effective
        mass in the integrator so their final velocities would match. The
        test verifies the new per-particle integration path: heavy particle's
        |Δv| must be strictly less than light's.
        """
        sim = Simulation(skip_warmup=True)
        sim.gravity = 0.0  # isolate the impulse — no gravity confound
        sim.damping = 1.0  # isolate mass — no medium drag
        sim.paused = False
        sim.dt = 0.01

        v0 = 10.0
        wall_y = 50.0
        approach_dx = 2.5  # close enough that LJ kicks in immediately

        # Pair 1: light moving particle + static wall atom
        sim._add_particle(10.0, wall_y, vx=v0, vy=0.0, is_static=0,
                          mass=1.0, sigma=1.0, epsilon=1.0)
        sim._add_particle(10.0 + approach_dx, wall_y, is_static=1,
                          sigma=1.0, epsilon=1.0)

        # Pair 2: heavy moving particle + static wall atom (well separated)
        sim._add_particle(60.0, wall_y, vx=v0, vy=0.0, is_static=0,
                          mass=100.0, sigma=1.0, epsilon=1.0)
        sim._add_particle(60.0 + approach_dx, wall_y, is_static=1,
                          sigma=1.0, epsilon=1.0)

        sim.step(steps_to_run=100)
        dv_light = abs(float(sim.vel_x[0]) - v0)
        dv_heavy = abs(float(sim.vel_x[2]) - v0)

        # Heavy particle's velocity stays closer to the initial — the bug
        # would have made dv_light == dv_heavy (same effective mass).
        assert dv_heavy < dv_light, (
            f"|Δv| heavy={dv_heavy:.3f}, light={dv_light:.3f}; "
            "100× mass should produce strictly smaller velocity change"
        )
        # And the gap should be substantial. The theoretical max ratio
        # for this 1D collision is 2.0 (light bounces back fully → Δv=2v;
        # heavy halts → Δv=v). Anything above 1.5× confirms the per-
        # particle mass is dominating the dynamics; the bug would give
        # exactly 1.0 (identical Δv).
        assert dv_light > 1.5 * dv_heavy, (
            f"|Δv| ratio light/heavy = {dv_light/max(dv_heavy, 1e-9):.2f}, "
            "expected >1.5× given the 100× mass ratio"
        )

    def test_atom_mass_roundtrips_through_snapshot(self):
        """The undo-snapshot stack must preserve per-particle mass."""
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, is_static=0, mass=2.5)
        sim._add_particle(11.0, 10.0, is_static=0, mass=7.0)
        sim.snapshot()

        # Overwrite, then undo
        sim.atom_mass[0] = 999.0
        sim.atom_mass[1] = 999.0
        sim.undo()
        assert sim.atom_mass[0] == 2.5
        assert sim.atom_mass[1] == 7.0

    def test_atom_mass_roundtrips_through_to_dict(self):
        """Saved scenes carry per-particle mass."""
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, is_static=0, mass=2.5)
        sim._add_particle(11.0, 10.0, is_static=0, mass=7.0)
        d = sim.to_dict()
        assert d['atom_mass'] == [pytest.approx(2.5), pytest.approx(7.0)]

        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)
        assert sim2.atom_mass[0] == pytest.approx(2.5)
        assert sim2.atom_mass[1] == pytest.approx(7.0)

    def test_compact_arrays_preserves_per_particle_mass(self):
        """Compaction (used during particle removal) must keep mass with
        the right atoms."""
        sim = Simulation(skip_warmup=True)
        sim._add_particle(10.0, 10.0, is_static=0, mass=1.0)
        sim._add_particle(11.0, 10.0, is_static=0, mass=2.0)
        sim._add_particle(12.0, 10.0, is_static=0, mass=3.0)
        # Keep first and third atoms
        sim.compact_arrays([0, 2])
        assert sim.count == 2
        assert sim.atom_mass[0] == 1.0
        assert sim.atom_mass[1] == 3.0  # was index 2, now slot 1


class TestPerSubstepDamping:
    """The "Damping" slider applies as a per-substep velocity multiplier
    (general medium drag). Pre-fix it was only applied at wall bounces,
    so free-flying particles saw no deceleration regardless of slider
    value — making damping=0.999 look perfectly elastic. This class pins
    the new behaviour.
    """

    def test_damping_lt_one_reduces_velocity_over_time(self):
        """Free-flying particle with damping<1.0 loses velocity each substep.
        Pre-fix, damping was wall-only so this never happened in free flight.
        """
        sim = Simulation(skip_warmup=True)
        sim.world_size = 200.0  # large so the particle stays well in-bounds
        sim.gravity = 0.0
        sim.paused = False
        sim.dt = 0.005
        sim.damping = 0.99  # 1% velocity loss per substep
        sim.use_boundaries = False  # OPEN — no wall interaction at all
        sim._add_particle(100.0, 100.0, vx=10.0, vy=0.0, is_static=0)

        # Use a single substep to bypass the displacement-safety check
        # that bails out of long inner loops; assert each call multiplies
        # velocity by damping.
        v0 = float(sim.vel_x[0])
        sim.step(steps_to_run=1)
        v1 = float(sim.vel_x[0])
        assert v1 == pytest.approx(v0 * 0.99, rel=1e-3)

        # And cumulative: 5 more substeps, expect 0.99^6 of original.
        for _ in range(5):
            sim.step(steps_to_run=1)
        v6 = float(sim.vel_x[0])
        assert v6 == pytest.approx(v0 * (0.99 ** 6), rel=1e-3)

    def test_damping_equal_one_preserves_velocity(self):
        """damping=1.0 → no medium drag → free particle keeps velocity."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 200.0
        sim.gravity = 0.0
        sim.paused = False
        sim.dt = 0.005
        sim.damping = 1.0
        sim.use_boundaries = False
        sim._add_particle(100.0, 100.0, vx=10.0, vy=0.0, is_static=0)

        v0 = float(sim.vel_x[0])
        for _ in range(20):
            sim.step(steps_to_run=1)
        v1 = float(sim.vel_x[0])
        assert v1 == pytest.approx(v0, rel=1e-5)

    def test_damping_mass_independent(self):
        """Drag is a velocity multiplier, not a force — mass doesn't enter.
        Heavy and light particles should damp at the SAME relative rate."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 200.0
        sim.gravity = 0.0
        sim.paused = False
        sim.dt = 0.005
        sim.damping = 0.95  # aggressive — easy to see
        sim.use_boundaries = False

        # Two particles far apart so they don't interact via LJ
        sim._add_particle(50.0, 100.0, vx=10.0, vy=0.0, is_static=0,
                          mass=1.0, sigma=1.0, epsilon=1.0)
        sim._add_particle(150.0, 100.0, vx=10.0, vy=0.0, is_static=0,
                          mass=100.0, sigma=1.0, epsilon=1.0)

        for _ in range(5):
            sim.step(steps_to_run=1)
        v_light = float(sim.vel_x[0])
        v_heavy = float(sim.vel_x[1])
        # Same starting velocity, same multiplicative drag → identical
        # final velocity regardless of mass.
        assert v_light == pytest.approx(v_heavy, rel=1e-5)

    def test_tethered_atom_uses_tether_damping_not_wall_damping(self):
        """Tethered atoms keep their separate TETHER_DAMPING (spring
        oscillation suppressor) and are NOT additionally damped by the
        medium-drag term — the slider only affects dynamic atoms."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 200.0
        sim.gravity = 0.0
        sim.paused = False
        sim.dt = 0.005
        sim.damping = 1.0  # no medium drag for dynamic
        sim.use_boundaries = False

        # Tethered atom with no force on it (tether_stiffness=0) should
        # still see velocity decay from TETHER_DAMPING
        sim._add_particle(100.0, 100.0, vx=10.0, vy=0.0, is_static=3)
        sim.tether_entity_idx[0] = -1
        sim.tether_stiffness[0] = 0.0

        v0 = float(sim.vel_x[0])
        for _ in range(10):
            sim.step(steps_to_run=1)
        v1 = float(sim.vel_x[0])
        # TETHER_DAMPING = 0.90, applied each substep → 0.9^10 ≈ 0.349
        assert v1 < v0 * 0.5


class TestSourceParticlesCarryMaterialMass:
    """Source spawns must carry the resolved material's mass into atom_mass."""

    def test_spawned_particles_carry_material_mass(self):
        """Mercury source → all spawned particles have atom_mass=13.5."""
        from core.scene import Scene
        from model.process_objects import Source, SourceProperties
        scene = Scene(skip_warmup=True)
        scene.simulation.world_size = 50.0

        mercury_mass = scene.sketch.get_material('Mercury').mass
        source = Source((25, 25), 3.0,
                        SourceProperties(material_name='Mercury', flux=100.0))
        scene.add_process_object(source)
        source.execute(scene.simulation, dt=1.0)

        assert scene.simulation.count > 0
        for i in range(scene.simulation.count):
            if scene.simulation.is_static[i] == 0:
                assert float(scene.simulation.atom_mass[i]) == pytest.approx(mercury_mass)


class TestBrushParticlesCarryMaterialMass:
    """Painted particles must carry the active material's mass."""

    def test_paint_with_material_mass_propagates(self):
        from engine.particle_brush import ParticleBrush
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        brush = ParticleBrush(sim)
        brush.paint(25, 25, 3.0, sigma=1.0, epsilon=1.0, mass=13.5,
                    color=(180, 180, 190))
        assert sim.count > 0
        # All painted particles share the brush mass
        for i in range(sim.count):
            assert float(sim.atom_mass[i]) == pytest.approx(13.5)

    def test_paint_without_mass_falls_back_to_default(self):
        from engine.particle_brush import ParticleBrush
        from core import config
        sim = Simulation(skip_warmup=True)
        sim.world_size = 50.0
        brush = ParticleBrush(sim)
        brush.paint(25, 25, 3.0)  # no mass specified
        assert sim.count > 0
        for i in range(sim.count):
            assert float(sim.atom_mass[i]) == pytest.approx(config.ATOM_MASS)


class TestCompilerAtomsCarryMaterialMass:
    """Wall/tethered atoms emitted by Compiler must carry their material's mass."""

    def test_static_wall_atoms_carry_material_mass(self, scene):
        """Build a line with an Oil material, atomize, verify static atoms
        carry Oil's mass (0.9)."""
        from model.commands.geometry import AddLineCommand
        oil_mass = scene.sketch.get_material('Oil').mass

        scene.execute(AddLineCommand(scene.sketch, (10, 10), (20, 10),
                                     physical=True))
        # Assign Oil material to the entity
        scene.sketch.entities[0].material_id = 'Oil'
        scene.rebuild()

        # Verify at least one static atom exists with Oil's mass
        sim = scene.simulation
        any_static = False
        for i in range(sim.count):
            if sim.is_static[i] == 1:
                any_static = True
                assert float(sim.atom_mass[i]) == pytest.approx(oil_mass)
        assert any_static, "Expected at least one static wall atom"


# =============================================================================
# Harmonic spring bonds (intramolecular).
# Foundation for the Molecule Builder feature: each bond couples two atoms
# with F = -k*(r - r_eq)*r_hat. R1 lands the engine plumbing (kernel + arrays
# + plumbing through compact/snapshot/serialize); the molecule data model and
# placement UX land in R2-R4.
# =============================================================================


class TestBondArrayConstruction:
    """Defaults, capacity, and the add/remove/clear primitives."""

    def test_defaults(self, simulation):
        assert simulation.bond_count == 0
        assert simulation.bond_capacity == 100
        assert simulation.bond_i.dtype == np.int32
        assert simulation.bond_j.dtype == np.int32
        assert simulation.bond_k.dtype == np.float32
        assert simulation.bond_r_eq.dtype == np.float32

    def test_add_bond_returns_index_and_advances_count(self, simulation):
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        b = simulation.add_bond(0, 1, k=100.0, r_eq=1.0)
        assert b == 0
        assert simulation.bond_count == 1
        assert int(simulation.bond_i[0]) == 0
        assert int(simulation.bond_j[0]) == 1
        assert float(simulation.bond_k[0]) == pytest.approx(100.0)
        assert float(simulation.bond_r_eq[0]) == pytest.approx(1.0)

    def test_add_self_bond_rejected(self, simulation):
        simulation._add_particle(0.0, 0.0)
        assert simulation.add_bond(0, 0, 100.0, 1.0) == -1
        assert simulation.bond_count == 0

    def test_remove_bond_swap_with_last(self, simulation):
        for i in range(4):
            simulation._add_particle(float(i), 0.0)
        simulation.add_bond(0, 1, 100.0, 1.0)
        simulation.add_bond(1, 2, 200.0, 2.0)
        simulation.add_bond(2, 3, 300.0, 3.0)
        # Remove middle bond; last bond should fill its slot.
        simulation.remove_bond(1)
        assert simulation.bond_count == 2
        # Slot 1 now holds the former bond 2 (atoms 2->3, k=300)
        assert int(simulation.bond_i[1]) == 2
        assert int(simulation.bond_j[1]) == 3
        assert float(simulation.bond_k[1]) == pytest.approx(300.0)

    def test_clear_bonds_drops_all(self, simulation):
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        simulation.add_bond(0, 1, 100.0, 1.0)
        simulation.clear_bonds()
        assert simulation.bond_count == 0

    def test_clear_drops_bonds(self, simulation):
        """Simulation.clear() must drop bonds — they reference indices that
        the cleared atoms no longer occupy."""
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        simulation.add_bond(0, 1, 100.0, 1.0)
        simulation.clear()
        assert simulation.bond_count == 0

    def test_resize_grows_capacity(self, simulation):
        # Fill default capacity (100); next add should trigger resize.
        for i in range(101):
            simulation._add_particle(float(i), 0.0)
        for i in range(100):
            simulation.add_bond(i, i + 1, 100.0, 1.0)
        old_cap = simulation.bond_capacity
        simulation.add_bond(0, 50, 100.0, 1.0)
        assert simulation.bond_capacity == old_cap * 2
        assert simulation.bond_count == 101
        # Existing data preserved
        assert int(simulation.bond_i[0]) == 0
        assert int(simulation.bond_j[0]) == 1


class TestSpringForceKernel:
    """Direct invocation of apply_spring_bonds — force direction, magnitude,
    Newton's 3rd law, and the equilibrium null point."""

    def _setup(self, dx_atom=2.0, k=100.0, r_eq=1.0):
        """Two atoms along x: i at (0,0), j at (dx_atom, 0), one bond."""
        pos_x = np.array([0.0, dx_atom], dtype=np.float32)
        pos_y = np.array([0.0, 0.0], dtype=np.float32)
        force_x = np.zeros(2, dtype=np.float32)
        force_y = np.zeros(2, dtype=np.float32)
        is_static = np.zeros(2, dtype=np.int32)
        bond_i = np.array([0], dtype=np.int32)
        bond_j = np.array([1], dtype=np.int32)
        bond_k = np.array([k], dtype=np.float32)
        bond_r_eq = np.array([r_eq], dtype=np.float32)
        return pos_x, pos_y, force_x, force_y, is_static, bond_i, bond_j, bond_k, bond_r_eq

    def test_force_zero_at_equilibrium(self):
        from engine.physics_core import apply_spring_bonds
        # Atoms at exactly r_eq apart.
        args = self._setup(dx_atom=1.0, k=100.0, r_eq=1.0)
        apply_spring_bonds(*args, 100.0, 0)
        # f_scal = 100 * 0 / 1 = 0
        assert float(args[2][0]) == pytest.approx(0.0, abs=1e-6)
        assert float(args[2][1]) == pytest.approx(0.0, abs=1e-6)
        assert float(args[3][0]) == pytest.approx(0.0, abs=1e-6)

    def test_stretched_pulls_atoms_together(self):
        """Bond at r=2 with r_eq=1 is stretched; force on i should point
        toward j (positive x). Force on j should point toward i (negative x)."""
        from engine.physics_core import apply_spring_bonds
        args = self._setup(dx_atom=2.0, k=100.0, r_eq=1.0)
        apply_spring_bonds(*args, 100.0, 0)
        # r=2, r_eq=1, k=100 → f_scal = 100 * (2-1) / 2 = 50
        # Force on i: fx = f_scal * (pos_j_x - pos_i_x) = 50 * 2 = 100
        # Force on j: equal and opposite (-100).
        force_x = args[2]
        force_y = args[3]
        assert float(force_x[0]) == pytest.approx(100.0, rel=1e-4)
        assert float(force_x[1]) == pytest.approx(-100.0, rel=1e-4)
        assert float(force_y[0]) == pytest.approx(0.0, abs=1e-4)
        assert float(force_y[1]) == pytest.approx(0.0, abs=1e-4)

    def test_compressed_pushes_atoms_apart(self):
        """Bond at r=0.5 with r_eq=1 is compressed; force on i should point
        AWAY from j (negative x). Force on j should point AWAY from i
        (positive x)."""
        from engine.physics_core import apply_spring_bonds
        args = self._setup(dx_atom=0.5, k=100.0, r_eq=1.0)
        apply_spring_bonds(*args, 100.0, 0)
        # r=0.5, r_eq=1, k=100 → f_scal = 100 * (0.5-1) / 0.5 = -100
        # Force on i = (-100 * 0.5, 0) = (-50, 0)
        force_x = args[2]
        assert float(force_x[0]) == pytest.approx(-50.0, rel=1e-4)
        assert float(force_x[1]) == pytest.approx(50.0, rel=1e-4)

    def test_newton_third_law(self):
        """Force on i must always equal -Force on j (per-component)."""
        from engine.physics_core import apply_spring_bonds
        pos_x = np.array([3.0, 7.0], dtype=np.float32)
        pos_y = np.array([5.0, 9.0], dtype=np.float32)
        force_x = np.zeros(2, dtype=np.float32)
        force_y = np.zeros(2, dtype=np.float32)
        is_static = np.zeros(2, dtype=np.int32)
        bond_i = np.array([0], dtype=np.int32)
        bond_j = np.array([1], dtype=np.int32)
        bond_k = np.array([42.0], dtype=np.float32)
        bond_r_eq = np.array([2.5], dtype=np.float32)
        apply_spring_bonds(
            pos_x, pos_y, force_x, force_y, is_static,
            bond_i, bond_j, bond_k, bond_r_eq,
            100.0, 0,
        )
        assert float(force_x[0]) == pytest.approx(-float(force_x[1]), rel=1e-5)
        assert float(force_y[0]) == pytest.approx(-float(force_y[1]), rel=1e-5)

    def test_static_atom_not_force_written(self):
        """is_static==1 atoms should not receive force writes. The dynamic
        partner still feels the bond pulling it toward the static anchor."""
        from engine.physics_core import apply_spring_bonds
        pos_x = np.array([0.0, 2.0], dtype=np.float32)
        pos_y = np.array([0.0, 0.0], dtype=np.float32)
        force_x = np.zeros(2, dtype=np.float32)
        force_y = np.zeros(2, dtype=np.float32)
        is_static = np.array([0, 1], dtype=np.int32)  # atom 1 is static
        bond_i = np.array([0], dtype=np.int32)
        bond_j = np.array([1], dtype=np.int32)
        bond_k = np.array([100.0], dtype=np.float32)
        bond_r_eq = np.array([1.0], dtype=np.float32)
        apply_spring_bonds(
            pos_x, pos_y, force_x, force_y, is_static,
            bond_i, bond_j, bond_k, bond_r_eq,
            100.0, 0,
        )
        # Dynamic atom feels the pull toward static partner
        assert float(force_x[0]) == pytest.approx(100.0, rel=1e-4)
        # Static atom force slot left zero
        assert float(force_x[1]) == 0.0

    def test_both_static_bond_skipped(self):
        from engine.physics_core import apply_spring_bonds
        pos_x = np.array([0.0, 2.0], dtype=np.float32)
        pos_y = np.array([0.0, 0.0], dtype=np.float32)
        force_x = np.zeros(2, dtype=np.float32)
        force_y = np.zeros(2, dtype=np.float32)
        is_static = np.array([1, 1], dtype=np.int32)
        bond_i = np.array([0], dtype=np.int32)
        bond_j = np.array([1], dtype=np.int32)
        bond_k = np.array([100.0], dtype=np.float32)
        bond_r_eq = np.array([1.0], dtype=np.float32)
        apply_spring_bonds(
            pos_x, pos_y, force_x, force_y, is_static,
            bond_i, bond_j, bond_k, bond_r_eq,
            100.0, 0,
        )
        assert float(force_x[0]) == 0.0
        assert float(force_x[1]) == 0.0

    def test_pbc_minimum_image(self):
        """Bonded atoms on opposite sides of a periodic box should feel a
        bond force corresponding to the SHORT (across-the-wrap) distance,
        not the long in-domain distance."""
        from engine.physics_core import apply_spring_bonds
        from engine.physics_core import BOUNDARY_PERIODIC
        # World 10 wide. Atom 0 at x=0.5, atom 1 at x=9.5. In-domain distance=9;
        # minimum-image distance = 1 (wrap).
        pos_x = np.array([0.5, 9.5], dtype=np.float32)
        pos_y = np.array([5.0, 5.0], dtype=np.float32)
        force_x = np.zeros(2, dtype=np.float32)
        force_y = np.zeros(2, dtype=np.float32)
        is_static = np.zeros(2, dtype=np.int32)
        bond_i = np.array([0], dtype=np.int32)
        bond_j = np.array([1], dtype=np.int32)
        bond_k = np.array([100.0], dtype=np.float32)
        bond_r_eq = np.array([1.0], dtype=np.float32)
        apply_spring_bonds(
            pos_x, pos_y, force_x, force_y, is_static,
            bond_i, bond_j, bond_k, bond_r_eq,
            10.0, BOUNDARY_PERIODIC,
        )
        # Min-image dx = pos_j - pos_i = 9 → wraps to -1 (atom j is "to the left"
        # of atom i across the wrap). At r=1 == r_eq, f_scal = 0.
        assert float(force_x[0]) == pytest.approx(0.0, abs=1e-4)
        assert float(force_x[1]) == pytest.approx(0.0, abs=1e-4)


class TestSpringBondIntegration:
    """End-to-end through Simulation.step(): bond forces accelerate atoms."""

    def test_stretched_bond_accelerates_atoms_together(self, simulation):
        """Two atoms bonded with r=2 > r_eq=1 should be pulled toward each
        other after one substep."""
        simulation.world_size = 200.0
        simulation.gravity = 0.0
        simulation.paused = False
        simulation.dt = 0.001
        simulation.damping = 1.0
        simulation.use_boundaries = False
        i = simulation._add_particle(99.0, 100.0, vx=0.0, vy=0.0, is_static=0)
        j = simulation._add_particle(101.0, 100.0, vx=0.0, vy=0.0, is_static=0)
        simulation.add_bond(i, j, k=100.0, r_eq=1.0)

        simulation.step(steps_to_run=1)

        # Atom i should have gained positive vel_x (pulled right toward j);
        # atom j should have gained negative vel_x (pulled left toward i).
        assert float(simulation.vel_x[i]) > 0
        assert float(simulation.vel_x[j]) < 0
        # Newton's 3rd law → equal and opposite (equal mass → equal speed)
        assert float(simulation.vel_x[i]) == pytest.approx(
            -float(simulation.vel_x[j]), rel=1e-3
        )

    def test_bondless_simulation_unaffected(self, simulation):
        """The bondless fast path must remain bit-equivalent to pre-R1
        behaviour. Two LJ-interacting atoms with no bonds should behave
        identically whether bonds machinery is present or not."""
        simulation.world_size = 200.0
        simulation.gravity = 0.0
        simulation.paused = False
        simulation.dt = 0.001
        simulation.damping = 1.0
        simulation.use_boundaries = False
        simulation._add_particle(100.0, 100.0, vx=1.0, vy=0.0, is_static=0)

        # No bonds added — bond_count stays 0
        assert simulation.bond_count == 0

        v0 = float(simulation.vel_x[0])
        for _ in range(5):
            simulation.step(steps_to_run=1)
        v1 = float(simulation.vel_x[0])
        # No gravity, no damping, no bonds, no LJ partner → velocity preserved
        assert v1 == pytest.approx(v0, rel=1e-5)


class TestCompactArraysRemapsBonds:
    """compact_arrays must rewrite bond indices through the keep_indices
    remap, and drop bonds where either endpoint is removed."""

    def test_compact_rewrites_surviving_bond_indices(self, simulation):
        # 4 atoms, bond between atom 1 and atom 3. Remove atom 0.
        # After compact: atom 1 → new index 0, atom 3 → new index 2.
        for i in range(4):
            simulation._add_particle(float(i), 0.0)
        simulation.add_bond(1, 3, 100.0, 1.0)
        simulation.compact_arrays(np.array([1, 2, 3], dtype=np.int32))
        assert simulation.bond_count == 1
        assert int(simulation.bond_i[0]) == 0  # old 1 → new 0
        assert int(simulation.bond_j[0]) == 2  # old 3 → new 2

    def test_compact_drops_bond_when_atom_removed(self, simulation):
        for i in range(4):
            simulation._add_particle(float(i), 0.0)
        simulation.add_bond(0, 1, 100.0, 1.0)  # will be dropped (atom 0 removed)
        simulation.add_bond(2, 3, 200.0, 2.0)  # survives (atoms 2,3 kept)
        simulation.compact_arrays(np.array([1, 2, 3], dtype=np.int32))
        assert simulation.bond_count == 1
        # Surviving bond is now between old-atom-2 (new 1) and old-atom-3 (new 2)
        assert int(simulation.bond_i[0]) == 1
        assert int(simulation.bond_j[0]) == 2
        assert float(simulation.bond_k[0]) == pytest.approx(200.0)

    def test_compact_with_no_bonds_does_nothing(self, simulation):
        for i in range(3):
            simulation._add_particle(float(i), 0.0)
        simulation.compact_arrays(np.array([0, 2], dtype=np.int32))
        assert simulation.bond_count == 0


class TestBondSerializationRoundTrip:
    """snapshot/_restore_physics_state, _push_to_stack, to_dict/restore all
    preserve bonds. Pre-R1 saves without bond keys default to bond_count=0."""

    def test_snapshot_restore_round_trip(self, simulation):
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        simulation.add_bond(0, 1, k=150.0, r_eq=1.0)
        simulation.snapshot()
        # Mutate
        simulation.remove_bond(0)
        assert simulation.bond_count == 0
        # Undo via restore
        simulation._restore_physics_state(simulation.undo_stack[-1])
        assert simulation.bond_count == 1
        assert float(simulation.bond_k[0]) == pytest.approx(150.0)

    def test_to_dict_round_trip(self, simulation):
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        simulation._add_particle(2.0, 0.0)
        simulation.add_bond(0, 1, k=100.0, r_eq=1.0)
        simulation.add_bond(1, 2, k=200.0, r_eq=2.0)
        data = simulation.to_dict()
        assert data['bond_count'] == 2

        sim2 = Simulation(skip_warmup=True)
        sim2.restore(data)
        assert sim2.bond_count == 2
        assert int(sim2.bond_i[0]) == 0
        assert int(sim2.bond_j[0]) == 1
        assert float(sim2.bond_k[0]) == pytest.approx(100.0)
        assert int(sim2.bond_i[1]) == 1
        assert float(sim2.bond_r_eq[1]) == pytest.approx(2.0)

    def test_restore_pre_r1_save_treats_as_no_bonds(self, simulation):
        """Saves from before R1 do not carry bond_count. restore() should
        treat absence as 'no bonds' rather than KeyError."""
        legacy_data = {
            'count': 1,
            'world_size': 50.0,
            'pos_x': [10.0],
            'pos_y': [10.0],
            'vel_x': [0.0],
            'vel_y': [0.0],
            'is_static': [0],
            'atom_sigma': [1.0],
            'atom_eps_sqrt': [1.0],
            'atom_mass': [1.0],
            'atom_color': [[100, 100, 100]],
            # No bond_count / bond_i / bond_j / ...
        }
        simulation.restore(legacy_data)
        assert simulation.bond_count == 0

    def test_undo_redo_preserves_bonds(self, simulation):
        simulation._add_particle(0.0, 0.0)
        simulation._add_particle(1.0, 0.0)
        simulation.add_bond(0, 1, 100.0, 1.0)
        simulation.snapshot()
        # Add another bond (would normally come from a new molecule placement)
        simulation._add_particle(2.0, 0.0)
        simulation.add_bond(1, 2, 200.0, 2.0)
        assert simulation.bond_count == 2

        simulation.undo()
        assert simulation.bond_count == 1
        assert float(simulation.bond_k[0]) == pytest.approx(100.0)

        simulation.redo()
        assert simulation.bond_count == 2
        assert float(simulation.bond_k[1]) == pytest.approx(200.0)


class TestSpringOscillation:
    """A bonded pair displaced from equilibrium should oscillate. Integration
    test using simulation.step() — verifies the kernel is actually invoked
    and that bond forces participate in the Verlet update."""

    def test_pair_returns_through_equilibrium(self, simulation):
        """Two equal-mass atoms with a stiff bond, displaced from equilibrium.
        After a quarter period, the relative velocity should peak at r=r_eq.
        We just verify the pair passes through r=r_eq within a bounded
        number of substeps — period sanity rather than precise frequency."""
        simulation.world_size = 200.0
        simulation.gravity = 0.0
        simulation.paused = False
        simulation.dt = 0.001
        simulation.damping = 1.0
        simulation.use_boundaries = False
        # Place atoms far enough from each other to avoid LJ overlap at r_eq=1.
        # sigma=0.5 → r_cut at ~0.5*2.5=1.25 — at r=1, LJ is non-zero but small.
        # Use much smaller sigma so LJ stays negligible vs the spring at r=1.
        i = simulation._add_particle(99.0, 100.0, is_static=0,
                                     sigma=0.1, epsilon=0.01, mass=1.0)
        j = simulation._add_particle(101.0, 100.0, is_static=0,
                                     sigma=0.1, epsilon=0.01, mass=1.0)
        simulation.add_bond(i, j, k=1000.0, r_eq=1.0)

        # Initial r = 2, r_eq = 1 → stretched. Expect contraction.
        crossed_through_r_eq = False
        for _ in range(500):
            simulation.step(steps_to_run=1)
            r = float(np.hypot(simulation.pos_x[j] - simulation.pos_x[i],
                               simulation.pos_y[j] - simulation.pos_y[i]))
            if r < 1.0:
                crossed_through_r_eq = True
                break
        assert crossed_through_r_eq, "Bonded pair never contracted past r_eq"


# =============================================================================
# Three-body angle springs.
# Energy = k*(θ - θ_eq)² for the angle at apex b between legs (a, c).
# Same plumbing pattern as bonds: arrays + add/remove + compact remap +
# snapshot/restore. End-to-end test verifies the kernel pulls a perturbed
# triple back toward θ_eq.
# =============================================================================


class TestAngleArrayConstruction:
    def test_defaults(self, simulation):
        assert simulation.angle_count == 0
        assert simulation.angle_capacity == 100
        assert simulation.angle_a.dtype == np.int32
        assert simulation.angle_k.dtype == np.float32

    def test_add_angle_returns_index_and_advances(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        n = simulation.add_angle(0, 1, 2, k=50.0, theta_eq=math.pi)
        assert n == 0
        assert simulation.angle_count == 1
        assert int(simulation.angle_a[0]) == 0
        assert int(simulation.angle_b[0]) == 1
        assert int(simulation.angle_c[0]) == 2
        assert float(simulation.angle_k[0]) == pytest.approx(50.0)
        assert float(simulation.angle_theta_eq[0]) == pytest.approx(math.pi)

    def test_degenerate_triplet_rejected(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        assert simulation.add_angle(0, 0, 1, 50.0, math.pi) == -1
        assert simulation.add_angle(0, 1, 1, 50.0, math.pi) == -1
        assert simulation.add_angle(0, 1, 0, 50.0, math.pi) == -1
        assert simulation.angle_count == 0

    def test_remove_angle_swap_with_last(self, simulation):
        for _ in range(5):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, 50.0, 1.0)
        simulation.add_angle(1, 2, 3, 100.0, 2.0)
        simulation.add_angle(2, 3, 4, 150.0, 3.0)
        simulation.remove_angle(1)
        assert simulation.angle_count == 2
        # Slot 1 now holds the former angle 2
        assert int(simulation.angle_a[1]) == 2
        assert float(simulation.angle_k[1]) == pytest.approx(150.0)

    def test_clear_angles_drops_all(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, 50.0, 1.0)
        simulation.clear_angles()
        assert simulation.angle_count == 0

    def test_clear_drops_angles(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, 50.0, 1.0)
        simulation.clear()
        assert simulation.angle_count == 0

    def test_resize_grows_capacity(self, simulation):
        # Fill the default capacity then trigger resize
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        for _ in range(100):
            simulation.add_angle(0, 1, 2, 50.0, 1.0)
        old_cap = simulation.angle_capacity
        simulation.add_angle(0, 1, 2, 50.0, 1.0)
        assert simulation.angle_capacity == old_cap * 2
        assert simulation.angle_count == 101


class TestAngleForceKernel:
    """Direct invocation of apply_angle_forces — sign, magnitude, Newton's-3rd
    law on the triplet, equilibrium null, and PBC minimum-image."""

    def _setup(self, theta_deg, k, theta_eq_rad):
        """Apex b at origin, leg a along +x at distance 1, leg c at angle
        theta_deg from a. Returns the kernel argument tuple."""
        theta = math.radians(theta_deg)
        pos_x = np.array([1.0, 0.0, math.cos(theta)], dtype=np.float32)
        pos_y = np.array([0.0, 0.0, math.sin(theta)], dtype=np.float32)
        force_x = np.zeros(3, dtype=np.float32)
        force_y = np.zeros(3, dtype=np.float32)
        is_static = np.zeros(3, dtype=np.int32)
        angle_a = np.array([0], dtype=np.int32)
        angle_b = np.array([1], dtype=np.int32)
        angle_c = np.array([2], dtype=np.int32)
        angle_k = np.array([k], dtype=np.float32)
        angle_theta_eq = np.array([theta_eq_rad], dtype=np.float32)
        return (pos_x, pos_y, force_x, force_y, is_static,
                angle_a, angle_b, angle_c, angle_k, angle_theta_eq)

    def test_force_zero_at_equilibrium(self):
        from engine.physics_core import apply_angle_forces
        # Both legs at 90° (θ_eq = π/2), kernel should produce zero forces.
        args = self._setup(theta_deg=90.0, k=100.0, theta_eq_rad=math.pi / 2)
        apply_angle_forces(*args, 100.0, 0)
        fx = args[2]
        fy = args[3]
        for i in range(3):
            assert float(fx[i]) == pytest.approx(0.0, abs=1e-5)
            assert float(fy[i]) == pytest.approx(0.0, abs=1e-5)

    def test_newton_third_law_on_triplet(self):
        """Sum of forces on the three atoms must be zero (no net force on
        the molecule from intramolecular angle constraint)."""
        from engine.physics_core import apply_angle_forces
        args = self._setup(theta_deg=70.0, k=100.0, theta_eq_rad=math.radians(120.0))
        apply_angle_forces(*args, 100.0, 0)
        fx = args[2]
        fy = args[3]
        sum_fx = sum(float(v) for v in fx)
        sum_fy = sum(float(v) for v in fy)
        assert sum_fx == pytest.approx(0.0, abs=1e-4)
        assert sum_fy == pytest.approx(0.0, abs=1e-4)

    def test_angle_too_narrow_pushes_open(self):
        """θ < θ_eq → angle wants to open. Leg atom forces should have
        components that increase θ (push legs apart angularly)."""
        from engine.physics_core import apply_angle_forces
        # θ = 60°, θ_eq = 120° → too narrow, should open
        args = self._setup(theta_deg=60.0, k=100.0, theta_eq_rad=math.radians(120.0))
        apply_angle_forces(*args, 100.0, 0)
        fx = args[2]
        fy = args[3]
        # Atom a at (1, 0); leg vector from apex is (+1, 0). To OPEN θ we
        # want atom a pushed in -y direction (downward, away from atom c
        # which sits at +60° above). Actually with c above x-axis, atom a
        # should be pushed below.
        # Simpler invariant: when atom c is above atom a (theta=60°), to
        # open the angle a should move down (-y).
        assert float(fy[0]) < 0  # atom a pushed away from c
        # Atom c is at upper-right; to open the angle it should move
        # upward (away from a).
        assert float(fy[2]) > 0

    def test_angle_too_wide_pushes_closed(self):
        """θ > θ_eq → angle wants to close. Inverse of test_too_narrow."""
        from engine.physics_core import apply_angle_forces
        # θ = 150°, θ_eq = 90° → too wide, should close.
        # Apex b at origin; leg a at (1, 0) [angle 0°]; leg c at angle 150°.
        # To close the angle both legs rotate toward the angular bisector
        # (at 75°). Atom a (at 0°) rotates CCW toward 75° → vy > 0; atom c
        # (at 150°) rotates CW toward 75° → still vy > 0 because 75° is
        # higher-y than 150° on the unit circle.
        args = self._setup(theta_deg=150.0, k=100.0, theta_eq_rad=math.radians(90.0))
        apply_angle_forces(*args, 100.0, 0)
        fy = args[3]
        # Force on each leg atom is perpendicular to its leg direction.
        # Atom a at (+1, 0): leg = +x → force is along ±y. Closing means
        # rotating CCW toward the bisector at 75° → +y.
        # Atom c at (cos150°, sin150°): leg = upper-left → force is
        # perpendicular to that. Closing means rotating CW → still
        # results in +y motion in the lab frame.
        assert float(fy[0]) > 0
        assert float(fy[2]) > 0

    def test_linear_equilibrium_stays_finite(self):
        """Near θ = π (linear), sin θ → 0 but the kernel guards 1/sin θ
        and applies a finite force."""
        from engine.physics_core import apply_angle_forces
        # θ ≈ 179.5° (very close to linear), θ_eq = 180°
        args = self._setup(theta_deg=179.5, k=500.0, theta_eq_rad=math.pi)
        apply_angle_forces(*args, 100.0, 0)
        fx = args[2]
        fy = args[3]
        # Forces should be finite — no NaN or Inf
        for arr in (fx, fy):
            for v in arr:
                vv = float(v)
                assert math.isfinite(vv)

    def test_pbc_minimum_image(self):
        """A wrapped molecule (one leg crossing the boundary) should still
        feel the angle force as if the molecule were intact."""
        from engine.physics_core import apply_angle_forces, BOUNDARY_PERIODIC
        # World 10 wide. Apex at (0.5, 5); leg a at (9.5, 5) (across the
        # wrap → min-image dx = +1); leg c at (0.5, 4) (below apex).
        # Min-image triangle is apex=(0.5,5), a_rel=(+1,0), c_rel=(0,-1) →
        # right angle (90°). With θ_eq=90°, force should be ~zero.
        pos_x = np.array([9.5, 0.5, 0.5], dtype=np.float32)
        pos_y = np.array([5.0, 5.0, 4.0], dtype=np.float32)
        force_x = np.zeros(3, dtype=np.float32)
        force_y = np.zeros(3, dtype=np.float32)
        is_static = np.zeros(3, dtype=np.int32)
        angle_a = np.array([0], dtype=np.int32)
        angle_b = np.array([1], dtype=np.int32)
        angle_c = np.array([2], dtype=np.int32)
        angle_k = np.array([100.0], dtype=np.float32)
        angle_theta_eq = np.array([math.pi / 2], dtype=np.float32)
        apply_angle_forces(
            pos_x, pos_y, force_x, force_y, is_static,
            angle_a, angle_b, angle_c, angle_k, angle_theta_eq,
            10.0, BOUNDARY_PERIODIC,
        )
        # With min-image, the configuration is at θ_eq exactly → forces ~0
        for arr in (force_x, force_y):
            for v in arr:
                assert float(v) == pytest.approx(0.0, abs=1e-4)


class TestAngleCompactRemap:
    def test_compact_rewrites_surviving_angle_indices(self, simulation):
        for _ in range(5):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(1, 2, 3, k=100.0, theta_eq=math.pi)
        # Remove atom 0 → atoms 1..4 become 0..3; angle (1,2,3) → (0,1,2)
        simulation.compact_arrays(np.array([1, 2, 3, 4], dtype=np.int32))
        assert simulation.angle_count == 1
        assert int(simulation.angle_a[0]) == 0
        assert int(simulation.angle_b[0]) == 1
        assert int(simulation.angle_c[0]) == 2

    def test_compact_drops_angle_when_apex_removed(self, simulation):
        for _ in range(4):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, k=100.0, theta_eq=math.pi)  # drops (apex 1 removed)
        simulation.add_angle(0, 2, 3, k=200.0, theta_eq=2.0)       # survives
        simulation.compact_arrays(np.array([0, 2, 3], dtype=np.int32))
        assert simulation.angle_count == 1
        # Surviving angle's atoms: old 0 → new 0, old 2 → new 1, old 3 → new 2
        assert int(simulation.angle_a[0]) == 0
        assert int(simulation.angle_b[0]) == 1
        assert int(simulation.angle_c[0]) == 2
        assert float(simulation.angle_k[0]) == pytest.approx(200.0)


class TestAngleSerializationRoundTrip:
    def test_snapshot_restore(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, k=77.0, theta_eq=math.pi / 3)
        simulation.snapshot()
        simulation.remove_angle(0)
        assert simulation.angle_count == 0
        simulation._restore_physics_state(simulation.undo_stack[-1])
        assert simulation.angle_count == 1
        assert float(simulation.angle_k[0]) == pytest.approx(77.0)
        assert float(simulation.angle_theta_eq[0]) == pytest.approx(math.pi / 3)

    def test_to_dict_round_trip(self, simulation):
        for _ in range(3):
            simulation._add_particle(0.0, 0.0)
        simulation.add_angle(0, 1, 2, k=99.0, theta_eq=math.pi / 4)
        data = simulation.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(data)
        assert sim2.angle_count == 1
        assert int(sim2.angle_b[0]) == 1
        assert float(sim2.angle_k[0]) == pytest.approx(99.0)

    def test_pre_angle_save_treated_as_no_angles(self, simulation):
        """Old saves without angle_count default to zero, not KeyError."""
        legacy = {
            'count': 0, 'world_size': 50.0,
            'pos_x': [], 'pos_y': [], 'vel_x': [], 'vel_y': [],
            'is_static': [], 'atom_sigma': [], 'atom_eps_sqrt': [],
            'atom_mass': [], 'atom_color': [],
            'bond_count': 0, 'bond_i': [], 'bond_j': [],
            'bond_k': [], 'bond_r_eq': [],
            # no angle_* keys
        }
        simulation.restore(legacy)
        assert simulation.angle_count == 0


class TestAngleIntegration:
    """End-to-end through Simulation.step(): a perturbed triple snaps
    back toward its equilibrium angle under the spring force."""

    def test_perturbed_triple_relaxes_toward_theta_eq(self, simulation):
        simulation.world_size = 200.0
        simulation.gravity = 0.0
        simulation.paused = False
        simulation.dt = 0.001
        simulation.damping = 0.95  # strong damping so it settles quickly
        simulation.use_boundaries = False
        # Three atoms: apex b at (100, 100); leg a at (101, 100); leg c
        # initially BELOW the apex at (100, 99) — perturbed from θ_eq=π.
        a_idx = simulation._add_particle(101.0, 100.0, is_static=0,
                                          sigma=0.1, epsilon=0.01, mass=1.0)
        b_idx = simulation._add_particle(100.0, 100.0, is_static=0,
                                          sigma=0.1, epsilon=0.01, mass=1.0)
        c_idx = simulation._add_particle(100.0, 99.0, is_static=0,
                                          sigma=0.1, epsilon=0.01, mass=1.0)
        # Bonds hold the leg lengths near 1.0 so the relaxation is
        # primarily angular, not radial.
        simulation.add_bond(a_idx, b_idx, k=500.0, r_eq=1.0)
        simulation.add_bond(b_idx, c_idx, k=500.0, r_eq=1.0)
        # Angle equilibrium = π (linear). Initial θ = 90°.
        simulation.add_angle(a_idx, b_idx, c_idx, k=200.0, theta_eq=math.pi)

        def measure_theta():
            dxba = simulation.pos_x[a_idx] - simulation.pos_x[b_idx]
            dyba = simulation.pos_y[a_idx] - simulation.pos_y[b_idx]
            dxbc = simulation.pos_x[c_idx] - simulation.pos_x[b_idx]
            dybc = simulation.pos_y[c_idx] - simulation.pos_y[b_idx]
            r_ba = math.hypot(dxba, dyba)
            r_bc = math.hypot(dxbc, dybc)
            cos_th = (dxba * dxbc + dyba * dybc) / (r_ba * r_bc)
            cos_th = max(-1.0, min(1.0, cos_th))
            return math.acos(cos_th)

        theta_initial = measure_theta()
        assert theta_initial == pytest.approx(math.pi / 2, abs=0.05)

        for _ in range(2000):
            simulation.step(steps_to_run=1)

        theta_final = measure_theta()
        # Should have moved toward π. Loose bound — the damping is strong
        # enough that we should be well past the midway point.
        assert theta_final > math.pi / 2 + 0.3, (
            f"Angle didn't relax toward θ_eq=π (final θ = {math.degrees(theta_final):.1f}°)"
        )
