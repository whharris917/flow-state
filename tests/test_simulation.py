"""Tests for engine/simulation.py — particle arrays, world, snapshot/restore."""

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

    def _kinetic_energy_per_atom(self, vel_x, vel_y, mass, is_static):
        """KE/N over dynamic atoms only — matches the kernel's `current_T`."""
        dyn = is_static == 0
        ke = 0.5 * mass * (vel_x[dyn]**2 + vel_y[dyn]**2).sum()
        return float(ke) / int(dyn.sum())

    def test_thermostat_pulls_hot_velocities_toward_target(self):
        from engine.physics_core import apply_thermostat
        rng = np.random.default_rng(0)
        n = 1000
        # Hot start: KE/atom ≈ 5.0; we want to pull toward 0.5
        vel_x = rng.standard_normal(n).astype(np.float32) * np.sqrt(10.0)
        vel_y = rng.standard_normal(n).astype(np.float32) * np.sqrt(10.0)
        is_static = np.zeros(n, dtype=np.int32)
        mass = np.float32(1.0)

        T_before = self._kinetic_energy_per_atom(vel_x, vel_y, mass, is_static)
        apply_thermostat(vel_x, vel_y, mass, is_static, np.float32(0.5), np.float32(0.1))
        T_after = self._kinetic_energy_per_atom(vel_x, vel_y, mass, is_static)

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
        mass = np.float32(1.0)

        T_before = self._kinetic_energy_per_atom(vel_x, vel_y, mass, is_static)
        apply_thermostat(vel_x, vel_y, mass, is_static, np.float32(0.5), np.float32(0.1))
        T_after = self._kinetic_energy_per_atom(vel_x, vel_y, mass, is_static)

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
        mass = np.float32(1.0)

        static_vel_before = vel_x[is_static == 1].copy()
        apply_thermostat(vel_x, vel_y, mass, is_static, np.float32(0.5), np.float32(0.1))

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

        before = vel_x.copy()
        apply_thermostat(vel_x, vel_y, np.float32(1.0), is_static,
                         np.float32(0.5), np.float32(0.1))
        np.testing.assert_array_equal(vel_x, before)

    def test_thermostat_zero_velocity_is_noop(self):
        """current_T <= 1e-6 path: no division by tiny-positive nonsense."""
        from engine.physics_core import apply_thermostat
        n = 50
        vel_x = np.zeros(n, dtype=np.float32)
        vel_y = np.zeros(n, dtype=np.float32)
        is_static = np.zeros(n, dtype=np.int32)

        apply_thermostat(vel_x, vel_y, np.float32(1.0), is_static,
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
