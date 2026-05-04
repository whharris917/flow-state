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

    @pytest.mark.xfail(reason="Bug: Simulation.restore() with count==0 hits a numpy shape mismatch on atom_color (empty list vs (0,3) target). Captured during CR-116 test suite authoring; fix is a separate CR.", strict=True)
    def test_restore_empty_simulation_xfail(self):
        sim = Simulation(skip_warmup=True)
        d = sim.to_dict()
        sim2 = Simulation(skip_warmup=True)
        sim2.restore(d)


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
        """After resize, new slots must have correct defaults (esp. tether_entity_idx=-1)."""
        # Add one particle, then force resize
        simulation._add_particle(1.0, 1.0)
        simulation._resize_arrays()
        # Slots beyond count should have the manually-set defaults
        assert int(simulation.tether_entity_idx[100]) == -1


# ----- Latent bug: world-escape compaction orphans tethered/static atoms ----

class TestWorldEscapeBug:
    @pytest.mark.xfail(reason="Bug: Simulation.step() compacts atoms whose pos is outside world_size REGARDLESS of is_static. A tethered or static atom that drifts out (e.g., during a drag) gets removed, orphaning the tether linkage. The escape filter at the end of step() does not gate on is_static. Captured during CR-116 TU collaboration.", strict=True)
    @pytest.mark.slow
    def test_step_preserves_out_of_bounds_tethered_atom(self):
        """A tethered atom pushed outside world bounds must NOT be removed by step()."""
        sim = Simulation(skip_warmup=True)
        sim.world_size = 10.0
        # Add a tethered atom and push it outside world bounds
        idx = sim._add_particle(15.0, 5.0)  # x > world_size
        sim.is_static[idx] = 3
        sim.tether_entity_idx[idx] = 0
        sim.tether_local_pos[idx, 0] = 0.5
        sim.tether_stiffness[idx] = 10000.0

        # Need a corresponding entity for the tether kernel; just sync zero
        sim.entity_count = 1
        sim.entity_positions[0] = [0, 0, 10, 0]
        sim.entity_types[0] = 0  # LINE

        before = sim.count
        sim.step(steps_to_run=1)
        # Currently this fails: the atom is compacted out by the world-bounds filter
        assert sim.count == before
        assert int(sim.tether_entity_idx[0]) == 0  # tether linkage intact
