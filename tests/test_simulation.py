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
