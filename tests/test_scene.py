"""Tests for core/scene.py — orchestration, dirty flags, ProcessObject management."""

import numpy as np
import pytest

from core.scene import Scene
from core.commands import AddLineCommand, MoveEntityCommand, SetPhysicalCommand
from model.geometry import Line, Point
from model.process_objects import Source, SourceProperties


class TestSceneOwnership:
    def test_scene_owns_sketch_simulation_compiler_commands(self, scene):
        assert scene.sketch is not None
        assert scene.simulation is not None
        assert scene.compiler is not None
        assert scene.commands is not None

    def test_alias_properties_delegate_to_sketch(self, scene):
        scene.sketch.add_line((0, 0), (1, 0))
        assert scene.entities is scene.sketch.entities
        assert scene.constraints is scene.sketch.constraints
        assert scene.materials is scene.sketch.materials


class TestDirtyFlags:
    def test_topology_changing_command_sets_topology_dirty(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        assert scene._topology_dirty is True

    def test_non_topology_changing_command_sets_geometry_dirty(self, scene):
        scene.sketch.add_line((0, 0), (1, 0))
        scene._topology_dirty = False
        scene._geometry_dirty = False
        scene.execute(MoveEntityCommand(scene.sketch, 0, 1.0, 0.0))
        # MoveEntityCommand has no changes_topology attribute => geometry_dirty
        assert scene._geometry_dirty is True
        assert scene._topology_dirty is False

    def test_undo_marks_topology_dirty_and_runs_rebuild(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene._topology_dirty = False
        scene.undo()
        # undo() conservatively marks topology dirty (so the next update() picks
        # it up) and calls rebuild() immediately. The flag is left True; only
        # update() resets it.
        assert scene._topology_dirty is True
        assert scene.sketch.entities == []


class TestUndoRedo:
    def test_can_undo_after_execute(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        assert scene.can_undo() is True

    def test_can_redo_after_undo(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.undo()
        assert scene.can_redo() is True

    def test_redo_restores_entity(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (5, 5)))
        scene.undo()
        scene.redo()
        assert len(scene.sketch.entities) == 1


class TestUpdateLoop:
    def test_update_with_empty_scene_does_not_crash(self, scene):
        # Empty sketch + simulation paused => update is a no-op
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)

    def test_update_clears_topology_dirty_flag_after_rebuild(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0)))
        scene.execute(SetPhysicalCommand(scene.sketch, 0, True))
        assert scene._topology_dirty is True
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        assert scene._topology_dirty is False

    def test_update_with_physical_line_emits_atoms(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        # Compiler emitted static atoms during the topology rebuild
        static_count = int(np.sum(scene.simulation.is_static[:scene.simulation.count] == 1))
        assert static_count > 0

    def test_paused_simulation_does_not_advance(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=True)
        # Sim is paused by default — no integration steps should run
        assert scene.simulation.total_steps == 0


class TestProcessObjects:
    def test_add_process_object_registers_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert source in scene.process_objects
        # Handle (center Point) should now appear in the sketch
        assert source.handles["center"] in scene.sketch.entities

    def test_remove_process_object_unregisters_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        scene.remove_process_object(source)
        assert source not in scene.process_objects
        assert source.handles["center"] not in scene.sketch.entities

    def test_get_process_object_for_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        center = source.handles["center"]
        assert scene.get_process_object_for_handle(center) is source

    def test_get_process_object_for_non_handle_returns_none(self, scene):
        non_handle = Point(0.0, 0.0)
        assert scene.get_process_object_for_handle(non_handle) is None

    def test_find_process_object_at_center(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert scene.find_process_object_at(25.0, 25.0) is source

    def test_find_process_object_at_returns_none_when_far(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert scene.find_process_object_at(100.0, 100.0) is None


class TestSceneClearAndNew:
    def test_clear_removes_everything(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.add_process_object(Source((10, 10), 2.0, SourceProperties()))
        scene.clear()
        assert scene.sketch.entities == []
        assert scene.simulation.count == 0
        assert scene.process_objects == []

    def test_new_resets_to_empty_state(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.new()
        assert scene.sketch.entities == []
        assert scene.simulation.count == 0
        assert scene.commands.can_undo() is False


class TestBrushDelegation:
    def test_paint_particles_delegates_to_brush(self, scene):
        added = scene.paint_particles(25.0, 25.0, radius=2.0)
        assert added > 0
        assert scene.simulation.count == added

    def test_erase_particles_delegates_to_brush(self, scene):
        scene.paint_particles(25.0, 25.0, radius=2.0)
        before = scene.simulation.count
        removed = scene.erase_particles(25.0, 25.0, radius=2.0)
        assert removed > 0
        assert scene.simulation.count < before
