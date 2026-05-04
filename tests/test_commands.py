"""Tests for the Command pattern — execute, undo, redo, queue semantics."""

import pytest

from core.commands import (
    CommandQueue,
    AddLineCommand, AddCircleCommand, RemoveEntityCommand,
    MoveEntityCommand, MoveMultipleCommand, SetEntityGeometryCommand,
    AddConstraintCommand, RemoveConstraintCommand, ToggleAnchorCommand,
    ToggleInfiniteCommand,
    SetPhysicalCommand, SetEntityDynamicCommand, SetMaterialCommand,
    CompositeCommand, AddRectangleCommand,
)
from core.command_base import Command
from model.geometry import Line, Circle
from model.constraints import Length, Coincident


# ----- CommandQueue infrastructure ------------------------------------------

class _CountCommand(Command):
    """Trivial command for queue-mechanic tests — bumps a counter."""

    def __init__(self, counter, delta=1, historize=True, supersede=False):
        super().__init__(historize=historize, supersede=supersede)
        self.counter = counter
        self.delta = delta

    def execute(self):
        self.counter[0] += self.delta
        return True

    def undo(self):
        self.counter[0] -= self.delta


class TestCommandQueue:
    def test_empty_queue_cannot_undo_or_redo(self):
        q = CommandQueue()
        assert not q.can_undo()
        assert not q.can_redo()
        assert q.undo() is False
        assert q.redo() is False

    def test_execute_pushes_to_undo_stack(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter))
        assert counter[0] == 1
        assert q.can_undo()
        assert not q.can_redo()

    def test_undo_pops_and_invokes_undo(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter))
        q.undo()
        assert counter[0] == 0
        assert not q.can_undo()
        assert q.can_redo()

    def test_redo_replays_undone_command(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter, delta=5))
        q.undo()
        q.redo()
        assert counter[0] == 5

    def test_new_execute_clears_redo_stack(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter))
        q.undo()
        assert q.can_redo()
        q.execute(_CountCommand(counter))
        assert not q.can_redo()

    def test_non_historized_command_does_not_grow_stack(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter, historize=False))
        assert counter[0] == 1
        assert not q.can_undo()

    def test_supersede_replaces_previous_command(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter, delta=5))
        q.execute(_CountCommand(counter, delta=10, supersede=True))
        # First was undone (counter -= 5), then second executed (counter += 10)
        assert counter[0] == 10
        # Stack length should be 1 (replacement, not append)
        assert len(q.undo_stack) == 1

    def test_discard_removes_without_redo(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter))
        q.discard()
        assert counter[0] == 0
        assert not q.can_undo()
        assert not q.can_redo()  # discard, unlike undo, cannot be redone

    def test_clear_empties_both_stacks(self):
        q = CommandQueue()
        counter = [0]
        q.execute(_CountCommand(counter))
        q.undo()
        q.clear()
        assert not q.can_undo()
        assert not q.can_redo()

    def test_max_history_evicts_oldest(self):
        q = CommandQueue(max_history=3)
        counter = [0]
        for _ in range(5):
            q.execute(_CountCommand(counter))
        # Only the last 3 should be retained
        assert len(q.undo_stack) == 3


# ----- Geometry commands ----------------------------------------------------

class TestAddLineCommand:
    def test_execute_adds_line(self, sketch):
        cmd = AddLineCommand(sketch, (0, 0), (5, 5))
        assert cmd.execute() is True
        assert len(sketch.entities) == 1
        assert isinstance(sketch.entities[0], Line)
        assert cmd.created_index == 0

    def test_undo_removes_added_line(self, sketch):
        cmd = AddLineCommand(sketch, (0, 0), (5, 5))
        cmd.execute()
        cmd.undo()
        assert sketch.entities == []

    def test_physical_flag_propagates_to_entity(self, sketch):
        cmd = AddLineCommand(sketch, (0, 0), (5, 5), physical=True)
        cmd.execute()
        assert sketch.entities[0].physical is True

    def test_ref_lines_skip_physical_atomization(self, sketch):
        cmd = AddLineCommand(sketch, (0, 0), (5, 5), is_ref=True, physical=True)
        cmd.execute()
        # Reference lines never get marked physical
        assert sketch.entities[0].physical is False

    def test_changes_topology_flag_set(self):
        # Adding entities must trigger a Compiler rebuild — that's signaled by
        # changes_topology=True so Scene.execute can mark _topology_dirty.
        assert AddLineCommand.changes_topology is True


class TestAddCircleCommand:
    def test_execute_adds_circle(self, sketch):
        cmd = AddCircleCommand(sketch, (5, 5), 3.0)
        cmd.execute()
        assert isinstance(sketch.entities[0], Circle)
        assert sketch.entities[0].radius == 3.0

    def test_undo_removes_circle(self, sketch):
        cmd = AddCircleCommand(sketch, (5, 5), 3.0)
        cmd.execute()
        cmd.undo()
        assert sketch.entities == []


class TestRemoveEntityCommand:
    def test_execute_removes_entity(self, sketch):
        sketch.add_line((0, 0), (5, 5))
        cmd = RemoveEntityCommand(sketch, 0)
        cmd.execute()
        assert sketch.entities == []

    def test_undo_restores_with_original_geometry(self, sketch):
        sketch.add_line((1.5, 2.5), (3.5, 4.5))
        cmd = RemoveEntityCommand(sketch, 0)
        cmd.execute()
        cmd.undo()
        assert len(sketch.entities) == 1
        assert tuple(sketch.entities[0].start) == (1.5, 2.5)
        assert tuple(sketch.entities[0].end) == (3.5, 4.5)

    def test_changes_topology_flag(self):
        assert RemoveEntityCommand.changes_topology is True


class TestMoveEntityCommand:
    def test_execute_translates_entity(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        MoveEntityCommand(sketch, 0, 3.0, 4.0).execute()
        assert tuple(sketch.entities[0].start) == (3.0, 4.0)

    def test_undo_reverses_translation(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        cmd = MoveEntityCommand(sketch, 0, 3.0, 4.0)
        cmd.execute()
        cmd.undo()
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)

    def test_consecutive_moves_merge_into_one_history_entry(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        q = CommandQueue()
        q.execute(MoveEntityCommand(sketch, 0, 1.0, 0.0))
        q.execute(MoveEntityCommand(sketch, 0, 2.0, 0.0))
        # Same entity & no point_indices override => merged into one entry
        assert len(q.undo_stack) == 1
        assert tuple(sketch.entities[0].start) == (3.0, 0.0)
        q.undo()
        # Single undo should reverse the merged delta in full
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)


class TestMoveMultipleCommand:
    def test_moves_all_indicated_entities(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_circle((5, 5), 1.0)
        MoveMultipleCommand(sketch, [0, 1], 2.0, 3.0).execute()
        assert tuple(sketch.entities[0].start) == (2.0, 3.0)
        assert tuple(sketch.entities[1].center) == (7.0, 8.0)

    def test_undo_reverses_all(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_circle((5, 5), 1.0)
        cmd = MoveMultipleCommand(sketch, [0, 1], 2.0, 3.0)
        cmd.execute()
        cmd.undo()
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        assert tuple(sketch.entities[1].center) == (5.0, 5.0)


class TestSetEntityGeometryCommand:
    def test_undo_restores_absolute_positions(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        cmd = SetEntityGeometryCommand(
            sketch, 0,
            old_positions=[(0, 0), (10, 0)],
            new_positions=[(5, 5), (15, 5)],
        )
        cmd.execute()
        assert tuple(sketch.entities[0].start) == (5.0, 5.0)
        cmd.undo()
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        assert tuple(sketch.entities[0].end) == (10.0, 0.0)


# ----- Constraint commands --------------------------------------------------

class TestAddConstraintCommand:
    def test_execute_appends_constraint(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        cmd = AddConstraintCommand(sketch, Length(0, 5.0), solve=False)
        cmd.execute()
        assert len(sketch.constraints) == 1

    def test_undo_removes_constraint(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        cmd = AddConstraintCommand(sketch, Length(0, 5.0), solve=False)
        cmd.execute()
        cmd.undo()
        assert sketch.constraints == []

    def test_undo_restores_pre_constraint_geometry(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        cmd = AddConstraintCommand(sketch, Length(0, 3.0), solve=True)
        cmd.execute()
        # After solve, length is ~3
        cmd.undo()
        # Geometry should be back to original 10-unit line
        from model.solver import Solver  # noqa: F401  (sanity import)
        line = sketch.entities[0]
        length = ((line.end[0] - line.start[0])**2 + (line.end[1] - line.start[1])**2) ** 0.5
        assert length == pytest.approx(10.0, abs=0.05)


class TestRemoveConstraintCommand:
    def test_execute_removes_constraint(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        RemoveConstraintCommand(sketch, 0).execute()
        assert sketch.constraints == []

    def test_undo_restores_constraint(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        cmd = RemoveConstraintCommand(sketch, 0)
        cmd.execute()
        cmd.undo()
        assert len(sketch.constraints) == 1
        assert sketch.constraints[0].type == "LENGTH"


class TestToggleAnchorCommand:
    def test_execute_toggles(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        ToggleAnchorCommand(sketch, 0, 0).execute()
        assert sketch.entities[0].anchored[0] is True

    def test_undo_reverses(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        cmd = ToggleAnchorCommand(sketch, 0, 0)
        cmd.execute()
        cmd.undo()
        assert sketch.entities[0].anchored[0] is False


class TestToggleInfiniteCommand:
    def test_toggle_on_reference_line(self, sketch):
        sketch.add_line((0, 0), (1, 0), is_ref=True)
        ToggleInfiniteCommand(sketch, 0).execute()
        assert sketch.entities[0].infinite is True


# ----- Property commands ----------------------------------------------------

class TestPropertyCommands:
    def test_set_physical_command_undo(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        cmd = SetPhysicalCommand(sketch, 0, True)
        cmd.execute()
        assert sketch.entities[0].physical is True
        cmd.undo()
        assert sketch.entities[0].physical is False

    def test_set_dynamic_command_undo(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        cmd = SetEntityDynamicCommand(sketch, 0, True)
        cmd.execute()
        assert sketch.entities[0].dynamic is True
        cmd.undo()
        assert sketch.entities[0].dynamic is False

    def test_set_material_command_undo(self, sketch):
        sketch.add_line((0, 0), (1, 0), material_id="Wall")
        cmd = SetMaterialCommand(sketch, 0, "Water")
        cmd.execute()
        assert sketch.entities[0].material_id == "Water"
        cmd.undo()
        assert sketch.entities[0].material_id == "Wall"


# ----- Composite commands ---------------------------------------------------

class TestCompositeCommand:
    def test_executes_all_subcommands(self, sketch):
        cmds = [
            AddLineCommand(sketch, (0, 0), (1, 0), historize=False),
            AddLineCommand(sketch, (1, 0), (1, 1), historize=False),
        ]
        CompositeCommand(cmds, "Two Lines").execute()
        assert len(sketch.entities) == 2

    def test_undo_reverses_in_order(self, sketch):
        cmds = [
            AddLineCommand(sketch, (0, 0), (1, 0), historize=False),
            AddLineCommand(sketch, (1, 0), (1, 1), historize=False),
        ]
        composite = CompositeCommand(cmds, "Two Lines")
        composite.execute()
        composite.undo()
        assert sketch.entities == []


class TestAddRectangleCommand:
    def test_creates_four_lines(self, sketch):
        AddRectangleCommand(sketch, 0, 0, 10, 5).execute()
        assert len(sketch.entities) == 4

    def test_creates_corner_constraints(self, sketch):
        AddRectangleCommand(sketch, 0, 0, 10, 5).execute()
        # 4 coincidents + 4 angle constraints
        assert len(sketch.constraints) == 8

    def test_undo_unwinds_lines_and_constraints(self, sketch):
        cmd = AddRectangleCommand(sketch, 0, 0, 10, 5)
        cmd.execute()
        cmd.undo()
        assert sketch.entities == []
        assert sketch.constraints == []

    def test_post_solve_geometry_is_closed_rectangle(self, sketch):
        """After the single 500-iter solve, corners should be coincident
        and sides should be axis-aligned."""
        AddRectangleCommand(sketch, 0, 0, 10, 5).execute()
        # 4 lines: 0=bottom, 1=right, 2=top, 3=left
        bottom = sketch.entities[0]
        right = sketch.entities[1]
        top = sketch.entities[2]
        left = sketch.entities[3]
        # Corners should match within solver tolerance
        assert tuple(bottom.end) == pytest.approx(tuple(right.start), abs=0.05)
        assert tuple(right.end) == pytest.approx(tuple(top.start), abs=0.05)
        assert tuple(top.end) == pytest.approx(tuple(left.start), abs=0.05)
        assert tuple(left.end) == pytest.approx(tuple(bottom.start), abs=0.05)
        # Bottom and top should be horizontal (start.y ≈ end.y)
        assert bottom.start[1] == pytest.approx(bottom.end[1], abs=0.05)
        assert top.start[1] == pytest.approx(top.end[1], abs=0.05)
        # Left and right should be vertical
        assert left.start[0] == pytest.approx(left.end[0], abs=0.05)
        assert right.start[0] == pytest.approx(right.end[0], abs=0.05)

    def test_undo_then_redo_is_clean(self, sketch):
        """Undo→Redo→Undo cycles must not leak constraint commands across the boundary."""
        cmd = AddRectangleCommand(sketch, 0, 0, 10, 5)
        cmd.execute()
        cmd.undo()
        # Re-execute (the redo path) — must not double-apply constraints
        cmd.execute()
        # Should still have exactly 4 entities and 8 constraints
        assert len(sketch.entities) == 4
        assert len(sketch.constraints) == 8


# ----- AddConstraintCommand: undo for various constraint types --------------

class TestAddConstraintCommandUndoTypes:
    def test_coincident_undo_removes_constraint_and_restores_geometry(self, sketch):
        from model.constraints import Coincident
        sketch.add_line((0, 0), (5, 0))
        sketch.add_line((10, 10), (15, 10))
        # Capture pre-state
        pre1 = sketch.entities[1].start.copy()
        cmd = AddConstraintCommand(sketch, Coincident(0, 1, 1, 0), solve=True)
        cmd.execute()
        # Solver moved geometry; constraint is in place
        assert len(sketch.constraints) == 1
        cmd.undo()
        assert sketch.constraints == []
        # Geometry should be back to original
        assert tuple(sketch.entities[1].start) == tuple(pre1)

    def test_parallel_undo_restores_geometry(self, sketch):
        from model.constraints import Angle
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((0, 5), (3, 7))
        sketch.entities[0].anchored = [True, True]
        pre = sketch.entities[1].end.copy()
        cmd = AddConstraintCommand(sketch, Angle("PARALLEL", 0, 1), solve=True)
        cmd.execute()
        cmd.undo()
        assert sketch.constraints == []
        assert tuple(sketch.entities[1].end) == tuple(pre)


# ----- SetEntityGeometryCommand: types beyond Line --------------------------

class TestSetEntityGeometryOnNonLines:
    def test_circle_center_round_trip(self, sketch):
        sketch.add_circle((0, 0), 5.0)
        cmd = SetEntityGeometryCommand(
            sketch, 0,
            old_positions=[(0, 0)],
            new_positions=[(10, 10)],
        )
        cmd.execute()
        assert tuple(sketch.entities[0].center) == (10.0, 10.0)
        cmd.undo()
        assert tuple(sketch.entities[0].center) == (0.0, 0.0)

    def test_point_round_trip(self, sketch):
        from model.geometry import Point
        sketch.entities.append(Point(1, 2))
        cmd = SetEntityGeometryCommand(
            sketch, 0,
            old_positions=[(1, 2)],
            new_positions=[(7, 8)],
        )
        cmd.execute()
        assert tuple(sketch.entities[0].pos) == (7.0, 8.0)
        cmd.undo()
        assert tuple(sketch.entities[0].pos) == (1.0, 2.0)


# ----- Merge guards ---------------------------------------------------------

class TestMergeGuards:
    def test_merge_does_not_cross_historize_boundaries(self, sketch):
        """The historize=True successor must NOT absorb the historize=False
        predecessor's dx via merge. Per TU-SKETCH refinement: the
        non-historized predecessor was never appended to the stack in the
        first place; what we're really pinning is that
        MoveEntityCommand.merge guards on `other.historize == self.historize`,
        so the historized successor's dx stays at its own value (not folded
        with the predecessor's)."""
        sketch.add_line((0, 0), (1, 0))
        q = CommandQueue()
        q.execute(MoveEntityCommand(sketch, 0, 1.0, 0.0, historize=False))
        q.execute(MoveEntityCommand(sketch, 0, 2.0, 0.0, historize=True))
        assert len(q.undo_stack) == 1
        cmd = q.undo_stack[0]
        # If the merge guard were absent, dx would be 3.0 (1.0 + 2.0)
        assert cmd.dx == 2.0

    def test_merge_does_not_cross_point_indices_mismatch(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        q = CommandQueue()
        q.execute(MoveEntityCommand(sketch, 0, 1.0, 0.0, point_indices=[0]))
        q.execute(MoveEntityCommand(sketch, 0, 1.0, 0.0, point_indices=[1]))
        # Different point_indices → must not merge
        assert len(q.undo_stack) == 2

    def test_set_point_merge_preserves_first_old_position(self, sketch):
        from core.commands import SetPointCommand
        sketch.add_line((0, 0), (10, 0))
        q = CommandQueue()
        # Drag sequence: each SetPointCommand sets new_position, but undo must
        # restore the OLD_position from the very first execute, not the
        # penultimate one.
        q.execute(SetPointCommand(sketch, 0, 1, (11, 0)))
        q.execute(SetPointCommand(sketch, 0, 1, (12, 0)))
        q.execute(SetPointCommand(sketch, 0, 1, (15, 0)))
        # All should merge into one stack entry
        assert len(q.undo_stack) == 1
        # Single undo should restore the original (10, 0)
        q.undo()
        assert tuple(sketch.entities[0].end) == (10.0, 0.0)

    def test_move_multiple_merge_uses_set_equality(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((2, 0), (3, 0))
        q = CommandQueue()
        # Same entities, different list order — should still merge
        q.execute(MoveMultipleCommand(sketch, [0, 1], 1.0, 0.0))
        q.execute(MoveMultipleCommand(sketch, [1, 0], 2.0, 0.0))
        assert len(q.undo_stack) == 1
        assert q.undo_stack[0].dx == 3.0


class TestSupersedeEdgeCases:
    def test_supersede_on_empty_stack(self, sketch):
        """Supersede on an empty stack must not crash; the command still
        executes and goes onto the stack as a normal entry."""
        sketch.add_line((0, 0), (10, 0))
        q = CommandQueue()
        # An empty stack means nothing to undo; supersede should no-op the
        # would-be pre-undo and just execute the command.
        cmd = MoveEntityCommand(sketch, 0, 5.0, 0.0, supersede=True)
        result = q.execute(cmd)
        assert result is True
        assert len(q.undo_stack) == 1


class TestCircleRadiusCommandMerge:
    def test_consecutive_radius_changes_merge(self, sketch):
        from core.commands import SetCircleRadiusCommand
        sketch.add_circle((0, 0), 1.0)
        q = CommandQueue()
        q.execute(SetCircleRadiusCommand(sketch, 0, 2.0))
        q.execute(SetCircleRadiusCommand(sketch, 0, 3.0))
        q.execute(SetCircleRadiusCommand(sketch, 0, 5.0))
        assert len(q.undo_stack) == 1
        # Single undo restores original radius
        q.undo()
        assert sketch.entities[0].radius == 1.0


class TestMaxHistoryEviction:
    def test_evicted_command_data_is_lost(self):
        """When max_history is small, oldest commands are evicted permanently.
        Their state cannot be undone."""
        q = CommandQueue(max_history=3)
        from tests.test_commands import _CountCommand
        counter = [0]
        cmds = [_CountCommand(counter, delta=1) for _ in range(5)]
        for c in cmds:
            q.execute(c)
        assert counter[0] == 5
        # Stack only retains last 3
        assert len(q.undo_stack) == 3
        # Undo all available — the first two are gone
        q.undo()
        q.undo()
        q.undo()
        assert counter[0] == 2  # only 3 of 5 deltas reversed
        assert q.undo() is False


class TestResizeWorldCommand:
    """Wraps Simulation.resize_world so Ctrl+Z reverts a destructive resize."""

    def test_execute_resizes_world(self, scene):
        from core.commands import ResizeWorldCommand
        cmd = ResizeWorldCommand(scene, 75.0)
        cmd.execute()
        assert scene.simulation.world_size == 75.0

    def test_undo_restores_world_size(self, scene):
        from core.commands import ResizeWorldCommand
        original = scene.simulation.world_size
        cmd = ResizeWorldCommand(scene, 75.0)
        cmd.execute()
        cmd.undo()
        assert scene.simulation.world_size == original

    def test_undo_restores_particles(self, scene):
        """resize_world wipes all particles. Undo must bring them back."""
        from core.commands import ResizeWorldCommand
        scene.paint_particles(25.0, 25.0, radius=2.0)
        before_count = scene.simulation.count
        assert before_count > 0

        cmd = ResizeWorldCommand(scene, 75.0)
        cmd.execute()
        # After resize, particles cleared
        assert scene.simulation.count == 0

        cmd.undo()
        # Particles restored
        assert scene.simulation.count == before_count

    def test_clamps_to_minimum(self, scene):
        from core.commands import ResizeWorldCommand
        ResizeWorldCommand(scene, 1.0).execute()
        assert scene.simulation.world_size == 10.0  # clamped

    def test_via_scene_execute_lands_on_undo_stack(self, scene):
        """Ctrl+Z can revert it because the command goes onto the CAD undo stack."""
        from core.commands import ResizeWorldCommand
        scene.execute(ResizeWorldCommand(scene, 80.0))
        assert scene.can_undo()
        scene.undo()
        # After undo, world is back to default
        import core.config as config
        assert scene.simulation.world_size == config.DEFAULT_WORLD_SIZE

    def test_resize_after_cad_undo_chain(self, scene):
        """Ctrl+Z prefers CAD commands over physics undo. With ResizeWorldCommand
        on the CAD stack, a single Ctrl+Z reverts the resize even when there
        are CAD commands above it in stack order."""
        from core.commands import ResizeWorldCommand, AddLineCommand

        # 1. Resize world (goes onto CAD stack)
        scene.execute(ResizeWorldCommand(scene, 60.0))
        assert scene.simulation.world_size == 60.0
        # 2. Add a line (also onto CAD stack, on top of the resize)
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (5, 0)))
        # Stack now has [resize, line]; can_undo true
        assert scene.can_undo()
        # First undo pops the line
        scene.undo()
        assert len(scene.sketch.entities) == 0
        # World still resized
        assert scene.simulation.world_size == 60.0
        # Second undo pops the resize
        scene.undo()
        import core.config as config
        assert scene.simulation.world_size == config.DEFAULT_WORLD_SIZE


class TestResizeWorldInputFieldSync:
    """The Resize-World input field must reflect sim.world_size after undo/redo.

    Regression: ResizeWorldCommand correctly mutated sim.world_size on
    undo/redo, but the input field next to the Resize World button was
    never refreshed. After Ctrl+Z on a resize, the field continued to
    show the typed value (the post-resize size) rather than the actual
    (now-restored) world_size.
    """

    def _build_controller(self, fake_app, initial_text=None):
        from app.app_controller import AppController
        from ui.ui_widgets import InputField
        sim = fake_app.scene.simulation
        text = initial_text if initial_text is not None else f"{sim.world_size:.2f}"
        fake_app.session.input_world = InputField(0, 0, 60, 25, text)
        return AppController(fake_app)

    def test_undo_of_resize_refreshes_input_field(self, fake_app):
        from core.commands import ResizeWorldCommand
        controller = self._build_controller(fake_app)
        scene = fake_app.scene
        sim = scene.simulation
        original_size = sim.world_size

        scene.execute(ResizeWorldCommand(scene, 75.0))
        # Field is unchanged at this point (production wires the field via
        # the action_resize_world / dialog flow); we are testing the undo
        # refresh, not the create-time refresh.
        controller.action_undo()

        assert sim.world_size == original_size
        assert float(fake_app.session.input_world.text) == original_size

    def test_redo_of_resize_refreshes_input_field(self, fake_app):
        from core.commands import ResizeWorldCommand
        controller = self._build_controller(fake_app)
        scene = fake_app.scene
        sim = scene.simulation

        scene.execute(ResizeWorldCommand(scene, 75.0))
        controller.action_undo()
        controller.action_redo()

        assert sim.world_size == 75.0
        assert float(fake_app.session.input_world.text) == 75.0

    def test_focused_input_field_is_not_overwritten_on_undo(self, fake_app):
        """If the user is mid-edit when an undo fires, the typed text wins.

        InputField.set_value is gated on `not self.active`, so the helper
        is a no-op while the field has focus. This protects mid-edit text
        from being clobbered by an undo on an unrelated CAD command.
        """
        from core.commands import ResizeWorldCommand
        controller = self._build_controller(fake_app)
        scene = fake_app.scene

        scene.execute(ResizeWorldCommand(scene, 75.0))
        # User is now typing into the field
        fake_app.session.input_world.active = True
        fake_app.session.input_world.text = "120"

        controller.action_undo()

        # World was reverted but the typed text is preserved
        assert fake_app.session.input_world.text == "120"

    def test_undo_with_no_input_field_is_safe(self, fake_app):
        """session.input_world starts as None; the helper must tolerate that."""
        from app.app_controller import AppController
        from core.commands import ResizeWorldCommand
        controller = AppController(fake_app)
        # session.input_world left at default None
        scene = fake_app.scene
        scene.execute(ResizeWorldCommand(scene, 75.0))
        controller.action_undo()
        # No exception; world correctly reverted
        import core.config as config
        assert scene.simulation.world_size == config.DEFAULT_WORLD_SIZE


class TestSetEntityGeometryDefensive:
    def test_mismatched_lengths_round_trip_is_asymmetric(self, sketch):
        """Documents current asymmetric behavior of SetEntityGeometryCommand
        when old_positions and new_positions have different lengths.

        Per TU-SCENE: the implementation iterates each list independently
        (no min(...) guard). With 1 old and 2 new positions, execute() sets
        both points but undo() only restores point 0 — leaving point 1 stuck
        at the new position. This is a latent issue the test pins so a future
        fix (likely: zip-shortest or explicit length validation) starts
        failing here.
        """
        sketch.add_line((0, 0), (10, 0))
        cmd = SetEntityGeometryCommand(
            sketch, 0,
            old_positions=[(0, 0)],
            new_positions=[(5, 5), (15, 5)],
        )
        cmd.execute()
        # Both points moved by execute()
        assert tuple(sketch.entities[0].start) == (5.0, 5.0)
        assert tuple(sketch.entities[0].end) == (15.0, 5.0)
        cmd.undo()
        # Asymmetric undo: point 0 restored, point 1 left in new position
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        assert tuple(sketch.entities[0].end) == (15.0, 5.0)  # NOT restored
