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
