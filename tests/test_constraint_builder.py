"""Tests for core/constraint_builder.py — pending-constraint state machine."""

import pytest

from core.constraint_builder import ConstraintBuilder


class TestInitialState:
    def test_starts_idle(self):
        b = ConstraintBuilder()
        assert b.pending_type is None
        assert b.is_pending is False
        assert b.target_walls == []
        assert b.target_points == []
        assert b.snap_target is None


class TestStartAndAdd:
    def test_start_initializes_state(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        assert b.pending_type == "PARALLEL"
        assert b.is_pending is True

    def test_start_with_initial_targets(self):
        b = ConstraintBuilder()
        b.start("COINCIDENT", initial_walls=[1], initial_points=[(0, 0)])
        assert b.target_walls == [1]
        assert b.target_points == [(0, 0)]

    def test_add_wall_skips_duplicates(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        assert b.add_wall(0) is True
        assert b.add_wall(0) is False
        assert b.target_walls == [0]

    def test_add_point_skips_duplicates(self):
        b = ConstraintBuilder()
        b.start("COINCIDENT")
        assert b.add_point(0, 0) is True
        assert b.add_point(0, 0) is False
        assert b.target_points == [(0, 0)]


class TestReset:
    def test_reset_clears_all_state(self):
        """reset() must clear pending_type, target_walls, target_points, snap_target —
        the contract underpinning ctx.clear_constraint_ui()."""
        b = ConstraintBuilder()
        b.start("COINCIDENT", initial_walls=[1], initial_points=[(0, 0)])
        b.snap_target = (2, 1)

        b.reset()

        assert b.pending_type is None
        assert b.is_pending is False
        assert b.target_walls == []
        assert b.target_points == []
        assert b.snap_target is None


class TestCheckReady:
    def test_returns_false_with_no_pending_type(self):
        b = ConstraintBuilder()
        assert b.check_ready() is False

    def test_returns_false_when_targets_insufficient(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        assert b.check_ready() is False

    def test_returns_true_when_targets_satisfy_rule(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        b.add_wall(0)
        b.add_wall(1)
        assert b.check_ready() is True


class TestQueries:
    def test_is_wall_targeted(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        b.add_wall(3)
        assert b.is_wall_targeted(3) is True
        assert b.is_wall_targeted(99) is False

    def test_is_point_targeted(self):
        b = ConstraintBuilder()
        b.start("COINCIDENT")
        b.add_point(0, 1)
        assert b.is_point_targeted(0, 1) is True
        assert b.is_point_targeted(0, 0) is False


class TestMultiAndBinaryFlags:
    def test_is_multi_apply_for_horizontal(self):
        b = ConstraintBuilder()
        b.start("HORIZONTAL")
        # HORIZONTAL has multi=True in CONSTRAINT_DEFS
        assert b.is_multi_apply() is True

    def test_is_binary_multi_for_parallel(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        assert b.is_binary_multi() is True

    def test_is_binary_multi_false_for_horizontal(self):
        b = ConstraintBuilder()
        b.start("HORIZONTAL")
        assert b.is_binary_multi() is False


class TestStatusMessage:
    def test_empty_when_idle(self):
        b = ConstraintBuilder()
        assert b.get_status_message() == ""

    def test_includes_constraint_type_when_pending(self):
        b = ConstraintBuilder()
        b.start("PARALLEL")
        msg = b.get_status_message()
        assert "PARALLEL" in msg


class TestTryBuildCommand:
    def test_returns_none_with_no_pending_type(self, sketch):
        b = ConstraintBuilder()
        assert b.try_build_command(sketch) is None

    def test_returns_command_when_targets_match(self, sketch):
        sketch.add_line((0, 0), (5, 0))
        sketch.add_line((0, 5), (5, 5))
        b = ConstraintBuilder()
        b.start("PARALLEL")
        b.add_wall(0)
        b.add_wall(1)
        cmd = b.try_build_command(sketch)
        assert cmd is not None

    def test_auto_trims_over_selected_targets(self, sketch):
        """If user selected too many walls/points, builder auto-trims."""
        sketch.add_line((0, 0), (5, 0))
        sketch.add_line((0, 5), (5, 5))
        sketch.add_line((0, 10), (5, 10))
        b = ConstraintBuilder()
        b.start("PERPENDICULAR")  # needs exactly 2
        b.add_wall(0)
        b.add_wall(1)
        b.add_wall(2)  # one too many
        cmd = b.try_build_command(sketch)
        assert cmd is not None
