"""Tests for model/solver.py — constraint satisfaction.

The PBD solver is iterative and approximate; tests use generous tolerances
appropriate to the default 500-iteration solve budget.
"""

import math
import pytest

from model.constraints import (
    Length, Radius, Coincident, Angle, EqualLength, FixedAngle, Midpoint
)


def _line_length(line):
    dx = line.end[0] - line.start[0]
    dy = line.end[1] - line.start[1]
    return math.hypot(dx, dy)


class TestLengthConstraint:
    def test_length_constraint_drives_line_to_target(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_constraint_object(Length(0, 5.0))
        # PBD with 500 iterations should easily reach a length-only target
        assert _line_length(sketch.entities[0]) == pytest.approx(5.0, abs=0.05)

    def test_length_constraint_respects_anchor(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.entities[0].anchored = [True, False]
        sketch.add_constraint_object(Length(0, 3.0))
        # Anchored start should not move; only end should adjust
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        assert _line_length(sketch.entities[0]) == pytest.approx(3.0, abs=0.05)


class TestRadiusConstraint:
    def test_radius_constraint_drives_circle_to_target(self, sketch):
        sketch.add_circle((0, 0), 1.0)
        sketch.add_constraint_object(Radius(0, 4.0))
        assert sketch.entities[0].radius == pytest.approx(4.0, abs=0.05)


class TestCoincidentConstraint:
    def test_coincident_brings_endpoints_together(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((5, 5), (6, 5))
        # Constrain end of line 0 to start of line 1
        sketch.add_constraint_object(Coincident(0, 1, 1, 0))
        end0 = sketch.entities[0].end
        start1 = sketch.entities[1].start
        assert tuple(end0) == pytest.approx(tuple(start1), abs=0.05)


class TestAngleConstraints:
    def test_horizontal_constraint_levels_line(self, sketch):
        sketch.add_line((0, 0), (3, 4))
        sketch.add_constraint_object(Angle("HORIZONTAL", 0))
        line = sketch.entities[0]
        # Start.y and end.y should match (line is horizontal)
        assert line.end[1] == pytest.approx(line.start[1], abs=0.05)

    def test_vertical_constraint_aligns_line(self, sketch):
        sketch.add_line((0, 0), (3, 4))
        sketch.add_constraint_object(Angle("VERTICAL", 0))
        line = sketch.entities[0]
        # Start.x and end.x should match (line is vertical)
        assert line.end[0] == pytest.approx(line.start[0], abs=0.05)

    def test_parallel_constraint_aligns_directions(self, sketch):
        sketch.add_line((0, 0), (10, 0))   # horizontal
        sketch.add_line((0, 5), (5, 7))    # arbitrary slope
        sketch.entities[0].anchored = [True, True]  # Pin reference
        sketch.add_constraint_object(Angle("PARALLEL", 0, 1))
        # Compute slope of line 1 — should match line 0 (horizontal => dy ~ 0)
        l1 = sketch.entities[1]
        assert (l1.end[1] - l1.start[1]) == pytest.approx(0.0, abs=0.1)

    def test_perpendicular_constraint_orthogonal_directions(self, sketch):
        sketch.add_line((0, 0), (10, 0))   # horizontal
        sketch.add_line((1, 0), (5, 5))    # arbitrary
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(Angle("PERPENDICULAR", 0, 1))
        l1 = sketch.entities[1]
        # Perpendicular to horizontal => line 1's dx should be ~0
        assert (l1.end[0] - l1.start[0]) == pytest.approx(0.0, abs=0.1)


class TestEqualLength:
    def test_equal_length_makes_two_lines_match(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((0, 5), (3, 5))
        sketch.entities[0].anchored = [True, True]  # Pin reference at length 10
        sketch.add_constraint_object(EqualLength(0, 1))
        assert _line_length(sketch.entities[1]) == pytest.approx(10.0, abs=0.1)


class TestFixedAngle:
    def test_fixed_angle_holds_specified_degree(self, sketch):
        sketch.add_line((0, 0), (10, 0))   # horizontal
        sketch.add_line((0, 0), (5, 0))    # initially also horizontal
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(FixedAngle(0, 1, 90.0))
        # Line 1 should be ~vertical relative to line 0 — its dx should ~ 0
        l1 = sketch.entities[1]
        dx = l1.end[0] - l1.start[0]
        dy = l1.end[1] - l1.start[1]
        assert abs(dx) < abs(dy)  # closer to vertical than horizontal


class TestMidpoint:
    def test_midpoint_constrains_point_to_line_center(self, sketch):
        from model.geometry import Point
        sketch.add_line((0, 0), (10, 0))
        sketch.entities.append(Point(3.0, 3.0))  # Point is index 1
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(Midpoint(1, 0, 0))
        # Point should land at the midpoint of line 0 (5, 0)
        p = sketch.entities[1]
        assert tuple(p.pos) == pytest.approx((5.0, 0.0), abs=0.1)


class TestSolverBackends:
    def test_legacy_backend_runs(self, sketch):
        sketch.use_numba = False
        sketch.add_line((0, 0), (1, 0))
        sketch.add_constraint_object(Length(0, 3.0))
        assert _line_length(sketch.entities[0]) == pytest.approx(3.0, abs=0.05)

    @pytest.mark.slow
    def test_numba_backend_runs(self, sketch):
        # Triggers Numba compilation on first use
        sketch.use_numba = True
        sketch.add_line((0, 0), (1, 0))
        sketch.add_constraint_object(Length(0, 3.0))
        assert _line_length(sketch.entities[0]) == pytest.approx(3.0, abs=0.05)
