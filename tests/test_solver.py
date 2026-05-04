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

    def test_empty_constraint_list_is_noop_legacy(self, sketch):
        sketch.use_numba = False
        sketch.solve()  # No entities, no constraints — must not raise

    def test_empty_constraint_list_is_noop_numba(self, sketch):
        sketch.use_numba = True
        sketch.solve()

    @pytest.mark.slow
    def test_numba_legacy_parity_combined_network(self, sketch):
        """Both backends should converge to within tolerance on the same network."""
        # Build identical sketch geometry twice
        from model.sketch import Sketch
        from model.constraints import Angle

        def build_and_solve(use_numba):
            s = Sketch()
            s.use_numba = use_numba
            s.add_line((0, 0), (1, 0))
            s.add_line((1, 0), (1, 1))
            s.entities[0].anchored = [True, True]
            s.add_constraint_object(Angle("PERPENDICULAR", 0, 1), solve=False)
            s.add_constraint_object(Length(1, 5.0))
            return s.entities[1].end.copy()

        legacy = build_and_solve(False)
        numba = build_and_solve(True)
        # Positions should match within solver tolerance
        assert abs(legacy[0] - numba[0]) < 0.1
        assert abs(legacy[1] - numba[1]) < 0.1


# ----- User Servo (interaction_data) ----------------------------------------

class TestUserServo:
    def test_endpoint_drag_with_anchored_opposite_rotates_about_anchor(self, sketch):
        """Endpoint drag with the other end anchored produces rotation, not pure translation.
        TU-SKETCH refinement: assert the rotational character explicitly — anchored start
        unchanged AND the angle between start→end has shifted from horizontal."""
        import math
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].anchored = [True, False]
        sketch.interaction_data = {
            "entity_idx": 0,
            "point_idx": 1,
            "handle_t": None,
            "target": (5.0, 5.0),
        }
        sketch.solve()
        # Anchor invariant
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        # Rotation invariant: the line is no longer horizontal
        end = sketch.entities[0].end
        ang = abs(math.atan2(end[1] - 0.0, end[0] - 0.0))
        assert ang > 0.1  # measurably rotated from the original 0-radian heading

    def test_body_drag_with_handle_t_translates_line(self, sketch):
        """Body drag at handle_t=0.5 translates the line so the midpoint lands at the target."""
        sketch.add_line((0, 0), (10, 0))
        sketch.interaction_data = {
            "entity_idx": 0,
            "point_idx": None,
            "handle_t": 0.5,
            "target": (5.0, 5.0),
        }
        sketch.solve()
        # Midpoint should now be ~ (5, 5)
        line = sketch.entities[0]
        mid = (line.start + line.end) / 2.0
        assert mid[0] == pytest.approx(5.0, abs=0.5)
        assert mid[1] == pytest.approx(5.0, abs=0.5)

    def test_clearing_interaction_data_stops_servo(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.interaction_data = {
            "entity_idx": 0, "point_idx": 1, "handle_t": None, "target": (5, 5),
        }
        sketch.solve()
        end_after_drag = sketch.entities[0].end.copy()
        # Clear servo and solve again with no constraints
        sketch.interaction_data = None
        sketch.solve()
        # Tolerance-based equality survives future damping/refactors
        assert sketch.entities[0].end[0] == pytest.approx(end_after_drag[0], abs=1e-9)
        assert sketch.entities[0].end[1] == pytest.approx(end_after_drag[1], abs=1e-9)


# ----- Combined network convergence -----------------------------------------

class TestCombinedConstraints:
    def test_length_plus_parallel_converges_to_both(self, sketch):
        """The Solver's unary/binary phase split must converge a combined network."""
        from model.constraints import Angle
        sketch.add_line((0, 0), (10, 0))     # reference, anchored
        sketch.add_line((0, 5), (3, 7))      # arbitrary
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(Angle("PARALLEL", 0, 1), solve=False)
        sketch.add_constraint_object(Length(1, 4.0))

        line1 = sketch.entities[1]
        # Should be ~ horizontal (parallel to line 0) AND ~4 units long
        dy = line1.end[1] - line1.start[1]
        assert abs(dy) < 0.2
        assert _line_length(line1) == pytest.approx(4.0, abs=0.1)


# ----- Coincident point-on-entity (second factory rule) --------------------

class TestCoincidentPointOnEntity:
    def test_point_pulled_to_circle_center(self, sketch):
        """For Circles, COINCIDENT-with-entity drives the point to the circle's
        center (the circle's "anchor"), not to the circumference. Documents
        current solver behavior — the second `_solve_coincident_pt_ent` rule.

        NOTE: TU-SKETCH flagged this semantics as surprising — coincident-with-circle
        pulling to center rather than nearest-point-on-circumference. If this is
        intended (treating "the entity itself" as its anchor point), the test pins
        it; if it's actually a latent bug, the future fix will need to flip this
        test (and TU-SKETCH would xfail it then). Keeping as a behavior pin for now.
        """
        from model.geometry import Point
        from model.constraints import Coincident
        sketch.add_circle((5, 5), 3.0)
        sketch.entities[0].anchored = [True]
        sketch.entities.append(Point(0, 0))  # Point is index 1
        sketch.add_constraint_object(Coincident(1, 0, 0, -1))
        p = sketch.entities[1]
        # Lands at center
        assert tuple(p.pos) == pytest.approx((5.0, 5.0), abs=0.2)

    def test_point_pulled_onto_line(self, sketch):
        from model.geometry import Point
        from model.constraints import Coincident
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].anchored = [True, True]
        sketch.entities.append(Point(5, 5))
        sketch.add_constraint_object(Coincident(1, 0, 0, -1))
        # Point should land on the horizontal line (y ~ 0)
        assert sketch.entities[1].pos[1] == pytest.approx(0.0, abs=0.2)


# ----- Driver update --------------------------------------------------------

class TestDriverUpdate:
    def test_sin_driver_modulates_length_value(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        c = sketch.constraints[0]
        c.driver = {"type": "sin", "amp": 2.0, "freq": 1.0, "phase": 0.0}
        c.base_value = 5.0
        c.base_time = 0.0
        # At t=0.25 with freq=1, sin(2*pi*0.25) = 1.0, so value = 5 + 2*1 = 7
        sketch.update_drivers(0.25)
        assert c.value == pytest.approx(7.0, abs=0.01)

    def test_lin_driver_advances_value_linearly(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        c = sketch.constraints[0]
        c.driver = {"type": "lin", "rate": 2.0}
        c.base_value = 5.0
        c.base_time = 0.0
        sketch.update_drivers(3.0)  # 3s * 2 = 6 added
        assert c.value == pytest.approx(11.0, abs=0.01)


# ----- FixedAngle non-orthogonal --------------------------------------------

class TestFixedAngleNonOrthogonal:
    def test_thirty_degree_angle(self, sketch):
        import math
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((0, 0), (5, 0))
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(FixedAngle(0, 1, 30.0))
        l1 = sketch.entities[1]
        v1 = l1.end - l1.start
        # Angle from horizontal should be ~30deg (or -30 / 150 / -150 — sign convention)
        ang = math.degrees(math.atan2(v1[1], v1[0]))
        # Allow either +30 or -30 depending on sign convention
        candidates = [30.0, -30.0, 150.0, -150.0]
        assert min(abs(ang - c) for c in candidates) < 5.0

    def test_135_degree_angle(self, sketch):
        import math
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((0, 0), (5, 0))
        sketch.entities[0].anchored = [True, True]
        sketch.add_constraint_object(FixedAngle(0, 1, 135.0))
        l1 = sketch.entities[1]
        v1 = l1.end - l1.start
        ang = math.degrees(math.atan2(v1[1], v1[0]))
        candidates = [135.0, -135.0, 45.0, -45.0]
        assert min(abs(ang - c) for c in candidates) < 5.0
