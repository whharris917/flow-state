"""Tests for model/geometry.py — Line, Circle, Point primitives."""

import math
import numpy as np
import pytest

from model.geometry import Line, Circle, Point
from model.protocols import EntityType


# ----- Construction & basic shape -------------------------------------------

class TestLineBasics:
    def test_construction_stores_endpoints(self):
        line = Line((0.0, 0.0), (3.0, 4.0))
        assert tuple(line.start) == (0.0, 0.0)
        assert tuple(line.end) == (3.0, 4.0)

    def test_length_is_euclidean(self):
        line = Line((0.0, 0.0), (3.0, 4.0))
        assert line.length() == pytest.approx(5.0)

    def test_default_anchored_state_is_two_false(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        assert line.anchored == [False, False]

    def test_default_material_id_is_wall(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        assert line.material_id == "Wall"

    def test_ref_flag_default_false(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        assert line.ref is False
        ref = Line((0.0, 0.0), (1.0, 0.0), is_ref=True)
        assert ref.ref is True

    def test_get_point_returns_endpoints(self):
        line = Line((1.0, 2.0), (3.0, 4.0))
        assert tuple(line.get_point(0)) == (1.0, 2.0)
        assert tuple(line.get_point(1)) == (3.0, 4.0)

    def test_set_point_updates_endpoint(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        line.set_point(1, np.array([5.0, 5.0]))
        assert tuple(line.end) == (5.0, 5.0)

    def test_center_of_mass_is_midpoint(self):
        line = Line((0.0, 0.0), (10.0, 0.0))
        com = line.get_center_of_mass()
        assert tuple(com) == (5.0, 0.0)

    def test_inertia_uses_uniform_rod_formula(self):
        line = Line((0.0, 0.0), (10.0, 0.0))
        line.dynamic = True
        line.mass = 12.0
        # I = (1/12) * m * L^2 * INERTIA_STABILITY_FACTOR (=5.0 by default)
        # = (1/12) * 12 * 100 * 5 = 500
        assert line.inertia == pytest.approx(500.0)

    def test_move_translates_both_endpoints(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        line.move(2.0, 3.0)
        assert tuple(line.start) == (2.0, 3.0)
        assert tuple(line.end) == (3.0, 3.0)

    def test_move_respects_anchored_endpoints(self):
        line = Line((0.0, 0.0), (1.0, 0.0))
        line.anchored = [True, False]
        line.move(5.0, 0.0)
        assert tuple(line.start) == (0.0, 0.0)  # anchored
        assert tuple(line.end) == (6.0, 0.0)

    def test_distance_to_point_on_segment(self):
        line = Line((0.0, 0.0), (10.0, 0.0))
        assert line.distance_to(5.0, 3.0) == pytest.approx(3.0)

    def test_distance_to_point_clamped_outside_segment(self):
        line = Line((0.0, 0.0), (10.0, 0.0))
        # Past the end — distance is to the end point, not the infinite line
        assert line.distance_to(15.0, 0.0) == pytest.approx(5.0)

    def test_entity_type_is_line(self):
        assert Line((0, 0), (1, 0)).entity_type == EntityType.LINE


class TestCircleBasics:
    def test_construction_stores_center_and_radius(self):
        c = Circle((5.0, 5.0), 3.0)
        assert tuple(c.center) == (5.0, 5.0)
        assert c.radius == 3.0

    def test_default_anchored_is_single_false(self):
        c = Circle((0, 0), 1.0)
        assert c.anchored == [False]

    def test_distance_to_returns_distance_from_edge(self):
        c = Circle((0.0, 0.0), 5.0)
        assert c.distance_to(8.0, 0.0) == pytest.approx(3.0)
        assert c.distance_to(2.0, 0.0) == pytest.approx(3.0)

    def test_inertia_uses_disk_formula(self):
        c = Circle((0.0, 0.0), 2.0)
        c.dynamic = True
        c.mass = 10.0
        # I = (1/2) * m * r^2 * INERTIA_STABILITY_FACTOR (=5.0)
        # = 0.5 * 10 * 4 * 5 = 100
        assert c.inertia == pytest.approx(100.0)

    def test_move_translates_center(self):
        c = Circle((0.0, 0.0), 1.0)
        c.move(3.0, 4.0)
        assert tuple(c.center) == (3.0, 4.0)

    def test_anchored_circle_does_not_move(self):
        c = Circle((0.0, 0.0), 1.0)
        c.anchored = [True]
        c.move(3.0, 4.0)
        assert tuple(c.center) == (0.0, 0.0)

    def test_entity_type_is_circle(self):
        assert Circle((0, 0), 1.0).entity_type == EntityType.CIRCLE


class TestPointBasics:
    def test_construction_stores_position(self):
        p = Point(1.0, 2.0)
        assert tuple(p.pos) == (1.0, 2.0)

    def test_anchor_default_false(self):
        p = Point(0.0, 0.0)
        assert p.anchored is False

    def test_inv_mass_zero_when_anchored(self):
        p = Point(0.0, 0.0, anchored=True)
        assert p.get_inv_mass(0) == 0.0

    def test_handle_flag_defaults_false(self):
        p = Point(0.0, 0.0)
        assert p.is_handle is False

    def test_entity_type_is_point(self):
        assert Point(0, 0).entity_type == EntityType.POINT


# ----- Inverse mass / dynamic flag ------------------------------------------

class TestInverseMass:
    def test_anchored_endpoint_has_zero_inv_mass(self):
        line = Line((0, 0), (1, 0))
        line.anchored = [True, False]
        assert line.get_inv_mass(0) == 0.0
        assert line.get_inv_mass(1) == 1.0

    def test_dynamic_entity_uses_actual_inv_mass(self):
        line = Line((0, 0), (1, 0))
        line.dynamic = True
        line.mass = 4.0
        # When dynamic, get_inv_mass returns 1/mass for unanchored points
        assert line.get_inv_mass(1) == pytest.approx(0.25)

    def test_static_entity_returns_unit_inv_mass(self):
        line = Line((0, 0), (1, 0))
        # Default: dynamic=False, anchored=False -> inv_mass=1.0
        assert line.get_inv_mass(0) == 1.0


# ----- Serialization round-trip ---------------------------------------------

class TestSerialization:
    def test_line_dict_roundtrip_preserves_basic_state(self):
        line = Line((1.0, 2.0), (3.0, 4.0), is_ref=True, material_id="Water")
        line.anchored = [True, False]
        d = line.to_dict()
        restored = Line.from_dict(d)
        assert tuple(restored.start) == (1.0, 2.0)
        assert tuple(restored.end) == (3.0, 4.0)
        assert restored.ref is True
        assert restored.material_id == "Water"
        assert restored.anchored == [True, False]

    def test_line_dict_roundtrip_preserves_dynamic_state(self):
        line = Line((0, 0), (1, 0))
        line.dynamic = True
        line.mass = 7.5
        line.angular_vel = 1.5
        line.velocity = np.array([2.0, 3.0])
        d = line.to_dict()
        restored = Line.from_dict(d)
        assert restored.dynamic is True
        assert restored.mass == 7.5
        assert restored.angular_vel == 1.5
        assert tuple(restored.velocity) == (2.0, 3.0)

    def test_circle_dict_roundtrip(self):
        c = Circle((5.0, 6.0), 2.5, material_id="Mercury")
        c.physical = True
        d = c.to_dict()
        restored = Circle.from_dict(d)
        assert tuple(restored.center) == (5.0, 6.0)
        assert restored.radius == 2.5
        assert restored.material_id == "Mercury"
        assert restored.physical is True

    def test_point_dict_roundtrip(self):
        p = Point(1.5, 2.5, anchored=True, material_id="Oil")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert tuple(restored.pos) == (1.5, 2.5)
        assert restored.anchored is True
        assert restored.material_id == "Oil"

    def test_line_dict_includes_physical_flag_only_when_set(self):
        line = Line((0, 0), (1, 0))
        assert "physical" not in line.to_dict()
        line.physical = True
        assert line.to_dict().get("physical") is True

    def test_point_handle_flag_is_not_serialized(self):
        p = Point(0, 0)
        p.is_handle = True
        # Handles are recreated by their owning ProcessObject on load
        assert "is_handle" not in p.to_dict()


# ----- Force/torque accumulation --------------------------------------------

class TestForceAccumulators:
    def test_apply_force_accumulates(self):
        line = Line((0, 0), (1, 0))
        line.apply_force(1.0, 2.0)
        line.apply_force(3.0, 4.0)
        assert tuple(line.force_accum) == (4.0, 6.0)

    def test_apply_torque_accumulates(self):
        line = Line((0, 0), (1, 0))
        line.apply_torque(0.5)
        line.apply_torque(1.5)
        assert line.torque_accum == 2.0

    def test_clear_accumulators_zeros_force_and_torque(self):
        line = Line((0, 0), (1, 0))
        line.apply_force(1.0, 2.0)
        line.apply_torque(3.0)
        line.clear_accumulators()
        assert tuple(line.force_accum) == (0.0, 0.0)
        assert line.torque_accum == 0.0

    def test_static_entity_integrate_is_noop(self):
        line = Line((0, 0), (1, 0))
        # dynamic=False — integrate should not move endpoints
        line.apply_force(100.0, 100.0)
        line.integrate(1.0)
        assert tuple(line.start) == (0.0, 0.0)
        assert tuple(line.end) == (1.0, 0.0)
