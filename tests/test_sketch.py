"""Tests for model/sketch.py — entity & constraint management."""

import pytest

from model.sketch import Sketch
from model.geometry import Line, Circle, Point
from model.constraints import Coincident, Length, Angle


class TestEntityCRUD:
    def test_new_sketch_is_empty(self, sketch):
        assert sketch.entities == []
        assert sketch.constraints == []

    def test_add_line_returns_index(self, sketch):
        idx = sketch.add_line((0, 0), (1, 0))
        assert idx == 0
        assert len(sketch.entities) == 1
        assert isinstance(sketch.entities[0], Line)

    def test_add_circle_returns_index(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        idx = sketch.add_circle((5, 5), 2.0)
        assert idx == 1
        assert isinstance(sketch.entities[1], Circle)

    def test_remove_entity_pops_from_list(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((1, 1), (2, 2))
        sketch.remove_entity(0)
        assert len(sketch.entities) == 1
        assert tuple(sketch.entities[0].start) == (1.0, 1.0)

    def test_get_entity_returns_none_for_invalid_index(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        assert sketch.get_entity(99) is None
        assert sketch.get_entity(-1) is None

    def test_clear_empties_entities_and_constraints(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((1, 0), (2, 0))
        sketch.add_constraint("LENGTH", [0], value=2.0)
        sketch.clear()
        assert sketch.entities == []
        assert sketch.constraints == []

    def test_clear_preserves_materials(self, sketch):
        original_count = len(sketch.materials)
        sketch.clear()
        assert len(sketch.materials) == original_count
        assert "Wall" in sketch.materials


class TestConstraintRegistration:
    def test_add_constraint_via_string_api(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        c = sketch.add_constraint("LENGTH", [0], value=5.0)
        assert c is not None
        assert len(sketch.constraints) == 1

    def test_add_constraint_object_appends(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((1, 0), (2, 0))
        sketch.add_constraint_object(Coincident(0, 1, 1, 0), solve=False)
        assert len(sketch.constraints) == 1
        assert sketch.constraints[0].type == "COINCIDENT"

    def test_remove_constraint_by_index(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint("LENGTH", [0], value=5.0)
        sketch.remove_constraint(0)
        assert sketch.constraints == []

    def test_remove_entity_drops_dependent_constraints(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        assert len(sketch.constraints) == 1
        sketch.remove_entity(0)
        assert sketch.constraints == []

    def test_remove_entity_shifts_indices_in_remaining_constraints(self, sketch):
        sketch.add_line((0, 0), (1, 0))   # idx 0 — to be removed
        sketch.add_line((2, 0), (3, 0))   # idx 1 — survives, becomes 0
        sketch.add_line((4, 0), (5, 0))   # idx 2 — survives, becomes 1
        sketch.add_constraint_object(Length(2, 3.0), solve=False)
        sketch.remove_entity(0)
        assert sketch.constraints[0].indices == [1]


class TestConstraintConflictResolution:
    def test_new_angle_constraint_replaces_old_on_same_entity(self, sketch):
        sketch.add_line((0, 0), (1, 1))
        sketch.add_constraint_object(Angle("HORIZONTAL", 0), solve=False)
        sketch.add_constraint_object(Angle("VERTICAL", 0), solve=False)
        # Only the latest angle constraint on entity 0 should remain
        assert len(sketch.constraints) == 1
        assert sketch.constraints[0].type == "VERTICAL"

    def test_non_angle_constraints_do_not_replace_each_other(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        sketch.add_constraint_object(Length(0, 7.0), solve=False)
        # Multiple LENGTH constraints can coexist (resolution is the solver's job)
        assert len(sketch.constraints) == 2

    def test_parallel_perpendicular_share_conflict_pool(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((0, 1), (1, 2))
        sketch.add_constraint_object(Angle("PARALLEL", 0, 1), solve=False)
        sketch.add_constraint_object(Angle("PERPENDICULAR", 0, 1), solve=False)
        # PERPENDICULAR replaces PARALLEL on the same entity pair
        assert len(sketch.constraints) == 1
        assert sketch.constraints[0].type == "PERPENDICULAR"


class TestMaterials:
    def test_default_presets_loaded(self, sketch):
        for name in ("Water", "Wall", "Oil", "Mercury", "Honey"):
            assert name in sketch.materials

    def test_get_material_returns_named(self, sketch):
        wall = sketch.get_material("Wall")
        assert wall.name == "Wall"

    def test_get_material_falls_back_when_unknown(self, sketch):
        # Unknown material should fall back to a default rather than raise
        mat = sketch.get_material("DoesNotExist")
        assert mat is not None

    def test_materials_preserved_after_clear(self, sketch):
        sketch.materials["Wall"].sigma = 99.0
        sketch.clear()
        # Materials persist across clear (per Sketch.clear's documented contract)
        assert sketch.materials["Wall"].sigma == 99.0


class TestAnchorToggle:
    def test_toggle_anchor_on_line_endpoint(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.toggle_anchor(0, 0)
        assert sketch.entities[0].anchored == [True, False]
        sketch.toggle_anchor(0, 0)
        assert sketch.entities[0].anchored == [False, False]

    def test_toggle_anchor_on_point_entity(self, sketch):
        sketch.entities.append(Point(0, 0))
        sketch.toggle_anchor(0, 0)
        assert sketch.entities[0].anchored is True


class TestSerialization:
    def test_to_dict_includes_all_subsystems(self, sketch):
        sketch.add_line((0, 0), (1, 0))
        sketch.add_constraint_object(Length(0, 1.0), solve=False)
        d = sketch.to_dict()
        assert "entities" in d
        assert "constraints" in d
        assert "materials" in d

    def test_restore_repopulates_entities(self, sketch):
        sketch.add_line((1.0, 2.0), (3.0, 4.0))
        sketch.add_circle((5, 5), 2.0)
        d = sketch.to_dict()

        new_sketch = Sketch()
        new_sketch.restore(d)
        assert len(new_sketch.entities) == 2
        assert isinstance(new_sketch.entities[0], Line)
        assert isinstance(new_sketch.entities[1], Circle)
        assert tuple(new_sketch.entities[0].start) == (1.0, 2.0)

    def test_restore_repopulates_constraints(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        d = sketch.to_dict()

        new_sketch = Sketch()
        new_sketch.restore(d)
        assert len(new_sketch.constraints) == 1
        assert new_sketch.constraints[0].type == "LENGTH"

    def test_restore_preserves_active_driver(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_constraint_object(Length(0, 5.0), solve=False)
        # Prime a driver with explicit base_value/base_time so to_dict captures them
        sketch.constraints[0].driver = {"type": "sin", "amp": 1.0, "freq": 1.0, "phase": 0.0}
        sketch.constraints[0].base_value = 5.0
        sketch.constraints[0].base_time = 2.0
        d = sketch.to_dict()

        new_sketch = Sketch()
        new_sketch.restore(d)
        c = new_sketch.constraints[0]
        assert c.driver is not None
        assert c.driver["type"] == "sin"
        assert c.base_value == 5.0
        assert c.base_time == 2.0


class TestRemoveEntityIndexShift:
    def test_coincident_tuple_indices_shift_correctly(self, sketch):
        sketch.add_line((0, 0), (1, 0))   # 0 — to be removed
        sketch.add_line((1, 0), (2, 0))   # 1 — survives, becomes 0
        sketch.add_line((2, 0), (3, 0))   # 2 — survives, becomes 1
        # Coincident between line 1's end and line 2's start
        sketch.add_constraint_object(Coincident(1, 1, 2, 0), solve=False)
        sketch.remove_entity(0)
        # Tuple-form indices should both shift down by one
        assert sketch.constraints[0].indices == [(0, 1), (1, 0)]

    def test_midpoint_mixed_indices_shift_correctly(self, sketch):
        from model.constraints import Midpoint
        from model.geometry import Point
        sketch.add_line((0, 0), (1, 0))   # 0 — to be removed
        sketch.add_line((2, 0), (12, 0))  # 1 — survives, becomes 0
        sketch.entities.append(Point(5, 5))  # 2 — survives, becomes 1
        # Midpoint constrains point at idx 2 to lie on line at idx 1
        sketch.add_constraint_object(Midpoint(2, 0, 1), solve=False)
        sketch.remove_entity(0)
        # Tuple part (point) shifts; scalar part (line ref) also shifts
        assert sketch.constraints[0].indices == [(1, 0), 0]


class TestConstraintConflictNuance:
    def test_horizontal_then_parallel_on_overlapping_set_keeps_both(self, sketch):
        from model.constraints import Angle
        sketch.add_line((0, 0), (1, 0))
        sketch.add_line((0, 1), (1, 2))
        sketch.add_constraint_object(Angle("HORIZONTAL", 0), solve=False)
        sketch.add_constraint_object(Angle("PARALLEL", 0, 1), solve=False)
        # Index sets {0} vs {0,1} differ, so neither replaces the other under
        # the angle-conflict rule. Documents the current behavior.
        types = sorted(c.type for c in sketch.constraints)
        assert types == ["HORIZONTAL", "PARALLEL"]


class TestAnchorBehavior:
    def test_fully_anchored_line_holds_under_length(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].anchored = [True, True]
        # Both endpoints anchored — Length cannot move them. Should not crash.
        sketch.add_constraint_object(Length(0, 99.0))
        assert tuple(sketch.entities[0].start) == (0.0, 0.0)
        assert tuple(sketch.entities[0].end) == (10.0, 0.0)


class TestTryCreateConstraint:
    def test_length_on_line_returns_length(self, sketch):
        sketch.add_line((0, 0), (5, 0))
        c = sketch.try_create_constraint("LENGTH", [0], [])
        assert c is not None
        assert c.type == "LENGTH"

    def test_length_on_circle_returns_radius(self, sketch):
        sketch.add_circle((5, 5), 3.0)
        c = sketch.try_create_constraint("LENGTH", [0], [])
        # CONSTRAINT_DEFS['LENGTH'] has two rules; the Circle path returns Radius
        assert c is not None
        assert c.type == "RADIUS"

    def test_radius_on_line_returns_none(self, sketch):
        sketch.add_line((0, 0), (5, 0))
        c = sketch.try_create_constraint("RADIUS", [0], [])
        assert c is None

    def test_parallel_with_one_entity_returns_none(self, sketch):
        sketch.add_line((0, 0), (5, 0))
        c = sketch.try_create_constraint("PARALLEL", [0], [])
        assert c is None  # PARALLEL needs 2 entities
