"""Tests for core/selection.py — entity / point selection state."""

import pytest

from core.selection import SelectionManager


class TestEntitySelection:
    def test_initial_empty(self):
        sel = SelectionManager()
        assert sel.entity_count == 0
        assert sel.has_entities is False
        assert sel.has_selection is False

    def test_select_entity_adds(self):
        sel = SelectionManager()
        sel.select_entity(3)
        assert sel.is_entity_selected(3)
        assert sel.entity_count == 1

    def test_deselect_entity_removes(self):
        sel = SelectionManager()
        sel.select_entity(3)
        sel.deselect_entity(3)
        assert not sel.is_entity_selected(3)

    def test_toggle_entity_flips_state(self):
        sel = SelectionManager()
        sel.toggle_entity(5)
        assert sel.is_entity_selected(5)
        sel.toggle_entity(5)
        assert not sel.is_entity_selected(5)

    def test_set_entity_selection_replaces(self):
        sel = SelectionManager()
        sel.select_entity(1)
        sel.select_entity(2)
        sel.set_entity_selection([5, 6])
        assert sel.entities == {5, 6}

    def test_walls_property_is_mutable_view(self):
        sel = SelectionManager()
        sel.select_entity(1)
        sel.walls.add(99)  # Direct mutation through .walls
        assert sel.is_entity_selected(99)

    def test_entities_property_is_a_copy(self):
        sel = SelectionManager()
        sel.select_entity(1)
        copy = sel.entities
        copy.add(99)
        # Mutating the returned copy should NOT affect internal state
        assert not sel.is_entity_selected(99)


class TestPointSelection:
    def test_select_point(self):
        sel = SelectionManager()
        sel.select_point(2, 1)
        assert sel.is_point_selected(2, 1)

    def test_toggle_point(self):
        sel = SelectionManager()
        sel.toggle_point(0, 0)
        assert sel.is_point_selected(0, 0)
        sel.toggle_point(0, 0)
        assert not sel.is_point_selected(0, 0)

    def test_point_count_tracks_size(self):
        sel = SelectionManager()
        sel.select_point(0, 0)
        sel.select_point(1, 1)
        assert sel.point_count == 2


class TestClearOperations:
    def test_clear_empties_both(self):
        sel = SelectionManager()
        sel.select_entity(1)
        sel.select_point(2, 0)
        sel.clear()
        assert not sel.has_selection

    def test_clear_entities_leaves_points(self):
        sel = SelectionManager()
        sel.select_entity(1)
        sel.select_point(2, 0)
        sel.clear_entities()
        assert sel.entity_count == 0
        assert sel.point_count == 1

    def test_clear_points_leaves_entities(self):
        sel = SelectionManager()
        sel.select_entity(1)
        sel.select_point(2, 0)
        sel.clear_points()
        assert sel.entity_count == 1
        assert sel.point_count == 0


class TestRemapAfterDeletion:
    def test_deleted_entity_dropped_from_selection(self):
        sel = SelectionManager()
        sel.select_entity(2)
        sel.remap_after_deletion(2)
        assert not sel.is_entity_selected(2)

    def test_higher_indices_shift_down(self):
        sel = SelectionManager()
        sel.select_entity(3)
        sel.select_entity(5)
        sel.remap_after_deletion(2)
        assert sel.entities == {2, 4}

    def test_lower_indices_unchanged(self):
        sel = SelectionManager()
        sel.select_entity(0)
        sel.select_entity(1)
        sel.remap_after_deletion(5)
        assert sel.entities == {0, 1}

    def test_points_on_deleted_entity_dropped(self):
        sel = SelectionManager()
        sel.select_point(2, 0)
        sel.select_point(2, 1)
        sel.select_point(3, 0)
        sel.remap_after_deletion(2)
        # Points on entity 2 dropped; entity 3's point shifts to entity 2
        assert sel.points == {(2, 0)}


class TestSerialization:
    def test_to_dict_then_from_dict_round_trip(self):
        sel = SelectionManager()
        sel.select_entity(2)
        sel.select_entity(5)
        sel.select_point(0, 1)
        d = sel.to_dict()

        restored = SelectionManager()
        restored.from_dict(d)
        assert restored.entities == {2, 5}
        assert restored.points == {(0, 1)}
