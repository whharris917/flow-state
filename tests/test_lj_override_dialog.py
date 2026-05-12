"""Tests for the LjOverrideDialog (UI for editing cross-pair ε)."""

import math

import pygame
import pytest

from model.sketch import Sketch
from ui.lj_override_dialog import LjOverrideDialog


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_construction_with_full_sketch(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        assert d.visible is True
        assert d.done is False
        assert d.apply is False
        # Dropdowns seeded from sketch.materials
        assert len(d.dropdown_a.options) == len(sketch.materials)
        assert len(d.dropdown_b.options) == len(sketch.materials)

    def test_initial_pair_distinct(self):
        """Default selection picks two distinct materials when possible."""
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        a, b = d._current_pair()
        # Sketch has many materials → A and B should differ
        assert a != b

    def test_input_field_shows_current_eps(self):
        """The input field should reflect the effective ε of the initial pair
        so the user sees the current value before editing."""
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        a, b = d._current_pair()
        expected = sketch.get_lj_epsilon(a, b)
        # InputField stores as string; round-trip through get_value
        # to compare numerically.
        got = d.in_eps.get_value(None)
        assert got == pytest.approx(expected, abs=0.01)


# =============================================================================
# Apply: set_lj_cross_override path
# =============================================================================

class TestApply:
    def test_apply_writes_override(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        # Force a known pair: Water + Oil (no R2 default override on this pair)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Oil")
        d.in_eps.set_value(0.42)
        d._do_apply()
        assert sketch.get_lj_epsilon("Water", "Oil") == pytest.approx(0.42)
        assert d.apply is True

    def test_apply_same_material_pair_is_noop(self):
        """Setting an override on (A, A) doesn't make sense — diagonal of
        the ε matrix is the material's own ε. Ignore silently."""
        sketch = Sketch()
        before = dict(sketch.lj_cross_overrides)
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Water")
        d.in_eps.set_value(0.5)
        d._do_apply()
        assert sketch.lj_cross_overrides == before
        assert d.apply is False

    def test_apply_negative_eps_clamps_to_floor(self):
        """Non-positive ε would make the LJ well disappear / flip sign.
        Should snap to a tiny positive floor rather than write nonsense."""
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Oil")
        d.in_eps.set_value(-1.0)
        d._do_apply()
        eps = sketch.get_lj_epsilon("Water", "Oil")
        assert eps > 0.0
        assert eps < 0.01  # floor should be small but positive


# =============================================================================
# Reset: remove_lj_cross_override path
# =============================================================================

class TestReset:
    def test_reset_drops_existing_override(self):
        sketch = Sketch()
        # Polar-Nonpolar has a seeded R2 override (0.25)
        assert sketch.get_lj_epsilon("Polar", "Nonpolar") == pytest.approx(0.25)
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Polar")
        d.dropdown_b.selected_index = names.index("Nonpolar")
        d._do_reset()
        # Now falls back to L-B (= 1.0 since both ε=1.0)
        assert sketch.get_lj_epsilon("Polar", "Nonpolar") == pytest.approx(1.0)
        assert d.apply is True

    def test_reset_when_no_override_is_safe_noop(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        # Water-Oil has no seeded override
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Oil")
        d._do_reset()
        # No KeyError, no mutation flag
        assert d.apply is False

    def test_reset_refreshes_input_to_lb_value(self):
        """After Reset, the input field should display the L-B value, not
        the previous override value — otherwise an immediate Re-Apply
        would re-introduce the override the user just removed."""
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Polar")
        d.dropdown_b.selected_index = names.index("Nonpolar")
        # Initial: shows the seeded 0.25
        d._refresh_input_from_pair()
        assert d.in_eps.get_value(None) == pytest.approx(0.25, abs=0.01)
        d._do_reset()
        # After reset, shows L-B = 1.0
        assert d.in_eps.get_value(None) == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Done button + apply flag semantics
# =============================================================================

class TestDoneFlag:
    def test_apply_flag_starts_false(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        assert d.apply is False

    def test_apply_flag_true_after_any_mutation(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Oil")
        d.in_eps.set_value(0.5)
        d._do_apply()
        assert d.apply is True

    def test_done_flag_set_by_button_click(self):
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        # Synthesise a click on the Done button
        click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {'pos': d.btn_done.rect.center, 'button': 1},
        )
        # Done buttons fire on MOUSEBUTTONUP; need to also synthesise an UP
        # to register the click — easier to just set the flag directly via
        # the internal helper: clicking is exercised by the dialog event
        # handler test below.
        d.handle_event(click)
        up = pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            {'pos': d.btn_done.rect.center, 'button': 1},
        )
        d.handle_event(up)
        assert d.done is True


# =============================================================================
# Dropdown switching auto-refreshes input field
# =============================================================================

class TestDropdownRefresh:
    def test_switching_pair_updates_input_field(self):
        """Changing either dropdown should update the input field to the
        new pair's effective ε so the user always sees the live value."""
        sketch = Sketch()
        d = LjOverrideDialog(0, 0, sketch)
        names = list(sketch.materials.keys())

        # Switch to Polar-Nonpolar (seeded 0.25)
        d.dropdown_a.selected_index = names.index("Polar")
        d.dropdown_b.selected_index = names.index("Nonpolar")
        d._refresh_input_from_pair()
        assert d.in_eps.get_value(None) == pytest.approx(0.25, abs=0.01)

        # Switch to Water-Oil (no override, L-B = √0.8)
        d.dropdown_a.selected_index = names.index("Water")
        d.dropdown_b.selected_index = names.index("Oil")
        d._refresh_input_from_pair()
        assert d.in_eps.get_value(None) == pytest.approx(math.sqrt(0.8), abs=0.01)
