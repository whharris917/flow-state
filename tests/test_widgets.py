"""Tests for ui/ui_widgets.py — UIElement, Button, InputField, ContextMenu, ConfirmDialog.

These tests run against pygame with the dummy SDL video driver (set in conftest).
They exercise interaction state machines, not pixel output.
"""

import pygame
import pytest

from ui.ui_widgets import (
    UIElement, UIContainer, Button, InputField,
    ContextMenu, ConfirmDialog, SmartSlider, MiniSlider,
)
from tests.conftest import make_event


# ----- UIElement / UIContainer ---------------------------------------------

class TestUIElement:
    def test_default_state(self):
        el = UIElement(0, 0, 100, 50)
        assert el.visible is True
        assert el.disabled is False
        assert el.hovered is False
        assert el.parent is None

    def test_set_position_updates_rect(self):
        el = UIElement(0, 0, 100, 50)
        el.set_position(20, 30)
        assert el.rect.x == 20
        assert el.rect.y == 30


class TestUIContainer:
    def test_add_child_sets_parent_pointer(self):
        parent = UIContainer(0, 0, 200, 200)
        child = UIElement(10, 10, 50, 20)
        parent.add_child(child)
        assert child.parent is parent

    def test_set_position_propagates_to_children(self):
        parent = UIContainer(0, 0, 200, 200, layout_type="free")
        child = UIElement(10, 10, 50, 20)
        parent.add_child(child)
        parent.set_position(100, 100)
        # Child rect should have shifted by the same delta
        assert child.rect.x == 110
        assert child.rect.y == 110

    def test_clear_children(self):
        parent = UIContainer(0, 0, 200, 200)
        parent.add_child(UIElement())
        parent.add_child(UIElement())
        parent.clear_children()
        assert parent.children == []


# ----- Button ---------------------------------------------------------------

class TestButton:
    def test_initial_state(self):
        btn = Button(0, 0, 100, 30, "Test")
        assert btn.active is False
        assert btn.clicked is False
        assert btn.hovered is False

    def test_mousemotion_inside_sets_hovered(self):
        btn = Button(0, 0, 100, 30, "Test")
        btn.handle_event(make_event(pygame.MOUSEMOTION, pos=(50, 15), rel=(0, 0), buttons=(0, 0, 0)))
        assert btn.hovered is True

    def test_mousemotion_outside_clears_hovered(self):
        btn = Button(0, 0, 100, 30, "Test")
        btn.hovered = True
        btn.handle_event(make_event(pygame.MOUSEMOTION, pos=(500, 500), rel=(0, 0), buttons=(0, 0, 0)))
        assert btn.hovered is False

    def test_mousedown_inside_sets_clicked(self):
        btn = Button(0, 0, 100, 30, "Test")
        consumed = btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        assert btn.clicked is True
        assert consumed is True

    def test_mouseup_completes_click_with_callback(self):
        log = []
        btn = Button(0, 0, 100, 30, "Test", toggle=False, callback=lambda: log.append("fired"))
        btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        btn.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(50, 15), button=1))
        assert log == ["fired"]
        assert btn.clicked is False

    def test_drag_off_button_before_release_does_not_fire(self):
        log = []
        btn = Button(0, 0, 100, 30, "Test", toggle=False, callback=lambda: log.append("fired"))
        btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        # Release outside the button — callback should NOT fire
        btn.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(500, 500), button=1))
        assert log == []
        assert btn.clicked is False

    def test_toggle_button_flips_active_on_click(self):
        btn = Button(0, 0, 100, 30, "Test", toggle=True, active=False)
        btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        btn.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(50, 15), button=1))
        assert btn.active is True
        btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        btn.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(50, 15), button=1))
        assert btn.active is False

    def test_disabled_button_ignores_input(self):
        log = []
        btn = Button(0, 0, 100, 30, "Test", toggle=False, callback=lambda: log.append("fired"))
        btn.disabled = True
        btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        btn.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(50, 15), button=1))
        assert log == []

    def test_reset_interaction_state_clears_clicked(self):
        btn = Button(0, 0, 100, 30, "Test")
        btn.clicked = True
        btn.reset_interaction_state()
        assert btn.clicked is False


# ----- InputField -----------------------------------------------------------

class TestInputField:
    def test_initial_text(self):
        f = InputField(0, 0, 100, 30, initial_text="hello")
        assert f.text == "hello"
        assert f.cursor_pos == len("hello")
        assert f.active is False

    def test_click_inside_activates(self):
        f = InputField(0, 0, 100, 30)
        f.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        assert f.active is True

    def test_click_outside_deactivates(self):
        f = InputField(0, 0, 100, 30)
        f.active = True
        f.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(500, 500), button=1))
        assert f.active is False

    def test_typing_inserts_at_cursor(self):
        f = InputField(0, 0, 100, 30)
        f.active = True
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_a, unicode="a"))
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_b, unicode="b"))
        assert f.text == "ab"
        assert f.cursor_pos == 2

    def test_backspace_removes_char_before_cursor(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode=""))
        assert f.text == "ab"
        assert f.cursor_pos == 2

    def test_return_deactivates_field(self):
        f = InputField(0, 0, 100, 30)
        f.active = True
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert f.active is False

    def test_left_right_arrows_move_cursor(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        assert f.cursor_pos == 3
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_LEFT, unicode=""))
        assert f.cursor_pos == 2
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RIGHT, unicode=""))
        assert f.cursor_pos == 3

    def test_get_value_parses_float(self):
        f = InputField(0, 0, 100, 30, initial_text="3.14")
        assert f.get_value() == 3.14

    def test_get_value_returns_default_on_invalid(self):
        f = InputField(0, 0, 100, 30, initial_text="not a number")
        assert f.get_value(default=99.0) == 99.0

    def test_set_value_only_updates_when_inactive(self):
        f = InputField(0, 0, 100, 30, initial_text="hello")
        f.active = True
        f.set_value("ignored")
        assert f.text == "hello"  # active fields are not overwritten

        f.active = False
        f.set_value(42)
        assert f.text == "42"


# ----- ContextMenu ----------------------------------------------------------

class TestContextMenu:
    def test_click_on_option_records_action(self):
        menu = ContextMenu(0, 0, ["Foo", "Bar", "Baz"])
        # First option lives at y in [5, 35) — click at y=20 selects Foo
        menu.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 20), button=1))
        assert menu.action == "Foo"

    def test_click_on_second_option(self):
        menu = ContextMenu(0, 0, ["Foo", "Bar"])
        menu.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 50), button=1))
        assert menu.action == "Bar"

    def test_click_outside_dismisses_menu(self):
        menu = ContextMenu(0, 0, ["Foo"])
        menu.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(500, 500), button=1))
        assert menu.action == "CLOSE"

    def test_mousemotion_updates_selected_idx(self):
        menu = ContextMenu(0, 0, ["Foo", "Bar"])
        menu.handle_event(make_event(pygame.MOUSEMOTION, pos=(50, 20), rel=(0, 0), buttons=(0, 0, 0)))
        assert menu.selected_idx == 0
        menu.handle_event(make_event(pygame.MOUSEMOTION, pos=(50, 50), rel=(0, 0), buttons=(0, 0, 0)))
        assert menu.selected_idx == 1


# ----- ConfirmDialog --------------------------------------------------------

class TestConfirmDialog:
    def test_initial_state(self):
        d = ConfirmDialog(0, 0, "Title", "Msg")
        assert d.done is False
        assert d.confirmed is False
        assert d.cancelled is False

    def test_enter_confirms_when_not_destructive(self):
        d = ConfirmDialog(0, 0, "Title", "Msg", destructive=False)
        d.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert d.confirmed is True
        assert d.done is True

    def test_enter_cancels_when_destructive(self):
        # Destructive guard: hasty Enter must not destroy state
        d = ConfirmDialog(0, 0, "Title", "Are you sure?", destructive=True)
        d.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert d.cancelled is True
        assert d.confirmed is False
        assert d.done is True

    def test_escape_cancels(self):
        d = ConfirmDialog(0, 0, "Title", "Msg")
        d.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode=""))
        assert d.cancelled is True
        assert d.confirmed is False
        assert d.done is True

    def test_click_confirm_button(self):
        d = ConfirmDialog(0, 0, "Title", "Msg")
        cx = d.btn_confirm.rect.centerx
        cy = d.btn_confirm.rect.centery
        d.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(cx, cy), button=1))
        assert d.confirmed is True
        assert d.done is True

    def test_click_cancel_button(self):
        d = ConfirmDialog(0, 0, "Title", "Msg")
        cx = d.btn_cancel.rect.centerx
        cy = d.btn_cancel.rect.centery
        d.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(cx, cy), button=1))
        assert d.cancelled is True
        assert d.done is True

    def test_other_keydown_blocked_from_propagation(self):
        # Per the dialog's contract: while modal, all key events are absorbed
        # so they don't leak to global hotkeys.
        d = ConfirmDialog(0, 0, "Title", "Msg")
        consumed = d.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_a, unicode="a"))
        assert consumed is True
        assert d.done is False


class TestDisabledButtonAbsorption:
    def test_disabled_button_does_not_consume_mousedown(self):
        """A disabled Button must NOT consume MOUSEBUTTONDOWN — clicks should
        pass through to widgets behind it."""
        btn = Button(0, 0, 100, 30, "Test")
        btn.disabled = True
        consumed = btn.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(50, 15), button=1))
        # Disabled widgets early-return False; the click is not consumed
        assert consumed is False


class TestInputFieldFocusLoss:
    def test_clicking_outside_active_field_deactivates_it(self):
        """The current `on_focus_lost` lifecycle for InputField: clicking
        outside while active flips active=False."""
        f = InputField(0, 0, 100, 30, initial_text="foo")
        f.active = True
        f.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(500, 500), button=1))
        assert f.active is False
        # Text is preserved (no implicit revert)
        assert f.text == "foo"


# ============================================================================
# Expanded InputField coverage
# ============================================================================

class TestInputFieldEditing:
    """Cursor / text-editing semantics beyond the original TestInputField."""

    def test_typing_inserts_at_cursor_middle(self):
        f = InputField(0, 0, 100, 30, initial_text="abcd")
        f.active = True
        f.cursor_pos = 2  # between 'b' and 'c'
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_x, unicode="x"))
        assert f.text == "abxcd"
        assert f.cursor_pos == 3

    def test_delete_key_removes_char_at_cursor(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.cursor_pos = 1
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_DELETE, unicode=""))
        assert f.text == "ac"
        assert f.cursor_pos == 1  # cursor stays put (char to the right is removed)

    def test_delete_at_end_is_noop(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.cursor_pos = 3  # at end
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_DELETE, unicode=""))
        assert f.text == "abc"

    def test_backspace_at_start_is_noop(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.cursor_pos = 0
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode=""))
        assert f.text == "abc"
        assert f.cursor_pos == 0

    def test_home_jumps_to_start(self):
        f = InputField(0, 0, 100, 30, initial_text="abcde")
        f.active = True
        f.cursor_pos = 3
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_HOME, unicode=""))
        assert f.cursor_pos == 0

    def test_end_jumps_to_end(self):
        f = InputField(0, 0, 100, 30, initial_text="abcde")
        f.active = True
        f.cursor_pos = 1
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_END, unicode=""))
        assert f.cursor_pos == 5

    def test_left_arrow_at_start_is_noop(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.cursor_pos = 0
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_LEFT, unicode=""))
        assert f.cursor_pos == 0

    def test_right_arrow_at_end_is_noop(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        f.cursor_pos = 3
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RIGHT, unicode=""))
        assert f.cursor_pos == 3

    def test_typing_ignored_when_inactive(self):
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = False
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_x, unicode="x"))
        assert f.text == "abc"

    def test_non_printable_unicode_ignored(self):
        """Tab, escape, control chars etc. shouldn't be inserted into text."""
        f = InputField(0, 0, 100, 30, initial_text="abc")
        f.active = True
        # Tab character — has unicode but isn't printable
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_TAB, unicode="\t"))
        assert f.text == "abc"

    def test_set_value_formats_float_to_two_dp(self):
        """set_value with a float uses {:.2f} formatting."""
        f = InputField(0, 0, 100, 30)
        f.active = False
        f.set_value(3.14159)
        assert f.text == "3.14"

    def test_set_value_with_int_stringifies(self):
        f = InputField(0, 0, 100, 30)
        f.active = False
        f.set_value(42)
        assert f.text == "42"

    def test_set_value_blocked_when_active(self):
        """Prevents external code from clobbering text the user is typing."""
        f = InputField(0, 0, 100, 30, initial_text="user_typing")
        f.active = True
        f.set_value(99.0)
        assert f.text == "user_typing"


class TestInputFieldCursorBlink:
    """Cursor visibility lifecycle — blink while active, stable while inactive."""

    def test_inactive_field_keeps_cursor_visible_and_timer_zero(self):
        """When inactive, cursor stays visible (constant) and timer doesn't advance."""
        f = InputField(0, 0, 100, 30)
        f.active = False
        f.cursor_visible = False  # corrupt state to verify reset
        f.cursor_blink_timer = 0.3
        f.update(dt=0.1)
        assert f.cursor_visible is True
        assert f.cursor_blink_timer == 0.0

    def test_active_field_toggles_cursor_after_blink_rate(self):
        f = InputField(0, 0, 100, 30)
        f.active = True
        f.cursor_visible = True
        f.cursor_blink_timer = 0.0
        # One full blink-rate elapses — cursor flips
        f.update(dt=f.CURSOR_BLINK_RATE)
        assert f.cursor_visible is False
        # Another full cycle — back to visible
        f.update(dt=f.CURSOR_BLINK_RATE)
        assert f.cursor_visible is True

    def test_keypress_resets_blink_state(self):
        """Any keypress while active resets the blink to visible+timer=0
        so the cursor doesn't disappear right as the user types."""
        f = InputField(0, 0, 100, 30)
        f.active = True
        f.cursor_visible = False
        f.cursor_blink_timer = 0.3
        f.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_a, unicode="a"))
        assert f.cursor_visible is True
        assert f.cursor_blink_timer == 0.0


# ============================================================================
# SmartSlider — the slider used by the main right-panel physics controls
# ============================================================================

def _slider_track_center(slider):
    """Helper: world coords at the centre of a slider's track."""
    return (slider.rect_track.centerx, slider.rect_track.centery)


class TestSmartSliderConstruction:
    def test_initial_state(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "Gravity")
        assert s.val == 5.0
        assert s.label == "Gravity"
        assert s.min_val == 0.0
        assert s.max_val == 10.0
        assert s.dragging is False

    def test_initial_readout_shows_initial_value(self):
        """The slider's InputField should mirror the initial value, so the
        user can see what the slider is set to before touching anything."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 7.5, "Temp")
        assert s.in_val.text == "7.5"  # str(7.5) — what the constructor uses


class TestSmartSliderDrag:
    """Click-and-drag on the slider track."""

    def test_click_on_track_starts_drag(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        cx, cy = _slider_track_center(s)
        # Hover first (the implementation requires hovered=True for the
        # MOUSEBUTTONDOWN handler to start dragging)
        s.handle_event(make_event(pygame.MOUSEMOTION, pos=(cx, cy), rel=(0, 0), buttons=(0, 0, 0)))
        s.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(cx, cy), button=1))
        assert s.dragging is True

    def test_mouseup_ends_drag(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(0, 0), button=1))
        assert s.dragging is False

    def test_drag_to_middle_sets_val_to_midpoint(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 0.0, "G")
        s.dragging = True
        cx, cy = _slider_track_center(s)
        s.handle_event(make_event(pygame.MOUSEMOTION, pos=(cx, cy), rel=(0, 0), buttons=(1, 0, 0)))
        # Middle of track on a 0..10 range should produce 5.0
        assert s.val == pytest.approx(5.0, abs=0.1)

    def test_drag_to_left_edge_sets_min(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(s.rect_track.left, s.rect_track.centery),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        assert s.val == pytest.approx(s.min_val)

    def test_drag_to_right_edge_sets_max(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(s.rect_track.right, s.rect_track.centery),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        assert s.val == pytest.approx(s.max_val)

    def test_drag_past_left_edge_clamps_to_min(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(s.rect_track.left - 100, s.rect_track.centery),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        assert s.val == pytest.approx(s.min_val)

    def test_drag_past_right_edge_clamps_to_max(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(s.rect_track.right + 500, s.rect_track.centery),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        assert s.val == pytest.approx(s.max_val)

    def test_drag_updates_input_field_readout(self):
        """The slider's InputField must reflect the slider's value during a
        drag — otherwise the user sees a stale number and concludes the
        slider isn't working. The InputField is the only on-screen readout
        of the slider's numeric value."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 0.0, "G")
        s.dragging = True
        cx, cy = _slider_track_center(s)
        s.handle_event(make_event(pygame.MOUSEMOTION, pos=(cx, cy),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        # After dragging to mid-track, the readout should show ~5.0
        # (allowing any sensible formatted form: "5.00", "5.0", etc.)
        readout = float(s.in_val.text)
        assert readout == pytest.approx(s.val, abs=0.01)

    def test_mousebutton_motion_when_not_dragging_does_not_change_val(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.dragging = False
        original_val = s.val
        cx, cy = _slider_track_center(s)
        s.handle_event(make_event(pygame.MOUSEMOTION, pos=(cx, cy),
                                  rel=(0, 0), buttons=(0, 0, 0)))
        assert s.val == original_val

    def test_repeated_drag_motions_accumulate_to_track_position(self):
        """Each MOUSEMOTION during drag sets val by absolute track position
        — not by delta — so multiple motions converge on the final cursor
        position rather than accumulating drift."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 0.0, "G")
        s.dragging = True
        # Drag through three positions; final val should match the final cursor
        for x in (s.rect_track.left + 50, s.rect_track.left + 100,
                  s.rect_track.left + 150):
            s.handle_event(make_event(pygame.MOUSEMOTION,
                                      pos=(x, s.rect_track.centery),
                                      rel=(0, 0), buttons=(1, 0, 0)))
        # Final position is 150 px into a w-10 track at x=0..w-10
        # So pct = 150 / (w-10), val = pct * 10
        expected_pct = 150 / s.rect_track.width
        assert s.val == pytest.approx(expected_pct * 10.0, abs=0.1)


class TestSmartSliderInputField:
    """Typing a value directly into the slider's InputField."""

    def test_input_entry_updates_val(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        # Simulate the user typing "7.5" by setting the field text + RETURN
        s.in_val.active = True
        s.in_val.text = "7.5"
        s.in_val.cursor_pos = 3
        # The handle_event(RETURN) path dispatches via the in_val branch
        # at the top of SmartSlider.handle_event
        s.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert s.val == 7.5

    def test_input_entry_above_max_clamps_to_max(self):
        """Typing a value above the slider's max should clamp to max — not
        leave the slider in an out-of-range state. This is the user's
        'modifications to the limits don't stay' bug."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.in_val.active = True
        s.in_val.text = "999.0"
        s.in_val.cursor_pos = 5
        s.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert s.val == pytest.approx(s.max_val)

    def test_input_entry_below_min_clamps_to_min(self):
        """Symmetric clamp on the low end."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.in_val.active = True
        s.in_val.text = "-100.0"
        s.in_val.cursor_pos = 6
        s.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert s.val == pytest.approx(s.min_val)

    def test_input_entry_in_range_passes_through(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        s.in_val.active = True
        s.in_val.text = "3.25"
        s.in_val.cursor_pos = 4
        s.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert s.val == 3.25


class TestSmartSliderLayout:
    """set_position must move all sub-widgets (track + input field) in lockstep."""

    def test_set_position_moves_track(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        original_track_offset_x = s.rect_track.x - s.rect.x
        original_track_offset_y = s.rect_track.y - s.rect.y
        s.set_position(500, 300)
        assert s.rect_track.x - s.rect.x == original_track_offset_x
        assert s.rect_track.y - s.rect.y == original_track_offset_y

    def test_set_position_moves_input_field(self):
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "G")
        original_input_offset_x = s.in_val.rect.x - s.rect.x
        original_input_offset_y = s.in_val.rect.y - s.rect.y
        s.set_position(500, 300)
        assert s.in_val.rect.x - s.rect.x == original_input_offset_x
        assert s.in_val.rect.y - s.rect.y == original_input_offset_y

    def test_drag_after_set_position_uses_new_track_location(self):
        """If the layout moves the slider, drag math must use the new track
        position — otherwise clicks at the wrong location."""
        s = SmartSlider(0, 0, 200, 0.0, 10.0, 0.0, "G")
        s.set_position(500, 300)
        s.dragging = True
        # Drag to centre of NEW track location
        cx = s.rect_track.centerx
        cy = s.rect_track.centery
        s.handle_event(make_event(pygame.MOUSEMOTION, pos=(cx, cy),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        assert s.val == pytest.approx(5.0, abs=0.1)


class TestSmartSliderIndependence:
    """Two SmartSliders must not bleed state into each other."""

    def test_two_sliders_have_independent_val(self):
        s1 = SmartSlider(0, 0, 200, 0.0, 10.0, 1.0, "A")
        s2 = SmartSlider(0, 100, 200, 0.0, 10.0, 9.0, "B")
        s1.dragging = True
        s1.handle_event(make_event(pygame.MOUSEMOTION,
                                   pos=(s1.rect_track.centerx, s1.rect_track.centery),
                                   rel=(0, 0), buttons=(1, 0, 0)))
        # s2 untouched
        assert s2.val == 9.0

    def test_dragging_one_does_not_set_dragging_on_other(self):
        s1 = SmartSlider(0, 0, 200, 0.0, 10.0, 5.0, "A")
        s2 = SmartSlider(0, 100, 200, 0.0, 10.0, 5.0, "B")
        s1.handle_event(make_event(pygame.MOUSEMOTION,
                                   pos=(s1.rect_track.centerx, s1.rect_track.centery),
                                   rel=(0, 0), buttons=(0, 0, 0)))
        s1.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                   pos=(s1.rect_track.centerx, s1.rect_track.centery),
                                   button=1))
        assert s1.dragging is True
        assert s2.dragging is False


# ============================================================================
# MiniSlider — compact slider used in property panels. Documented here as
# the reference behaviour that SmartSlider should match.
# ============================================================================


class TestMiniSlider:
    def test_initial_state(self):
        s = MiniSlider(0, 0, 200, 0.0, 10.0, 5.0, "Sigma")
        assert s.val == 5.0
        assert s.label == "Sigma"
        assert s.min_val == 0.0
        assert s.max_val == 10.0
        # Initial readout formatted to 2 dp
        assert s.in_val.text == "5.00"

    def test_drag_updates_val_and_readout(self):
        s = MiniSlider(0, 0, 200, 0.0, 10.0, 0.0, "S")
        s.dragging = True
        s.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(s.rect_track.centerx, s.rect_track.centery),
                                  rel=(0, 0), buttons=(1, 0, 0)))
        # Val at midpoint
        assert s.val == pytest.approx(5.0, abs=0.1)
        # Readout matches
        assert float(s.in_val.text) == pytest.approx(s.val, abs=0.01)

    def test_set_value_clamps_to_range(self):
        s = MiniSlider(0, 0, 200, 0.0, 10.0, 5.0, "S")
        s.set_value(999.0)
        assert s.val == 10.0
        s.set_value(-5.0)
        assert s.val == 0.0

    def test_input_field_entry_clamps(self):
        """MiniSlider clamps typed values to its range — this is what
        SmartSlider should also do."""
        s = MiniSlider(0, 0, 200, 0.0, 10.0, 5.0, "S")
        s.in_val.active = True
        s.in_val.text = "999"
        s.in_val.cursor_pos = 3
        s.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode=""))
        assert s.val == 10.0

    def test_set_position_moves_track_and_input(self):
        s = MiniSlider(0, 0, 200, 0.0, 10.0, 5.0, "S")
        track_dx = s.rect_track.x - s.rect.x
        input_dx = s.in_val.rect.x - s.rect.x
        s.set_position(500, 300)
        assert s.rect_track.x - s.rect.x == track_dx
        assert s.in_val.rect.x - s.rect.x == input_dx
