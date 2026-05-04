"""Tests for ui/ui_widgets.py — UIElement, Button, InputField, ContextMenu, ConfirmDialog.

These tests run against pygame with the dummy SDL video driver (set in conftest).
They exercise interaction state machines, not pixel output.
"""

import pygame
import pytest

from ui.ui_widgets import (
    UIElement, UIContainer, Button, InputField,
    ContextMenu, ConfirmDialog,
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
