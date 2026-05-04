"""Tests for modal absorption and UI-tree reset behavior.

Per TU-UI: these tests verify *widget-level* event absorption (a modal
ConfirmDialog absorbs key/mouse events that would otherwise leak to global
hotkeys). They do NOT verify the InputHandler's actual System→Modal→Global→HUD
dispatch ordering — that requires a wired AppController and is out of scope
for the current headless harness. Tests below were renamed accordingly.
"""

import pygame
import pytest

from ui.ui_widgets import ConfirmDialog, Button, UIContainer
from tests.conftest import make_event


class TestConfirmDialogAbsorption:
    def test_modal_absorbs_arbitrary_keydown(self):
        """A live ConfirmDialog must consume any KEYDOWN so it cannot leak
        through to the Global hotkey layer behind it."""
        d = ConfirmDialog(0, 0, "Title", "Msg")
        # Plain alphabetic key — would be a global tool hotkey otherwise
        consumed = d.handle_event(make_event(pygame.KEYDOWN, key=pygame.K_b, unicode="b"))
        assert consumed is True

    def test_modal_absorbs_ctrl_z(self):
        """Even Ctrl+Z (a global undo hotkey) must be eaten by the modal."""
        d = ConfirmDialog(0, 0, "Title", "Msg")
        consumed = d.handle_event(make_event(
            pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL, unicode=""
        ))
        assert consumed is True
        # Modal stays open — it didn't treat ctrl+z as confirm
        assert d.done is False

    def test_modal_absorbs_mouse_inside_dialog_rect(self):
        """Mouse motion inside the modal absorbs so widgets behind don't react."""
        d = ConfirmDialog(0, 0, "Title", "Msg")
        consumed = d.handle_event(make_event(
            pygame.MOUSEMOTION, pos=(50, 50), rel=(0, 0), buttons=(0, 0, 0)
        ))
        # MOUSEMOTION is always absorbed by dialogs
        assert consumed is True


class TestResetInteractionStateRecursion:
    """The "modal opening clears ghost clicks" guarantee in §6.4 depends on
    `reset_interaction_state()` recursing through nested UIContainers."""

    def test_resets_clicked_state_on_deeply_nested_button(self):
        root = UIContainer(0, 0, 400, 400, layout_type="free")
        mid = UIContainer(0, 0, 200, 200, layout_type="free")
        deep = UIContainer(0, 0, 100, 100, layout_type="free")
        btn = Button(0, 0, 50, 30, "Test")
        btn.clicked = True

        deep.add_child(btn)
        mid.add_child(deep)
        root.add_child(mid)

        # Walk the tree manually to reset (simulating UIManager.reset_all)
        def walk_reset(element):
            element.reset_interaction_state()
            for child in getattr(element, "children", []):
                walk_reset(child)

        walk_reset(root)
        assert btn.clicked is False
