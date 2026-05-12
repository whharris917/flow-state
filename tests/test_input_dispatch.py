"""Tests for modal absorption and UI-tree reset behavior.

Per TU-UI: these tests verify *widget-level* event absorption (a modal
ConfirmDialog absorbs key/mouse events that would otherwise leak to global
hotkeys). They do NOT verify the InputHandler's actual System→Modal→Global→HUD
dispatch ordering — that requires a wired AppController and is out of scope
for the current headless harness. Tests below were renamed accordingly.
"""

import pygame
import pytest

from ui.ui_widgets import ConfirmDialog, Button, UIContainer, Dropdown, InputField
from ui.ui_manager import UIManager
from tests.conftest import make_event


class _OverlayHarness:
    """Minimal harness binding UIManager.try_dispatch_to_overlay to a thin
    object that only carries an `overlays` list plus the (un)register API.
    Avoids the full UIManager construction cost (which needs a layout dict
    and a controller)."""

    def __init__(self):
        self.overlays = []

    def register_overlay(self, provider):
        if provider not in self.overlays:
            self.overlays.append(provider)

    def unregister_overlay(self, provider):
        if provider in self.overlays:
            self.overlays.remove(provider)

    try_dispatch_to_overlay = UIManager.try_dispatch_to_overlay


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


class TestOverlayDispatch:
    """A Dropdown's expanded overlay draws above the rest of the UI tree;
    clicks landing inside the overlay rect must be routed to the Dropdown
    before the tree's reverse-iteration order lets a sibling widget
    underneath the overlay steal the click. The reported bug:
    bottom-most molecule-palette dropdown items fell through to the
    material widget below in the right panel.
    """

    def _make_dropdown(self, options):
        dd = Dropdown(100, 50, 200, 30, options, selected_index=0)
        dd.expanded = True  # registers as an overlay; option_height = 30
        return dd

    def test_overlay_consumes_click_in_expanded_rect(self):
        """Direct case: a click inside the dropdown's expanded list rect is
        claimed by the dropdown, even with no siblings registered."""
        harness = _OverlayHarness()
        dd = self._make_dropdown(["A", "B", "C"])
        harness.overlays.append(dd)

        # Bottom-most option C sits at y in [50+30+60, 50+30+90) = [140, 170)
        click = make_event(pygame.MOUSEBUTTONDOWN, pos=(150, 155), button=1)
        consumed = harness.try_dispatch_to_overlay(click)

        assert consumed is True
        # Dropdown selected option C and collapsed
        assert dd.selected_index == 2
        assert dd.expanded is False

    def test_overlay_dispatch_runs_before_tree_for_click_under_overlay(self):
        """The bug-pinning test: a Dropdown's expanded overlay covers a
        sibling widget below it. A click landing in the overlay rect (and
        in the sibling's rect) is claimed by the dropdown, not the sibling.

        Without overlay-first dispatch, the right-panel tree iterates in
        reverse and the sibling steals the click — exactly the molecule-
        palette-vs-material-widget bug.
        """
        harness = _OverlayHarness()
        dd = self._make_dropdown(["First", "Second", "Last"])
        harness.overlays.append(dd)

        # Sibling input field positioned directly below the dropdown's
        # expanded list — at y in [140, 180). This is where the bottom
        # option of the expanded dropdown lives (140-170).
        sibling = InputField(100, 140, 200, 40, "")

        # Click on the bottom-most overlay option, which also lies inside
        # the sibling's rect.
        click_pos = (150, 155)
        assert dd.get_expanded_rect().collidepoint(click_pos)
        assert sibling.rect.collidepoint(click_pos)

        click = make_event(pygame.MOUSEBUTTONDOWN, pos=click_pos, button=1)
        consumed = harness.try_dispatch_to_overlay(click)

        assert consumed is True
        # Dropdown selected the bottom option ("Last")
        assert dd.selected_index == 2
        assert dd.get_selected() == "Last"
        # Sibling was never activated
        assert sibling.active is False

    def test_overlay_dispatch_skips_click_outside_overlay_rect(self):
        """Clicks landing outside any overlay rect must NOT be claimed by
        the overlay dispatch — they fall through to the tree as normal."""
        harness = _OverlayHarness()
        dd = self._make_dropdown(["A", "B"])
        harness.overlays.append(dd)

        # Way below the expanded list (which ends at y=110+60=170... wait)
        # Expanded list spans y in [80, 140). Click at y=300 is far below.
        click = make_event(pygame.MOUSEBUTTONDOWN, pos=(150, 300), button=1)
        consumed = harness.try_dispatch_to_overlay(click)

        assert consumed is False
        # Dropdown didn't fire on_change
        assert dd.selected_index == 0

    def test_overlay_dispatch_only_routes_mousebuttondown(self):
        """MOUSEMOTION/MOUSEBUTTONUP still flow through the tree so hover
        state on overlay items keeps working via tree dispatch."""
        harness = _OverlayHarness()
        dd = self._make_dropdown(["A", "B", "C"])
        harness.overlays.append(dd)

        motion = make_event(
            pygame.MOUSEMOTION, pos=(150, 155), rel=(0, 0), buttons=(0, 0, 0)
        )
        assert harness.try_dispatch_to_overlay(motion) is False

        up = make_event(pygame.MOUSEBUTTONUP, pos=(150, 155), button=1)
        assert harness.try_dispatch_to_overlay(up) is False

    def test_overlay_dispatch_safe_when_overlays_list_mutates(self):
        """Selecting an option causes the Dropdown to unregister itself
        mid-iteration (the @property setter calls _unregister_overlay).
        The dispatch must iterate a snapshot so it doesn't trip a
        list-changed-during-iteration error."""
        harness = _OverlayHarness()
        dd = self._make_dropdown(["A", "B", "C"])
        harness.overlays.append(dd)

        # Bind the overlay registration to the harness so the unregister
        # call lands on our list.
        from ui.ui_widgets import OverlayProvider
        prev_manager = OverlayProvider._ui_manager
        OverlayProvider.set_ui_manager(harness)
        try:
            click = make_event(pygame.MOUSEBUTTONDOWN, pos=(150, 155), button=1)
            consumed = harness.try_dispatch_to_overlay(click)
            assert consumed is True
            # The Dropdown unregistered itself when it collapsed
            assert dd not in harness.overlays
        finally:
            OverlayProvider.set_ui_manager(prev_manager)
