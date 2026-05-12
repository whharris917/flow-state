"""Tests for the MenuBar dropdown overlay-draw lift.

The MenuBar is a child of `ui.root` and so its `draw()` runs during the
tree-order pass — BEFORE panels, viewport, and other children of the root
container. The original implementation rendered the active dropdown inside
that same `draw()` call, which meant the dropdown was painted, then panels
drew over it. Visually the dropdown was invisible; events worked but users
clicked blind.

The fix splits rendering:
- `MenuBar.draw()` — bar only (background, top-bar labels), keeps
  `dropdown_rect` up to date for hit-testing.
- `MenuBar.draw_dropdown_overlay()` — the dropdown body, called from
  `UIManager._draw_overlays()` so it lands on top of everything.

These tests verify both the surface API and the integration through
UIManager + the FlowStateApp render pipeline.
"""

import pygame
import pytest

from ui.ui_widgets import MenuBar
from tests.conftest import make_event


# =============================================================================
# MenuBar API surface
# =============================================================================

class TestMenuBarDrawSplit:
    def _make_menu_with_items(self):
        m = MenuBar(800, h=30)
        m.items["File"] = ["New", "Open...", "Save"]
        m.items["Tools"] = ["Settings..."]
        m.items["Demos"] = ["Demixing", "Micelles"]
        return m

    def test_draw_alone_does_not_render_dropdown_body(self):
        """When draw() is called on a MenuBar with an active menu, it
        should NOT paint the dropdown body — that's draw_dropdown_overlay's
        job. We verify by checking that draw() doesn't crash and that
        dropdown_rect gets set for hit-testing."""
        m = self._make_menu_with_items()
        m.active_menu = "File"
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Populate item_rects via handle_event so dropdown_rect can compute
        m.handle_event(make_event(pygame.MOUSEMOTION, pos=(0, 0), rel=(0, 0), buttons=(0, 0, 0)))
        # draw() should set dropdown_rect (used for hit-testing) but not crash.
        m.draw(screen, font)
        # Dropdown rect is computed because active_menu is set
        assert m.dropdown_rect is not None
        # The dropdown rect should sit BELOW the menu bar
        assert m.dropdown_rect.y >= m.rect.height

    def test_draw_dropdown_overlay_is_noop_when_inactive(self):
        m = self._make_menu_with_items()
        # active_menu starts as None
        assert m.active_menu is None
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Should not crash and should not require dropdown_rect
        m.draw_dropdown_overlay(screen, font)
        # Still inactive — no side effects
        assert m.active_menu is None

    def test_draw_clears_dropdown_rect_when_menu_closed(self):
        """After activating then deactivating the menu, dropdown_rect goes
        back to None so stale clicks can't fire phantom items."""
        m = self._make_menu_with_items()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Open File menu — populate item_rects via handle_event first
        m.handle_event(make_event(pygame.MOUSEMOTION, pos=(0, 0), rel=(0, 0), buttons=(0, 0, 0)))
        m.active_menu = "File"
        m.draw(screen, font)
        assert m.dropdown_rect is not None
        # Close the menu
        m.active_menu = None
        m.draw(screen, font)
        assert m.dropdown_rect is None

    def test_dropdown_rect_position_matches_active_item(self):
        """dropdown_rect.x must align with the active top-bar item's x so
        the dropdown visually anchors to its parent label."""
        m = self._make_menu_with_items()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Trigger a draw to populate item_rects with real text widths
        m.draw(screen, font)
        # Now activate Tools (the 2nd item) and re-draw
        m.active_menu = "Tools"
        m.draw(screen, font)
        # dropdown_rect.x should match Tools' item_rect.x
        assert m.dropdown_rect.x == m.item_rects["Tools"].x

    def test_draw_dropdown_overlay_renders_when_active(self):
        """After activating a menu, draw_dropdown_overlay should write
        actual pixels for the dropdown body. We verify by checking the
        framebuffer in the dropdown region differs from pre-draw."""
        m = self._make_menu_with_items()
        screen = pygame.Surface((800, 600))
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        m.active_menu = "File"
        m.draw(screen, font)  # updates dropdown_rect

        # Snapshot a pixel that the dropdown body will touch
        # (the dropdown sits at y >= ~32, x = active item's x)
        center_x = m.dropdown_rect.centerx
        center_y = m.dropdown_rect.centery
        before = screen.get_at((center_x, center_y))[:3]
        m.draw_dropdown_overlay(screen, font)
        after = screen.get_at((center_x, center_y))[:3]
        # Dropdown body is config.PANEL_BG_COLOR which is not pure black
        assert before != after, "draw_dropdown_overlay should paint the dropdown body"


# =============================================================================
# Click flow through the dropdown
# =============================================================================

class TestDropdownClickFlow:
    """The user must be able to click an item inside an open dropdown. The
    click requires a prior MOUSEMOTION inside the dropdown to set
    hover_item_idx; then a MOUSEBUTTONDOWN in the same area fires the item."""

    def _make_open_menu(self):
        m = MenuBar(800, h=30)
        m.items["File"] = ["New", "Open...", "Save"]
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)  # populate item_rects
        # Click File to open it
        file_rect = m.item_rects["File"]
        m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                  pos=file_rect.center, button=1))
        # Re-draw so dropdown_rect updates
        m.draw(screen, font)
        return m

    def test_mousemotion_inside_dropdown_sets_hover_idx(self):
        m = self._make_open_menu()
        assert m.dropdown_rect is not None
        # Hover over the 2nd item ("Open..." at idx 1)
        item_y = m.dropdown_rect.y + 5 + 1 * 30 + 15  # middle of item 1
        item_x = m.dropdown_rect.centerx
        m.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(item_x, item_y), rel=(0, 0),
                                  buttons=(0, 0, 0)))
        assert m.hover_item_idx == 1

    def test_click_on_dropdown_item_returns_item_label(self):
        m = self._make_open_menu()
        item_y = m.dropdown_rect.y + 5 + 1 * 30 + 15  # middle of item 1 ("Open...")
        item_x = m.dropdown_rect.centerx
        # First a motion to set hover_item_idx
        m.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(item_x, item_y), rel=(0, 0),
                                  buttons=(0, 0, 0)))
        # Then the click
        result = m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                            pos=(item_x, item_y), button=1))
        assert result == "Open..."
        # Menu auto-closes after item selection
        assert m.active_menu is None

    def test_click_outside_dropdown_closes_menu(self):
        m = self._make_open_menu()
        # Click far below the dropdown
        m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                  pos=(400, 500), button=1))
        assert m.active_menu is None


# =============================================================================
# Integration: UIManager._draw_overlays calls draw_dropdown_overlay
# =============================================================================

class TestUIManagerDrawsDropdownLast:
    """The whole point of the fix: UIManager's overlay phase must call
    MenuBar.draw_dropdown_overlay so the dropdown lands on top of every
    other UI surface (panels, viewport, dropdowns from the right panel)."""

    def test_draw_overlays_invokes_menu_dropdown_overlay(self):
        """The simplest check: _draw_overlays calls menu.draw_dropdown_overlay.
        We monkeypatch the method and verify it gets called."""
        # Build a stand-alone MenuBar (we don't need a full UIManager).
        # Use a shim that mirrors the _draw_overlays method's call site.
        class _OverlayShim:
            overlays = []
            def __init__(self, menu):
                self.menu = menu
            from ui.ui_manager import UIManager
            _draw_overlays = UIManager._draw_overlays

        m = MenuBar(800, h=30)
        m.items["File"] = ["New"]
        shim = _OverlayShim(m)
        # Spy on draw_dropdown_overlay
        calls = []
        m.draw_dropdown_overlay = lambda screen, font: calls.append((screen, font))
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        shim._draw_overlays(screen, font)
        assert len(calls) == 1

    def test_draw_overlays_safe_when_no_menu_attribute(self):
        """The hasattr/None guard in _draw_overlays must keep the method
        working for any UIManager-like shim that doesn't carry a menu."""
        class _OverlayShim:
            overlays = []
            from ui.ui_manager import UIManager
            _draw_overlays = UIManager._draw_overlays

        shim = _OverlayShim()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Should not raise
        shim._draw_overlays(screen, font)

    def test_draw_overlays_safe_when_menu_is_none(self):
        class _OverlayShim:
            overlays = []
            menu = None
            from ui.ui_manager import UIManager
            _draw_overlays = UIManager._draw_overlays

        shim = _OverlayShim()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        # Should not raise
        shim._draw_overlays(screen, font)


# =============================================================================
# Bug regression: handle_event must NOT rebuild item_rects with a fixed width
# (was: lines 1030-1035 used width=60, draw used actual text width — clicking
# "Demos" hit the "Help" rect drawn to its left)
# =============================================================================

class TestHitRectMatchesVisualLabel:
    """The hit-rects used by handle_event must match the rects used by draw,
    otherwise clicks land on the wrong category. Specifically: clicking on
    the visual position of "Demos" must open Demos, not the empty/missing
    Help or some other earlier category."""

    def _make_menu(self):
        m = MenuBar(800, h=30)
        m.items["File"] = ["New"]
        m.items["Tools"] = ["Settings..."]
        m.items["Demos"] = ["Demixing", "Micelles"]
        return m

    def test_handle_event_does_not_overwrite_draw_item_rects(self):
        """Regression: handle_event used to rebuild item_rects with fixed
        width=60, replacing the actual-width rects set by draw. After the
        fix, item_rects from draw survive across a handle_event call."""
        m = self._make_menu()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        rects_after_draw = dict(m.item_rects)  # snapshot
        # Send a no-op mouse motion (outside any item)
        m.handle_event(make_event(pygame.MOUSEMOTION,
                                  pos=(500, 5), rel=(0, 0),
                                  buttons=(0, 0, 0)))
        # item_rects must be unchanged — handle_event must not overwrite them
        for key, r in rects_after_draw.items():
            assert key in m.item_rects, f"{key} dropped from item_rects"
            assert m.item_rects[key] == r, (
                f"{key} rect changed: was {r}, now {m.item_rects[key]}"
            )

    def test_clicking_demos_label_opens_demos(self):
        """The literal bug: render the menu bar, click at the centre of
        the Demos visible label, verify Demos opens (not the item that
        was visually adjacent)."""
        m = self._make_menu()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)  # populate item_rects with real widths

        demos_rect = m.item_rects["Demos"]
        click_pos = demos_rect.center
        result = m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                            pos=click_pos, button=1))
        assert result is True
        assert m.active_menu == "Demos", (
            f"Expected Demos to open after clicking its centre at {click_pos}; "
            f"active_menu={m.active_menu}"
        )

    def test_clicking_each_label_opens_its_own_menu(self):
        """Sweep: each item's centre click must open that exact item."""
        m = self._make_menu()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        for key in ("File", "Tools", "Demos"):
            # Close any prior active menu by clicking outside
            m.active_menu = None
            target = m.item_rects[key]
            m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                       pos=target.center, button=1))
            assert m.active_menu == key, (
                f"Clicking centre of {key} opened {m.active_menu} instead"
            )

    def test_no_item_rect_overlaps_another(self):
        """The visible labels must not overlap. (Pre-fix the actual-width
        rects from draw didn't overlap either — but the fixed-width rects
        from handle_event DID extend past the visible label and into the
        next item's visible area. After removing the duplicate rebuild,
        the only item_rects in play are draw's actual-width rects, which
        are well-spaced by construction.)"""
        m = self._make_menu()
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        rects = list(m.item_rects.values())
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not rects[i].colliderect(rects[j]), (
                    f"Items {i} and {j} overlap: {rects[i]} vs {rects[j]}"
                )


# =============================================================================
# Bug regression: empty menu categories must not render as ghost-clickable
# =============================================================================

class TestEmptyCategoriesSkipped:
    """The pre-fix default `Help: []` rendered an empty label in the bar
    that absorbed clicks and opened a useless empty dropdown. Verify that
    categories with no items are skipped at render and hit-detection."""

    def test_empty_category_not_in_item_rects(self):
        m = MenuBar(800, h=30)
        m.items["File"] = ["New"]
        m.items["EmptyCat"] = []  # explicitly empty
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        # EmptyCat should not have an entry in item_rects
        assert "EmptyCat" not in m.item_rects
        assert "File" in m.item_rects

    def test_click_at_empty_category_position_does_not_open_it(self):
        """If an EmptyCat existed visually between File and a populated
        item, clicks anywhere outside the populated items should not open
        EmptyCat. After the skip-empty fix there's no rect at all for
        EmptyCat, so the click falls through cleanly."""
        m = MenuBar(800, h=30)
        m.items["File"] = ["New"]
        m.items["EmptyCat"] = []
        m.items["Demos"] = ["Demixing"]
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)

        # Click between File and Demos (would have been EmptyCat's visual
        # slot, but EmptyCat is skipped). No item should activate.
        if "Demos" in m.item_rects and "File" in m.item_rects:
            file_right = m.item_rects["File"].right
            demos_left = m.item_rects["Demos"].left
            if demos_left > file_right + 2:
                # Click in the gap between them
                gap_x = (file_right + demos_left) // 2
                m.handle_event(make_event(pygame.MOUSEBUTTONDOWN,
                                           pos=(gap_x, 15), button=1))
                assert m.active_menu is None

    def test_default_menubar_has_no_help_category(self):
        """Regression: previously MenuBar.__init__ seeded `Help: []` which
        rendered in the bar even though it was never populated. After
        the cleanup, defaults carry only `File` (which the caller is
        expected to populate)."""
        m = MenuBar(800, h=30)
        # Help should not be in defaults anymore
        assert "Help" not in m.items
        # File is still there as a placeholder (UIManager populates it)
        assert "File" in m.items

    def test_categories_added_after_construction_render_in_order(self):
        """UIManager adds Tools and Demos after construction. They should
        appear in the visual order they were inserted (Python dict semantics)
        and each get a clickable rect."""
        m = MenuBar(800, h=30)
        m.items["File"] = ["New"]
        m.items["Tools"] = ["Cross-ε"]
        m.items["Demos"] = ["Demixing"]
        screen = pygame.Surface((800, 600))
        font = pygame.font.Font(None, 18)
        m.draw(screen, font)
        keys = list(m.item_rects.keys())
        assert keys == ["File", "Tools", "Demos"]
        # x positions strictly increasing
        xs = [m.item_rects[k].x for k in keys]
        assert xs == sorted(xs)
