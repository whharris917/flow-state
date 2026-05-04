"""Tests for ui/tools.py — tool state machines.

Each tool is exercised by feeding pygame events into handle_event() and
asserting the effect on the underlying Scene (entity count, constraints,
interaction state, etc.). The ToolContext is a real one wrapping a FakeApp,
so commands flow through Scene.execute and undo/redo behaves correctly.
"""

import time
import pygame
import pytest

import core.config as config
from core.session import InteractionState
from model.geometry import Line, Circle
from ui.tools import (
    LineTool, RectTool, CircleTool, PointTool, BrushTool, SelectTool,
)
from tests.conftest import make_event


# ----- LineTool -------------------------------------------------------------

class TestLineTool:
    def test_drag_creates_one_line(self, tool_ctx, layout):
        tool = LineTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR

        # Mouse down inside viewport
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        # Drag to a new screen position
        tool.handle_event(
            make_event(pygame.MOUSEMOTION, pos=(700, 350), rel=(100, 50), buttons=(1, 0, 0)),
            layout,
        )
        # Hold long enough that release is treated as drag-end (not quick click)
        time.sleep(0.15)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(700, 350), button=1), layout)

        # Exactly one line should remain
        sketch = tool_ctx._get_sketch()
        assert len(sketch.entities) == 1
        assert isinstance(sketch.entities[0], Line)
        assert tool.dragging is False
        assert tool_ctx.interaction_state == InteractionState.IDLE

    def test_click_outside_viewport_is_ignored(self, tool_ctx, layout):
        tool = LineTool(tool_ctx)
        # Click in the left panel area (mx <= LEFT_X)
        consumed = tool.handle_event(
            make_event(pygame.MOUSEBUTTONDOWN, pos=(100, 300), button=1),
            layout,
        )
        assert consumed is False
        assert len(tool_ctx._get_sketch().entities) == 0

    def test_quick_click_enters_click_click_mode(self, tool_ctx, layout):
        tool = LineTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR

        # Down + immediate up — under QUICK_CLICK_THRESHOLD => click-click mode
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(600, 300), button=1), layout)
        assert tool.click_click_mode is True
        # The preview line still exists; tool stays in dragging state until 2nd click
        assert tool.dragging is True

    def test_supersede_keeps_stack_size_one(self, tool_ctx, layout):
        tool = LineTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR

        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        # Multiple motions superseding the previous line
        for x in range(610, 700, 10):
            tool.handle_event(
                make_event(pygame.MOUSEMOTION, pos=(x, 320), rel=(10, 0), buttons=(1, 0, 0)),
                layout,
            )
        time.sleep(0.15)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(700, 320), button=1), layout)

        # Despite many MOUSEMOTION events, the undo stack contains only one entry
        scene = tool_ctx._get_scene()
        assert scene.commands.can_undo() is True
        assert len(scene.commands.undo_stack) == 1

    def test_cancel_during_drag_discards_preview(self, tool_ctx, layout):
        tool = LineTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR

        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        assert tool.dragging is True
        tool.cancel()
        # Discard removes the preview entirely; nothing in the sketch
        assert len(tool_ctx._get_sketch().entities) == 0
        assert tool.dragging is False
        assert tool_ctx.interaction_state == InteractionState.IDLE


# ----- RectTool -------------------------------------------------------------

class TestRectTool:
    def test_drag_creates_four_lines(self, tool_ctx, layout):
        tool = RectTool(tool_ctx)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(
            make_event(pygame.MOUSEMOTION, pos=(800, 400), rel=(200, 100), buttons=(1, 0, 0)),
            layout,
        )
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(800, 400), button=1), layout)

        sketch = tool_ctx._get_sketch()
        assert len(sketch.entities) == 4

    def test_drag_creates_eight_constraints(self, tool_ctx, layout):
        tool = RectTool(tool_ctx)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(
            make_event(pygame.MOUSEMOTION, pos=(800, 400), rel=(200, 100), buttons=(1, 0, 0)),
            layout,
        )
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(800, 400), button=1), layout)
        # 4 corner coincidents + 4 angle (H/V/H/V) constraints
        assert len(tool_ctx._get_sketch().constraints) == 8

    def test_supersede_yields_one_undo_entry(self, tool_ctx, layout):
        tool = RectTool(tool_ctx)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        for x in range(620, 800, 20):
            tool.handle_event(
                make_event(pygame.MOUSEMOTION, pos=(x, 320), rel=(20, 0), buttons=(1, 0, 0)),
                layout,
            )
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(800, 320), button=1), layout)

        scene = tool_ctx._get_scene()
        # Single undo should remove the entire rectangle (lines + constraints)
        scene.undo()
        assert tool_ctx._get_sketch().entities == []
        assert tool_ctx._get_sketch().constraints == []


# ----- CircleTool -----------------------------------------------------------

class TestCircleTool:
    def test_drag_creates_circle_with_computed_radius(self, tool_ctx, layout):
        tool = CircleTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR

        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(
            make_event(pygame.MOUSEMOTION, pos=(700, 300), rel=(100, 0), buttons=(1, 0, 0)),
            layout,
        )
        time.sleep(0.15)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(700, 300), button=1), layout)

        sketch = tool_ctx._get_sketch()
        assert len(sketch.entities) == 1
        assert isinstance(sketch.entities[0], Circle)
        # Radius was computed from drag distance; should be > minimum 0.1
        assert sketch.entities[0].radius > 0.1

    def test_quick_click_enters_click_click_mode(self, tool_ctx, layout):
        tool = CircleTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(600, 300), button=1), layout)
        assert tool.click_click_mode is True


# ----- PointTool ------------------------------------------------------------

class TestPointTool:
    def test_click_creates_degenerate_line(self, tool_ctx, layout):
        tool = PointTool(tool_ctx)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        sketch = tool_ctx._get_sketch()
        assert len(sketch.entities) == 1
        assert isinstance(sketch.entities[0], Line)
        # Point is encoded as a degenerate line (start == end)
        assert tuple(sketch.entities[0].start) == tuple(sketch.entities[0].end)


# ----- BrushTool ------------------------------------------------------------

class TestBrushTool:
    def test_mousedown_in_sim_mode_enters_painting_state(self, tool_ctx, layout):
        tool = BrushTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_SIM

        consumed = tool.handle_event(
            make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1),
            layout,
        )
        assert consumed is True
        assert tool_ctx.interaction_state == InteractionState.PAINTING

    def test_mouseup_returns_to_idle(self, tool_ctx, layout):
        tool = BrushTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_SIM

        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.handle_event(make_event(pygame.MOUSEBUTTONUP, pos=(600, 300), button=1), layout)
        assert tool_ctx.interaction_state == InteractionState.IDLE

    def test_mousedown_takes_particle_snapshot(self, tool_ctx, layout):
        tool = BrushTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_SIM

        sim = tool_ctx._app.scene.simulation
        before = len(sim.undo_stack)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        # Snapshot stored so a subsequent undo can reverse the upcoming brush stroke
        assert len(sim.undo_stack) == before + 1

    def test_editor_mode_ignores_mouse(self, tool_ctx, layout):
        tool = BrushTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_EDITOR
        consumed = tool.handle_event(
            make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1),
            layout,
        )
        assert consumed is False
        assert tool_ctx.interaction_state == InteractionState.IDLE

    def test_cancel_resets_painting_state(self, tool_ctx, layout):
        tool = BrushTool(tool_ctx)
        tool_ctx._app.session.mode = config.MODE_SIM
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        tool.cancel()
        assert tool_ctx.interaction_state == InteractionState.IDLE


# ----- SelectTool -----------------------------------------------------------

class TestSelectTool:
    def test_click_on_empty_space_does_not_select(self, tool_ctx, layout):
        tool = SelectTool(tool_ctx)
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(600, 300), button=1), layout)
        # No entities to hit, so selection should still be empty
        assert not tool_ctx.selection.has_selection

    def test_click_on_entity_body_selects_it(self, tool_ctx, layout):
        # Place a horizontal line at y=25 (world center) spanning x=10..40
        tool_ctx._get_sketch().add_line((10, 25), (40, 25))

        tool = SelectTool(tool_ctx)
        # World center maps to viewport center — click there to hit the line
        cx_screen = layout["MID_X"] + layout["MID_W"] // 2
        cy_screen = config.TOP_MENU_H + layout["MID_H"] // 2
        tool.handle_event(make_event(pygame.MOUSEBUTTONDOWN, pos=(cx_screen, cy_screen), button=1), layout)
        # The line should now be selected
        assert tool_ctx.selection.is_entity_selected(0)

    def test_deactivate_clears_selection(self, tool_ctx, layout):
        tool_ctx._get_sketch().add_line((10, 25), (40, 25))
        tool_ctx.selection.select_entity(0)
        tool = SelectTool(tool_ctx)
        tool.deactivate()
        assert not tool_ctx.selection.has_selection
