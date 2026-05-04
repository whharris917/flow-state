"""Tests for core/session.py — interaction state coordinator."""

import pytest

import core.config as config
from core.session import Session, InteractionState


class TestDefaults:
    def test_initial_mode_is_sim(self, session):
        assert session.mode == config.MODE_SIM

    def test_initial_state_is_idle(self, session):
        assert session.state == InteractionState.IDLE

    def test_default_active_material_is_water(self, session):
        assert session.active_material.name == "Water"

    def test_managers_are_constructed(self, session):
        assert session.camera is not None
        assert session.selection is not None
        assert session.constraint_builder is not None
        assert session.status is not None

    def test_focused_element_starts_none(self, session):
        assert session.focused_element is None

    def test_auto_atomize_default_off(self, session):
        assert session.auto_atomize is False


class _StubTool:
    """Tool stand-in for change_tool tests — records activate/deactivate calls."""

    def __init__(self):
        self.activated = 0
        self.deactivated = 0

    def activate(self):
        self.activated += 1

    def deactivate(self):
        self.deactivated += 1


class TestToolManagement:
    def test_change_tool_activates_target(self, session):
        tool = _StubTool()
        session.tools[config.TOOL_BRUSH] = tool
        session.change_tool(config.TOOL_BRUSH)
        assert tool.activated == 1
        assert session.current_tool is tool

    def test_change_tool_deactivates_previous(self, session):
        a = _StubTool()
        b = _StubTool()
        session.tools[config.TOOL_BRUSH] = a
        session.tools[config.TOOL_LINE] = b

        session.change_tool(config.TOOL_BRUSH)
        session.change_tool(config.TOOL_LINE)
        assert a.deactivated == 1
        assert b.activated == 1
        assert session.current_tool is b

    def test_change_tool_records_per_mode(self, session):
        tool = _StubTool()
        session.tools[config.TOOL_BRUSH] = tool
        session.mode = config.MODE_SIM
        session.change_tool(config.TOOL_BRUSH)
        assert session.sim_tool == config.TOOL_BRUSH

        session.mode = config.MODE_EDITOR
        session.change_tool(config.TOOL_BRUSH)
        assert session.editor_tool == config.TOOL_BRUSH


class TestClearInteractionState:
    def test_resets_state_to_idle(self, session):
        session.state = InteractionState.PAINTING
        session.clear_interaction_state()
        assert session.state == InteractionState.IDLE

    def test_clears_selection(self, session):
        session.selection.select_entity(0)
        session.clear_interaction_state()
        assert not session.selection.has_selection

    def test_clears_placing_geo_data(self, session):
        session.placing_geo_data = {"foo": "bar"}
        session.clear_interaction_state()
        assert session.placing_geo_data is None


class TestToolSwitchMidDrag:
    def test_change_tool_deactivates_previous_even_during_drag(self, session):
        """Switching tools mid-drag MUST call deactivate on the previous tool
        so it can clean up interaction state. Otherwise: orphan drag state."""
        import core.config as config
        from core.session import InteractionState

        class _MidDragTool:
            def __init__(self):
                self.deactivated = False
                self.activated = False

            def activate(self):
                self.activated = True

            def deactivate(self):
                self.deactivated = True

        a = _MidDragTool()
        b = _MidDragTool()
        session.tools[config.TOOL_LINE] = a
        session.tools[config.TOOL_BRUSH] = b

        session.change_tool(config.TOOL_LINE)
        # Now simulate "mid-drag"
        session.state = InteractionState.DRAGGING_GEOMETRY

        session.change_tool(config.TOOL_BRUSH)
        assert a.deactivated is True
        assert b.activated is True


class TestFocusedElementLifecycle:
    def test_focused_element_assignment_and_clear(self, session):
        """focused_element is plain attribute storage; cleanup via on_focus_lost
        is the InputHandler's job. Verify the slot accepts and releases."""
        class _StubFocusable:
            def __init__(self):
                self.focus_lost_calls = 0
            def on_focus_lost(self):
                self.focus_lost_calls += 1

        target = _StubFocusable()
        session.focused_element = target
        assert session.focused_element is target

        # Caller-driven release
        old = session.focused_element
        session.focused_element = None
        old.on_focus_lost()
        assert target.focus_lost_calls == 1
        assert session.focused_element is None
