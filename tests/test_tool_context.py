"""Tests for core/tool_context.py — Air Gap facade integrity.

The ToolContext is the only sanctioned surface tools may access. These tests
pin its contract: no leaks of Sketch/Simulation/Scene through public attrs;
the documented escape hatches stay underscore-prefixed; servo / material
write paths return copies and round-trip cleanly.
"""

import pytest

from core.tool_context import ToolContext
from core.scene import Scene
from model.sketch import Sketch
from engine.simulation import Simulation


class TestFacadeIntegrity:
    def test_no_public_attribute_leaks_app_or_model_objects(self, tool_ctx):
        """A reflective sweep over public attributes must not return a
        Scene / Sketch / Simulation. Underscore-prefixed names (the documented
        escape hatches) are exempt."""
        leaks = []
        for name in dir(tool_ctx):
            if name.startswith("_"):
                continue
            if name in ("execute", "discard"):
                continue  # methods that take a command — fine
            try:
                value = getattr(tool_ctx, name)
            except Exception:
                continue
            if isinstance(value, (Scene, Sketch, Simulation)):
                leaks.append((name, type(value).__name__))
        assert leaks == [], f"Public attributes leak model/engine objects: {leaks}"

    def test_app_attribute_is_underscore_prefixed(self, tool_ctx):
        # The internal app reference must not be on a public name
        assert hasattr(tool_ctx, "_app")
        assert not hasattr(tool_ctx, "app")

    def test_get_sketch_and_get_scene_are_underscore_only(self, tool_ctx):
        """The sanctioned subclass-property delegation surface stays
        underscore-prefixed. Public renames would make the discipline-only
        Air Gap visible at every tool callsite."""
        assert callable(getattr(tool_ctx, "_get_sketch"))
        assert callable(getattr(tool_ctx, "_get_scene"))
        assert not hasattr(tool_ctx, "get_sketch")
        assert not hasattr(tool_ctx, "get_scene")


class TestServoFacade:
    def test_set_interaction_data_writes_four_keys(self, tool_ctx):
        tool_ctx.set_interaction_data((1.0, 2.0), entity_idx=0, handle_t=0.5)
        data = tool_ctx._app.scene.sketch.interaction_data
        assert data is not None
        assert set(data.keys()) == {"target", "entity_idx", "handle_t", "point_idx"}
        assert data["target"] == (1.0, 2.0)
        assert data["entity_idx"] == 0
        assert data["handle_t"] == 0.5
        assert data["point_idx"] is None

    def test_set_interaction_data_with_point_idx(self, tool_ctx):
        tool_ctx.set_interaction_data((3.0, 4.0), entity_idx=1, point_idx=1)
        data = tool_ctx._app.scene.sketch.interaction_data
        assert data["point_idx"] == 1
        assert data["handle_t"] is None

    def test_update_interaction_target_only_changes_target(self, tool_ctx):
        tool_ctx.set_interaction_data((1.0, 2.0), entity_idx=0, handle_t=0.5)
        tool_ctx.update_interaction_target((9.0, 9.0))
        data = tool_ctx._app.scene.sketch.interaction_data
        assert data["target"] == (9.0, 9.0)
        # Entity, handle_t, point_idx should be unchanged
        assert data["entity_idx"] == 0
        assert data["handle_t"] == 0.5

    def test_update_interaction_target_when_no_active_drag_is_noop(self, tool_ctx):
        # interaction_data is None
        assert not tool_ctx.has_interaction_data()
        # Should not crash, must not create a fresh dict
        tool_ctx.update_interaction_target((9.0, 9.0))
        assert tool_ctx._app.scene.sketch.interaction_data is None

    def test_clear_interaction_data_nulls_buffer(self, tool_ctx):
        tool_ctx.set_interaction_data((1.0, 2.0), entity_idx=0, handle_t=0.5)
        assert tool_ctx.has_interaction_data()
        tool_ctx.clear_interaction_data()
        assert not tool_ctx.has_interaction_data()
        assert tool_ctx._app.scene.sketch.interaction_data is None


class TestActiveMaterial:
    def test_returns_a_copy_not_a_reference(self, tool_ctx):
        """get_active_material must return a copy — mutating the returned
        dict must not affect the underlying material."""
        original_sigma = tool_ctx._app.session.active_material.sigma
        m = tool_ctx.get_active_material()
        m["sigma"] = 999.0
        # Underlying material must be unchanged
        assert tool_ctx._app.session.active_material.sigma == original_sigma
        # And a fresh fetch returns the original sigma, not 999
        m2 = tool_ctx.get_active_material()
        assert m2["sigma"] == original_sigma

    def test_returns_expected_keys(self, tool_ctx):
        m = tool_ctx.get_active_material()
        assert set(m.keys()) >= {"name", "sigma", "epsilon", "color"}


class TestEntityIteration:
    def test_iter_entities_yields_read_only_views(self, tool_ctx):
        tool_ctx._get_sketch().add_line((0, 0), (1, 0))
        results = list(tool_ctx.iter_entities())
        assert len(results) == 1
        idx, data = results[0]
        assert idx == 0
        # Read-only view should expose render data — not the entity itself
        assert "entity_type" in data
        assert "render_data" in data
        # The yielded dict should not be a Line
        from model.geometry import Line
        assert not isinstance(data, Line)

    def test_get_entity_count_matches_sketch(self, tool_ctx):
        assert tool_ctx.get_entity_count() == 0
        tool_ctx._get_sketch().add_line((0, 0), (1, 0))
        tool_ctx._get_sketch().add_circle((5, 5), 1.0)
        assert tool_ctx.get_entity_count() == 2


class TestCommandFactory:
    def test_create_coincident_command_returns_command_when_valid(self, tool_ctx):
        sketch = tool_ctx._get_sketch()
        sketch.add_line((0, 0), (5, 0))
        sketch.add_line((5, 0), (10, 0))
        cmd = tool_ctx.create_coincident_command(0, 1, 1, 0)
        assert cmd is not None

    def test_create_coincident_command_with_valid_indices_succeeds(self, tool_ctx):
        """Even with no entities currently in the sketch, the COINCIDENT
        factory rule (w=0, p=2) matches and produces a constraint object —
        validation against entity existence happens at solve time, not at
        constraint creation. Documents this as current behavior."""
        cmd = tool_ctx.create_coincident_command(0, 0, 1, 0)
        # Rule matches → returns a command (which would later fail to solve)
        assert cmd is not None
