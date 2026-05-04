"""Tests for core/utils.py — coordinate transforms and grouping helpers."""

import pytest

import core.utils as utils
from model.constraints import Coincident
from tests.conftest import make_layout


class TestCoordinateTransforms:
    def test_inverse_round_trip(self):
        layout = make_layout()
        # Pick a screen position inside the viewport
        sx_in, sy_in = 600, 300
        wx, wy = utils.screen_to_sim(
            sx_in, sy_in,
            zoom=1.0, pan_x=0.0, pan_y=0.0,
            world_size=50.0, layout=layout,
        )
        sx_out, sy_out = utils.sim_to_screen(
            wx, wy,
            zoom=1.0, pan_x=0.0, pan_y=0.0,
            world_size=50.0, layout=layout,
        )
        assert sx_out == pytest.approx(sx_in, abs=1)
        assert sy_out == pytest.approx(sy_in, abs=1)

    def test_pan_translates_world_position(self):
        layout = make_layout()
        # Same screen point with different pans should map to different world points
        wx0, _ = utils.screen_to_sim(600, 300, 1.0, 0.0, 0.0, 50.0, layout)
        wx1, _ = utils.screen_to_sim(600, 300, 1.0, 100.0, 0.0, 50.0, layout)
        assert wx0 != wx1

    def test_zoom_scales_world_units_per_pixel(self):
        layout = make_layout()
        # Two screen points 100 px apart at zoom=1 should be 100/scale apart in world
        wx0, _ = utils.screen_to_sim(600, 300, 1.0, 0.0, 0.0, 50.0, layout)
        wx1, _ = utils.screen_to_sim(700, 300, 1.0, 0.0, 0.0, 50.0, layout)
        d_at_1x = wx1 - wx0
        # Same delta at zoom=2 should be half the world distance
        wx0z, _ = utils.screen_to_sim(600, 300, 2.0, 0.0, 0.0, 50.0, layout)
        wx1z, _ = utils.screen_to_sim(700, 300, 2.0, 0.0, 0.0, 50.0, layout)
        d_at_2x = wx1z - wx0z
        assert d_at_2x == pytest.approx(d_at_1x / 2.0, rel=1e-3)


class TestConnectedGroup:
    def test_isolated_entity_is_its_own_group(self):
        # No constraints — entity 5 is alone
        group = utils.get_connected_group(constraints=[], start_wall_idx=5)
        assert group == {5}

    def test_coincident_pair_forms_group(self):
        constraints = [Coincident(0, 1, 1, 0)]  # line 0 end → line 1 start
        group = utils.get_connected_group(constraints, start_wall_idx=0)
        assert group == {0, 1}

    def test_transitive_chain_is_grouped(self):
        # 0-1, 1-2 ⇒ {0,1,2} reachable from any starting node
        constraints = [
            Coincident(0, 1, 1, 0),
            Coincident(1, 1, 2, 0),
        ]
        for start in (0, 1, 2):
            group = utils.get_connected_group(constraints, start_wall_idx=start)
            assert group == {0, 1, 2}

    def test_non_coincident_constraints_dont_group(self):
        from model.constraints import Length
        constraints = [Length(0, 5.0)]
        group = utils.get_connected_group(constraints, start_wall_idx=0)
        # LENGTH alone doesn't connect entities; only COINCIDENT does
        assert group == {0}


class TestGroupAnchored:
    def test_unanchored_group_returns_false(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((10, 0), (20, 0))
        assert utils.is_group_anchored(sketch.entities, {0, 1}) is False

    def test_one_anchored_endpoint_anchors_group(self, sketch):
        sketch.add_line((0, 0), (10, 0))
        sketch.add_line((10, 0), (20, 0))
        sketch.entities[0].anchored = [True, False]
        assert utils.is_group_anchored(sketch.entities, {0, 1}) is True

    def test_anchored_circle_anchors_group(self, sketch):
        sketch.add_circle((10, 10), 2.0)
        sketch.entities[0].anchored = [True]
        assert utils.is_group_anchored(sketch.entities, {0}) is True
