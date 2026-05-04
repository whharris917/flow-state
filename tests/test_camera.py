"""Tests for core/camera.py — view transforms, zoom, pan, stored views."""

import pytest

import core.config as config
from core.camera import CameraController
from tests.conftest import make_layout


class TestDefaults:
    def test_initial_state(self):
        cam = CameraController()
        assert cam.zoom == 1.0
        assert cam.pan_x == 0.0
        assert cam.pan_y == 0.0


class TestCoordinateTransforms:
    def test_screen_world_inverse_at_default_view(self):
        cam = CameraController()
        layout = make_layout()
        # Pick an arbitrary screen point near the viewport center
        sx, sy = 600, 300
        wx, wy = cam.screen_to_world(sx, sy, world_size=50.0, layout=layout)
        rx, ry = cam.world_to_screen(wx, wy, world_size=50.0, layout=layout)
        assert rx == pytest.approx(sx, abs=1)
        assert ry == pytest.approx(sy, abs=1)

    def test_world_center_maps_to_viewport_center(self):
        cam = CameraController()
        layout = make_layout()
        # Center of a world (size=50) is (25, 25) — should land at viewport center
        sx, sy = cam.world_to_screen(25.0, 25.0, world_size=50.0, layout=layout)
        cx_screen = layout["MID_X"] + (layout["MID_W"] / 2.0)
        cy_screen = config.TOP_MENU_H + (layout["MID_H"] / 2.0)
        assert sx == pytest.approx(cx_screen, abs=1)
        assert sy == pytest.approx(cy_screen, abs=1)

    def test_pan_offsets_screen_position(self):
        cam = CameraController()
        layout = make_layout()
        s0 = cam.world_to_screen(25.0, 25.0, 50.0, layout)
        cam.apply_pan(50, 100)
        s1 = cam.world_to_screen(25.0, 25.0, 50.0, layout)
        assert s1[0] - s0[0] == 50
        assert s1[1] - s0[1] == 100

    def test_zoom_scales_screen_distance(self):
        cam = CameraController()
        layout = make_layout()
        # World point 1 unit from center
        s_at_1x = cam.world_to_screen(26.0, 25.0, 50.0, layout)
        cam.zoom = 2.0
        s_at_2x = cam.world_to_screen(26.0, 25.0, 50.0, layout)
        center_x = layout["MID_X"] + layout["MID_W"] / 2.0
        # The point's screen distance from center should double
        d_1x = abs(s_at_1x[0] - center_x)
        d_2x = abs(s_at_2x[0] - center_x)
        assert d_2x == pytest.approx(2 * d_1x, abs=1)


class TestRadiusConversion:
    def test_world_radius_inverse_screen_radius(self):
        cam = CameraController()
        layout = make_layout()
        screen_r = 25.0
        world_r = cam.get_world_radius(screen_r, 50.0, layout)
        round_trip = cam.get_screen_radius(world_r, 50.0, layout)
        assert round_trip == pytest.approx(screen_r, abs=0.01)


class TestZoomControls:
    def test_zoom_in_increases_zoom(self):
        cam = CameraController()
        cam.apply_zoom(1)  # positive => zoom in
        assert cam.zoom > 1.0

    def test_zoom_out_decreases_zoom(self):
        cam = CameraController()
        cam.apply_zoom(-1)
        assert cam.zoom < 1.0

    def test_zoom_clamped_to_min(self):
        cam = CameraController()
        for _ in range(100):
            cam.apply_zoom(-1, min_zoom=0.5)
        assert cam.zoom == 0.5

    def test_zoom_clamped_to_max(self):
        cam = CameraController()
        for _ in range(100):
            cam.apply_zoom(1, max_zoom=5.0)
        assert cam.zoom == 5.0


class TestPanControls:
    def test_pan_accumulates(self):
        cam = CameraController()
        cam.apply_pan(10, 20)
        cam.apply_pan(5, -5)
        assert cam.pan_x == 15
        assert cam.pan_y == 15


class TestStoredViews:
    def test_default_views_per_mode(self):
        """Constructor pre-populates SIM and EDITOR with sensible defaults.
        Per TU-UI: read defaults from the camera's own pre-populated dict
        rather than hardcoding 1.0/1.5, so cosmetic UX tuning doesn't break
        this test for the wrong reason."""
        cam = CameraController()
        sim_default = cam._stored_views[config.MODE_SIM]["zoom"]
        editor_default = cam._stored_views[config.MODE_EDITOR]["zoom"]
        cam.zoom = 999.0
        cam.restore_view(config.MODE_SIM)
        assert cam.zoom == sim_default
        cam.restore_view(config.MODE_EDITOR)
        assert cam.zoom == editor_default
        # And SIM and EDITOR should be different (otherwise the per-mode mechanism is broken)
        assert sim_default != editor_default

    def test_store_then_restore_round_trip(self):
        cam = CameraController()
        cam.zoom = 3.7
        cam.pan_x = 100
        cam.pan_y = 200
        cam.store_view(config.MODE_SIM)

        # Move away
        cam.zoom = 0.5
        cam.pan_x = 0
        cam.pan_y = 0

        cam.restore_view(config.MODE_SIM)
        assert cam.zoom == 3.7
        assert cam.pan_x == 100
        assert cam.pan_y == 200


class TestSerialization:
    def test_view_state_dict_round_trip(self):
        cam = CameraController()
        cam.zoom = 2.5
        cam.pan_x = 10
        cam.pan_y = 20
        d = cam.get_view_state()

        cam2 = CameraController()
        cam2.set_view_state(d)
        assert cam2.zoom == 2.5
        assert cam2.pan_x == 10
        assert cam2.pan_y == 20

    def test_set_view_state_with_none_is_safe(self):
        cam = CameraController()
        cam.zoom = 2.0
        cam.set_view_state(None)
        # Should not raise; existing values preserved
        assert cam.zoom == 2.0


class TestReset:
    def test_reset_zeroes_state(self):
        cam = CameraController()
        cam.zoom = 5.0
        cam.pan_x = 100
        cam.pan_y = 200
        cam.reset()
        assert cam.zoom == 1.0
        assert cam.pan_x == 0.0
        assert cam.pan_y == 0.0
