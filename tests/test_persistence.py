"""Tests for Scene .scn / .mdl save / load round-trips."""

import json
import os
import pytest

from core.scene import Scene
from model.constraints import Length
from model.process_objects import Source, SourceProperties


@pytest.fixture
def tmp_scene_path(tmp_path):
    return str(tmp_path / "test.scn")


@pytest.fixture
def tmp_model_path(tmp_path):
    return str(tmp_path / "test.mdl")


def _prime_scene_with_particle(scene):
    """Helper: add at least one particle so save/load round-trips don't hit
    the empty-particle restore bug (captured separately in test_simulation.py).
    """
    scene.simulation._add_particle(1.0, 1.0)


class TestSceneSaveLoad:
    def test_save_creates_file(self, scene, tmp_scene_path):
        scene.sketch.add_line((0, 0), (10, 0))
        msg = scene.save_scene(tmp_scene_path)
        assert os.path.exists(tmp_scene_path)
        assert "saved" in msg.lower()

    def test_saved_file_is_valid_scene_format(self, scene, tmp_scene_path):
        scene.sketch.add_line((0, 0), (10, 0))
        scene.save_scene(tmp_scene_path)
        with open(tmp_scene_path) as f:
            data = json.load(f)
        assert data["type"] == "SCENE"
        assert "sketch" in data
        assert "simulation" in data
        assert "process_objects" in data

    def test_round_trip_preserves_entities(self, scene, tmp_scene_path):
        _prime_scene_with_particle(scene)
        scene.sketch.add_line((1.0, 2.0), (3.0, 4.0))
        scene.sketch.add_circle((5.0, 5.0), 2.5)
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert len(loaded.sketch.entities) == 2
        assert tuple(loaded.sketch.entities[0].start) == (1.0, 2.0)
        assert loaded.sketch.entities[1].radius == 2.5

    def test_round_trip_preserves_constraints(self, scene, tmp_scene_path):
        _prime_scene_with_particle(scene)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.add_constraint_object(Length(0, 5.0), solve=False)
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert len(loaded.sketch.constraints) == 1
        assert loaded.sketch.constraints[0].type == "LENGTH"

    def test_round_trip_preserves_process_objects(self, scene, tmp_scene_path):
        _prime_scene_with_particle(scene)
        scene.add_process_object(Source((25, 25), 3.0, SourceProperties(rate=15.0)))
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert len(loaded.process_objects) == 1
        source = loaded.process_objects[0]
        assert source.x == 25.0
        assert source.radius == 3.0
        assert source.properties.rate == 15.0

    def test_view_state_preserved(self, scene, tmp_scene_path):
        _prime_scene_with_particle(scene)
        view = {"zoom": 2.0, "pan_x": 100.0, "pan_y": 50.0}
        scene.save_scene(tmp_scene_path, view_state=view)
        _, loaded_view, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert loaded_view == view

    def test_load_invalid_format_returns_error(self, tmp_path):
        bad_path = tmp_path / "bad.scn"
        bad_path.write_text(json.dumps({"type": "NOT_A_SCENE"}))
        scene, view, msg = Scene.load_scene(str(bad_path), skip_warmup=True)
        assert scene is None
        assert "invalid" in msg.lower() or "format" in msg.lower()


class TestModelSaveImport:
    def test_save_model_creates_file(self, scene, tmp_model_path):
        scene.sketch.add_line((0, 0), (10, 0))
        scene.save_model(tmp_model_path)
        assert os.path.exists(tmp_model_path)

    def test_save_model_refuses_empty_sketch(self, scene, tmp_model_path):
        msg = scene.save_model(tmp_model_path)
        assert "no geometry" in msg.lower()

    def test_import_model_appends_geometry(self, scene, tmp_model_path):
        scene.sketch.add_line((0, 0), (10, 0))
        scene.save_model(tmp_model_path)

        target = Scene(skip_warmup=True)
        target.sketch.add_circle((20, 20), 1.0)  # pre-existing
        ok, _ = target.import_model(tmp_model_path)
        assert ok is True
        # Original circle plus imported line
        assert len(target.sketch.entities) == 2

    def test_import_with_offset_translates_geometry(self, scene, tmp_model_path):
        scene.sketch.add_line((0, 0), (10, 0))
        scene.save_model(tmp_model_path)

        target = Scene(skip_warmup=True)
        target.import_model(tmp_model_path, offset_x=100.0, offset_y=50.0)
        line = target.sketch.entities[0]
        assert tuple(line.start) == (100.0, 50.0)
        assert tuple(line.end) == (110.0, 50.0)
