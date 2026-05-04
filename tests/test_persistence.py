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
        """Per TU-UI: round-trip the loaded view through CameraController so the
        camera's serialization contract is exercised end-to-end, not just
        dict-equality on the raw dict."""
        from core.camera import CameraController
        _prime_scene_with_particle(scene)
        view = {"zoom": 2.0, "pan_x": 100.0, "pan_y": 50.0}
        scene.save_scene(tmp_scene_path, view_state=view)
        _, loaded_view, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)

        cam = CameraController()
        cam.set_view_state(loaded_view)
        assert cam.zoom == 2.0
        assert cam.pan_x == 100.0
        assert cam.pan_y == 50.0

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

    def test_import_with_offset_remaps_coincident_tuple_indices(self, scene, tmp_model_path):
        """Coincident's (entity_idx, point_idx) tuple indices must be offset
        by base_idx during import — the _remap_constraint_indices tuple branch."""
        from model.constraints import Coincident
        scene.sketch.add_line((0, 0), (5, 0))
        scene.sketch.add_line((5, 0), (10, 0))
        scene.sketch.add_constraint_object(Coincident(0, 1, 1, 0), solve=False)
        scene.save_model(tmp_model_path)

        target = Scene(skip_warmup=True)
        target.sketch.add_line((100, 0), (101, 0))  # pre-existing — base_idx will be 1
        target.import_model(tmp_model_path, offset_x=20.0, offset_y=0.0)
        # Imported constraint indices should reference the new entity indices (1+0=1, 1+1=2)
        coinc = target.sketch.constraints[0]
        assert coinc.indices == [(1, 1), (2, 0)]


class TestPhysicalDynamicFlagsRoundTrip:
    def test_physical_flag_survives_scn_round_trip(self, scene, tmp_scene_path):
        scene.simulation._add_particle(1.0, 1.0)  # prime to avoid empty-restore xfail
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert loaded.sketch.entities[0].physical is True

    def test_dynamic_flag_survives_scn_round_trip(self, scene, tmp_scene_path):
        """Per TU-SIM: rebuild the compiler before saving so the saved file
        actually contains tethered atoms. This makes the test exercise both
        the entity-flag round-trip AND the saved-with-atoms scenario.
        (The downstream tether-linkage gap is captured by the load_scene
        topology_dirty xfail in this same file.)

        Note: Compiler auto-bumps dynamic-entity mass to ensure the entity is
        heavier than its atoms (see Compiler._compile_line ENTITY_MASS_MULTIPLIER).
        We capture the post-rebuild mass and round-trip *that*, since the user-
        supplied mass is overridden if too low.
        """
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.sketch.entities[0].dynamic = True
        scene.sketch.entities[0].mass = 25.0
        scene.rebuild()
        assert scene.simulation.count > 0
        # Capture the (possibly auto-bumped) mass after rebuild
        expected_mass = scene.sketch.entities[0].mass
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        line = loaded.sketch.entities[0]
        assert line.dynamic is True
        assert line.mass == expected_mass


class TestCoincidentInScnRoundTrip:
    def test_scn_load_preserves_coincident_tuple_indices(self, scene, tmp_scene_path):
        """Coincident's tuple indices must survive .scn save/load (via Sketch.restore)."""
        from model.constraints import Coincident
        scene.simulation._add_particle(1.0, 1.0)
        scene.sketch.add_line((0, 0), (5, 0))
        scene.sketch.add_line((5, 0), (10, 0))
        scene.sketch.add_constraint_object(Coincident(0, 1, 1, 0), solve=False)
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        c = loaded.sketch.constraints[0]
        assert c.type == "COINCIDENT"
        # Indices should be tuples (or lists treated as tuples)
        assert tuple(c.indices[0]) == (0, 1)
        assert tuple(c.indices[1]) == (1, 0)


class TestProcessObjectHandleIdentity:
    def test_handle_identity_after_load(self, scene, tmp_scene_path):
        """After load_scene, get_process_object_for_handle should resolve the handle
        back to its owning Source."""
        scene.simulation._add_particle(1.0, 1.0)
        scene.add_process_object(Source((25, 25), 3.0, SourceProperties()))
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        source = loaded.process_objects[0]
        center = source.handles["center"]
        assert loaded.get_process_object_for_handle(center) is source


class TestImportModelMaterialMerge:
    def test_import_brings_in_custom_materials(self, scene, tmp_model_path):
        """Per TU-SKETCH: also assert material *properties* round-tripped, not
        just the name registration. A buggy import that creates an empty
        Custom material with default fields would otherwise pass."""
        from model.properties import Material
        scene.sketch.materials["Custom"] = Material("Custom", color=(123, 45, 6))
        scene.sketch.add_line((0, 0), (5, 0), material_id="Custom")
        scene.save_model(tmp_model_path)

        target = Scene(skip_warmup=True)
        assert "Custom" not in target.sketch.materials
        target.import_model(tmp_model_path)
        assert "Custom" in target.sketch.materials
        assert tuple(target.sketch.materials["Custom"].color) == (123, 45, 6)


# ----- Latent bug: load_scene does not mark topology dirty -----------------

class TestLoadSceneTopologyDirty:
    def test_load_scene_marks_topology_dirty_for_physical_entities(self, scene, tmp_scene_path):
        """Fixed in CR-116 EI-3: Scene.load_scene() now sets _topology_dirty=True
        at the end of restoration. Simulation.to_dict() does not serialize tether
        arrays (tether_entity_idx, tether_local_pos, tether_stiffness), so atoms
        restored from disk have no entity coupling. The next update() runs
        rebuild() which re-establishes linkage from the restored sketch geometry."""
        scene.simulation._add_particle(1.0, 1.0)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.save_scene(tmp_scene_path)

        loaded, _, _ = Scene.load_scene(tmp_scene_path, skip_warmup=True)
        assert loaded._topology_dirty is True
        loaded.update(dt=0.016, geo_time=0.0, run_physics=False)
        # After rebuild, atoms have valid tether linkage
        sim = loaded.simulation
        valid_links = sum(1 for i in range(sim.count) if int(sim.tether_entity_idx[i]) >= 0)
        assert valid_links > 0
