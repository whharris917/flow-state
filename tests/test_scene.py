"""Tests for core/scene.py — orchestration, dirty flags, ProcessObject management."""

import numpy as np
import pytest

from core.scene import Scene
from core.commands import AddLineCommand, MoveEntityCommand, SetPhysicalCommand
from model.geometry import Line, Point
from model.process_objects import Source, SourceProperties


class TestSceneOwnership:
    def test_scene_owns_sketch_simulation_compiler_commands(self, scene):
        assert scene.sketch is not None
        assert scene.simulation is not None
        assert scene.compiler is not None
        assert scene.commands is not None

    def test_alias_properties_delegate_to_sketch(self, scene):
        scene.sketch.add_line((0, 0), (1, 0))
        assert scene.entities is scene.sketch.entities
        assert scene.constraints is scene.sketch.constraints
        assert scene.materials is scene.sketch.materials


class TestDirtyFlags:
    def test_topology_changing_command_sets_topology_dirty(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        assert scene._topology_dirty is True

    def test_topology_dirty_flag_drives_rebuild_through_update(self, scene):
        """End-to-end: execute(topology cmd) → update() → flag cleared AND
        atoms emitted. Per TU-SCENE: the dirty-flag tests verify the entry
        point only; this test verifies the *consumption* side as well."""
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        assert scene._topology_dirty is False
        assert scene.simulation.count > 0

    def test_non_topology_changing_command_sets_geometry_dirty(self, scene):
        scene.sketch.add_line((0, 0), (1, 0))
        scene._topology_dirty = False
        scene._geometry_dirty = False
        scene.execute(MoveEntityCommand(scene.sketch, 0, 1.0, 0.0))
        # MoveEntityCommand has no changes_topology attribute => geometry_dirty
        assert scene._geometry_dirty is True
        assert scene._topology_dirty is False

    def test_undo_marks_topology_dirty_and_runs_rebuild(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene._topology_dirty = False
        scene.undo()
        # undo() conservatively marks topology dirty (so the next update() picks
        # it up) and calls rebuild() immediately. The flag is left True; only
        # update() resets it.
        assert scene._topology_dirty is True
        assert scene.sketch.entities == []


class TestUndoRedo:
    def test_can_undo_after_execute(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        assert scene.can_undo() is True

    def test_can_redo_after_undo(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.undo()
        assert scene.can_redo() is True

    def test_redo_restores_entity(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (5, 5)))
        scene.undo()
        scene.redo()
        assert len(scene.sketch.entities) == 1


class TestUpdateLoop:
    def test_update_with_empty_scene_does_not_crash(self, scene):
        # Empty sketch + simulation paused => update is a no-op
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)

    def test_update_clears_topology_dirty_flag_after_rebuild(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0)))
        scene.execute(SetPhysicalCommand(scene.sketch, 0, True))
        assert scene._topology_dirty is True
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        assert scene._topology_dirty is False

    def test_update_with_physical_line_emits_atoms(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        # Compiler emitted static atoms during the topology rebuild
        static_count = int(np.sum(scene.simulation.is_static[:scene.simulation.count] == 1))
        assert static_count > 0

    def test_paused_simulation_does_not_advance(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=True)
        # Sim is paused by default — no integration steps should run
        assert scene.simulation.total_steps == 0


class TestProcessObjects:
    def test_add_process_object_registers_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert source in scene.process_objects
        # Handle (center Point) should now appear in the sketch
        assert source.handles["center"] in scene.sketch.entities

    def test_remove_process_object_unregisters_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        scene.remove_process_object(source)
        assert source not in scene.process_objects
        assert source.handles["center"] not in scene.sketch.entities

    def test_get_process_object_for_handle(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        center = source.handles["center"]
        assert scene.get_process_object_for_handle(center) is source

    def test_get_process_object_for_non_handle_returns_none(self, scene):
        non_handle = Point(0.0, 0.0)
        assert scene.get_process_object_for_handle(non_handle) is None

    def test_find_process_object_at_center(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert scene.find_process_object_at(25.0, 25.0) is source

    def test_find_process_object_at_returns_none_when_far(self, scene):
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        assert scene.find_process_object_at(100.0, 100.0) is None


class TestSinkIntegration:
    """Sink uses the same Scene plumbing as Source — verify end-to-end."""

    def test_add_sink_registers_handle(self, scene):
        from model.process_objects import Sink, SinkProperties
        sink = Sink((25.0, 25.0), 3.0, SinkProperties())
        scene.add_process_object(sink)
        assert sink in scene.process_objects
        assert sink.handles["center"] in scene.sketch.entities

    def test_remove_sink_unregisters_handle(self, scene):
        from model.process_objects import Sink
        sink = Sink((25.0, 25.0), 3.0)
        scene.add_process_object(sink)
        scene.remove_process_object(sink)
        assert sink not in scene.process_objects
        assert sink.handles["center"] not in scene.sketch.entities

    def test_find_process_object_at_resolves_sink(self, scene):
        from model.process_objects import Sink
        sink = Sink((25.0, 25.0), 3.0)
        scene.add_process_object(sink)
        assert scene.find_process_object_at(25.0, 25.0) is sink

    def test_add_sink_command_undo_redo(self, scene):
        """AddSinkCommand round-trips through undo and redo."""
        from core.sink_commands import AddSinkCommand
        cmd = AddSinkCommand(scene=scene, center=(25, 25), radius=3.0)
        scene.execute(cmd)

        sink = cmd.sink
        assert sink in scene.process_objects
        assert sink.handles["center"] in scene.sketch.entities

        scene.undo()
        assert sink not in scene.process_objects
        assert sink.handles["center"] not in scene.sketch.entities

        scene.redo()
        assert sink in scene.process_objects
        assert sink.handles["center"] in scene.sketch.entities

    def test_delete_sink_command_round_trip(self, scene):
        """DeleteSinkCommand preserves sink data through undo (re-creates from dict)."""
        from model.process_objects import Sink, SinkProperties
        from core.sink_commands import DeleteSinkCommand

        sink = Sink((30, 30), 4.0, SinkProperties())
        scene.add_process_object(sink)

        cmd = DeleteSinkCommand(scene=scene, sink=sink)
        scene.execute(cmd)
        assert sink not in scene.process_objects

        scene.undo()
        # Restored sink is a fresh instance reconstructed from to_dict()
        assert len(scene.process_objects) == 1
        restored = scene.process_objects[0]
        assert restored.x == 30.0
        assert restored.y == 30.0
        assert restored.radius == 4.0

    def test_source_and_sink_coexist(self, scene):
        """Both ProcessObject types can live in the scene simultaneously."""
        from model.process_objects import Sink
        source = Source((10, 10), 2.0, SourceProperties())
        sink = Sink((40, 40), 3.0)
        scene.add_process_object(source)
        scene.add_process_object(sink)

        assert source in scene.process_objects
        assert sink in scene.process_objects
        assert scene.find_process_object_at(10, 10) is source
        assert scene.find_process_object_at(40, 40) is sink

    def test_set_sink_radius_undo_redo(self, scene):
        """SetSinkRadiusCommand round-trips through undo/redo."""
        from model.process_objects import Sink
        from core.sink_commands import SetSinkRadiusCommand

        sink = Sink((25, 25), 3.0)
        scene.add_process_object(sink)

        cmd = SetSinkRadiusCommand(sink, new_radius=7.5, old_radius=3.0)
        scene.execute(cmd)
        assert sink.radius == 7.5

        scene.undo()
        assert sink.radius == 3.0

        scene.redo()
        assert sink.radius == 7.5

    def test_set_sink_radius_supersede_merge(self, scene):
        """Supersede merge collapses sequential SetSinkRadius into one undo step."""
        from model.process_objects import Sink
        from core.sink_commands import SetSinkRadiusCommand

        sink = Sink((25, 25), 3.0)
        scene.add_process_object(sink)

        # Initial command (anchor for supersede chain)
        scene.execute(SetSinkRadiusCommand(sink, 3.0, old_radius=3.0,
                                           historize=True, supersede=False))
        # Drag steps
        for r in [4.0, 5.0, 6.0, 7.5]:
            scene.execute(SetSinkRadiusCommand(sink, r, old_radius=3.0,
                                               historize=True, supersede=True))
        assert sink.radius == 7.5

        # Single undo should restore the original radius despite many drag steps
        scene.undo()
        assert sink.radius == 3.0


def _make_app_controller_stub(scene):
    """Build a minimal AppController bound to `scene` for action-layer tests.

    Skips `__init__` (which expects a full FlowStateApp) and wires only the
    surface that action_delete_selection / apply_source_properties_from_dialog
    actually read: session.selection, session.status, sound_manager, scene,
    sketch (via @property → app.scene.sketch), modal stack, ctx_vars.
    """
    from app.app_controller import AppController

    class _StubSession:
        def __init__(self):
            from core.selection import SelectionManager
            from core.status_bar import StatusBar
            self.selection = SelectionManager()
            self.status = StatusBar()

    class _StubApp:
        pass

    app = _StubApp()
    app.scene = scene
    app.session = _StubSession()
    app.input_handler = None

    ctrl = AppController.__new__(AppController)
    ctrl.app = app
    from core.sound_manager import SoundManager
    ctrl.sound_manager = SoundManager.get()
    ctrl._modal_stack = []
    ctrl.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}
    return ctrl


class TestProcessObjectHandleCascadeDelete:
    """Deleting a handle Point must cascade-delete its owning ProcessObject."""

    def _make_app_controller_stub(self, scene):
        """Per-class shim — delegates to the module-level helper."""
        return _make_app_controller_stub(scene)

    def test_delete_source_handle_cascades(self, scene):
        """Selecting the center handle of a Source and deleting it removes the Source."""
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        handle = source.handles['center']
        handle_idx = scene.sketch.entities.index(handle)

        ctrl = self._make_app_controller_stub(scene)
        ctrl.app.session.selection.walls.add(handle_idx)
        ctrl.action_delete_selection()

        # Source removed from scene
        assert source not in scene.process_objects
        # Handle Point removed from sketch
        assert handle not in scene.sketch.entities

    def test_delete_sink_handle_cascades(self, scene):
        """Same cascade for Sink handles."""
        from model.process_objects import Sink
        sink = Sink((30.0, 30.0), 4.0)
        scene.add_process_object(sink)
        handle = sink.handles['center']
        handle_idx = scene.sketch.entities.index(handle)

        ctrl = self._make_app_controller_stub(scene)
        ctrl.app.session.selection.walls.add(handle_idx)
        ctrl.action_delete_selection()

        assert sink not in scene.process_objects
        assert handle not in scene.sketch.entities

    def test_delete_handle_cascade_is_undoable(self, scene):
        """Cascade-delete via DeleteSourceCommand undoes back to the live Source."""
        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        handle_idx = scene.sketch.entities.index(source.handles['center'])

        ctrl = self._make_app_controller_stub(scene)
        ctrl.app.session.selection.walls.add(handle_idx)
        ctrl.action_delete_selection()

        scene.undo()
        # After undo, a Source is back in the scene (reconstructed from to_dict)
        assert len(scene.process_objects) == 1
        restored = scene.process_objects[0]
        assert restored.x == 25.0
        assert restored.y == 25.0
        assert restored.radius == 3.0

    def test_non_handle_point_deletes_normally(self, scene):
        """A regular Point (not a handle) still goes through RemoveEntityCommand."""
        from model.geometry import Point
        regular_point = Point(10.0, 10.0)
        scene.sketch.entities.append(regular_point)
        idx = scene.sketch.entities.index(regular_point)

        ctrl = self._make_app_controller_stub(scene)
        ctrl.app.session.selection.walls.add(idx)
        ctrl.action_delete_selection()

        assert regular_point not in scene.sketch.entities
        # No process_object touched
        assert scene.process_objects == []

    def test_mixed_selection_handles_and_entities(self, scene):
        """A selection containing both a handle and a regular line: both are removed."""
        from model.commands.geometry import AddLineCommand
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        line_idx = len(scene.sketch.entities) - 1

        source = Source((25.0, 25.0), 3.0, SourceProperties())
        scene.add_process_object(source)
        handle = source.handles['center']
        handle_idx = scene.sketch.entities.index(handle)

        ctrl = self._make_app_controller_stub(scene)
        ctrl.app.session.selection.walls.add(line_idx)
        ctrl.app.session.selection.walls.add(handle_idx)
        ctrl.action_delete_selection()

        # Both gone
        assert source not in scene.process_objects
        assert handle not in scene.sketch.entities
        # And the line entity is also gone (composite delete completed)
        assert len(scene.sketch.entities) == 0


class TestSourcePropertiesDialog:
    """End-to-end coverage of the dialog → command → applied-state pipeline.

    The dialog itself is constructed with real Dropdown/InputField/Button
    widgets (pygame is initialized via conftest's session-scoped fixture).
    Tests drive the widgets programmatically rather than via pygame events,
    then invoke `apply_source_properties_from_dialog` directly to exercise
    the command pipeline.
    """

    def _make_source(self, scene, *, center=(25.0, 25.0), radius=3.0,
                     material_name='Water', flux=0.5, temperature=1.0):
        """Create a Source, attach it to the scene, return it."""
        source = Source(
            center, radius,
            SourceProperties(material_name=material_name,
                             flux=flux, temperature=temperature),
        )
        scene.add_process_object(source)
        return source

    def _open_dialog(self, source, sketch):
        """Construct a SourcePropertiesDialog at (0,0)."""
        from ui.ui_widgets import SourcePropertiesDialog
        return SourcePropertiesDialog(0, 0, source, sketch)

    def _set_dropdown(self, dropdown, value):
        """Programmatically select a Dropdown option by string value."""
        dropdown.selected_index = dropdown.options.index(value)

    # ---- seed verification --------------------------------------------------

    def test_dialog_seeds_inputs_from_source_state(self, scene):
        source = self._make_source(scene, radius=4.25, material_name='Oil',
                                   flux=2.5, temperature=0.7)
        dialog = self._open_dialog(source, scene.sketch)

        assert dialog.dropdown.get_selected() == 'Oil'
        assert dialog.in_radius.text == '4.25'
        # Flux formatted to 3 dp (typical values are in the 0.1–10 range)
        assert dialog.in_flux.text == '2.500'
        assert dialog.in_temp.text == '0.70'

    def test_dialog_lists_all_sketch_materials_in_dropdown(self, scene):
        source = self._make_source(scene)
        dialog = self._open_dialog(source, scene.sketch)
        # All preset materials should appear; user-added materials would too
        for name in ['Water', 'Oil', 'Mercury', 'Honey', 'Wall']:
            assert name in dialog.dropdown.options

    def test_dialog_with_unknown_material_falls_back_to_first_option(self, scene):
        """If the Source carries a material_name that isn't in the sketch's
        library, the dropdown falls back to index 0 so the user can pick a
        valid option instead of seeing a blank."""
        source = self._make_source(scene, material_name='Ghost')
        dialog = self._open_dialog(source, scene.sketch)
        # selected_index defaulted to 0 in the dialog's ValueError handler
        assert dialog.dropdown.selected_index == 0

    # ---- get_values reads from current widget state ------------------------

    def test_get_values_reads_current_inputs(self, scene):
        source = self._make_source(scene)
        dialog = self._open_dialog(source, scene.sketch)
        self._set_dropdown(dialog.dropdown, 'Honey')
        dialog.in_radius.set_value(8.0)
        dialog.in_flux.set_value(3.5)
        dialog.in_temp.set_value(2.25)

        vals = dialog.get_values()
        assert vals['material_name'] == 'Honey'
        assert vals['radius'] == 8.0
        assert vals['flux'] == 3.5
        assert vals['temperature'] == 2.25

    def test_get_values_clamps_radius_to_min(self, scene):
        """Radius is clamped to 0.5 to match the placement-tool floor."""
        source = self._make_source(scene)
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_radius.set_value(0.1)
        vals = dialog.get_values()
        assert vals['radius'] == 0.5

    def test_get_values_clamps_flux_and_temp_non_negative(self, scene):
        """Flux and temperature can't go negative."""
        source = self._make_source(scene)
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_flux.set_value(-5.0)
        dialog.in_temp.set_value(-1.5)
        vals = dialog.get_values()
        assert vals['flux'] == 0.0
        assert vals['temperature'] == 0.0

    # ---- apply path (per-property) -----------------------------------------

    def _apply(self, scene, source, dialog):
        """Helper: mark dialog as applied and route through AppController."""
        dialog.apply = True
        dialog.done = True
        ctrl = _make_app_controller_stub(scene)
        ctrl.apply_source_properties_from_dialog(dialog)

    def test_apply_changes_material_name(self, scene):
        source = self._make_source(scene, material_name='Water')
        dialog = self._open_dialog(source, scene.sketch)
        self._set_dropdown(dialog.dropdown, 'Mercury')
        self._apply(scene, source, dialog)
        assert source.properties.material_name == 'Mercury'

    def test_apply_changes_radius(self, scene):
        source = self._make_source(scene, radius=3.0)
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_radius.set_value(9.5)
        self._apply(scene, source, dialog)
        assert source.radius == 9.5

    def test_apply_changes_flux(self, scene):
        source = self._make_source(scene, flux=0.5)
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_flux.set_value(2.5)
        self._apply(scene, source, dialog)
        assert source.properties.flux == 2.5

    def test_apply_changes_temperature(self, scene):
        source = self._make_source(scene, temperature=1.0)
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_temp.set_value(3.75)
        self._apply(scene, source, dialog)
        assert source.properties.temperature == 3.75

    def test_apply_preserves_unchanged_injection_direction_and_spread(self, scene):
        """The dialog doesn't expose injection_direction/spread; those must
        round-trip unchanged through the apply pipeline (not get reset to defaults)."""
        import math
        source = self._make_source(scene)
        source.properties.injection_direction = math.pi / 3
        source.properties.injection_spread = 0.4
        dialog = self._open_dialog(source, scene.sketch)
        dialog.in_flux.set_value(2.0)
        self._apply(scene, source, dialog)

        assert source.properties.flux == 2.0
        assert source.properties.injection_direction == math.pi / 3
        assert source.properties.injection_spread == 0.4

    # ---- apply path (composite undo) ---------------------------------------

    def test_apply_with_all_changes_collapses_to_one_undo_step(self, scene):
        """All four changes (material, radius, flux, temp) bundle into one
        CompositeCommand — a single undo reverts everything to pre-dialog state."""
        source = self._make_source(scene, radius=3.0, material_name='Water',
                                   flux=0.5, temperature=1.0)
        dialog = self._open_dialog(source, scene.sketch)
        self._set_dropdown(dialog.dropdown, 'Honey')
        dialog.in_radius.set_value(7.0)
        dialog.in_flux.set_value(3.0)
        dialog.in_temp.set_value(2.0)
        self._apply(scene, source, dialog)

        # Sanity: all four changed
        assert source.radius == 7.0
        assert source.properties.material_name == 'Honey'
        assert source.properties.flux == 3.0
        assert source.properties.temperature == 2.0

        # One undo restores everything
        scene.undo()
        assert source.radius == 3.0
        assert source.properties.material_name == 'Water'
        assert source.properties.flux == 0.5
        assert source.properties.temperature == 1.0

    def test_apply_with_only_property_change_emits_single_command(self, scene):
        """When radius is unchanged, only SetSourcePropertiesCommand is emitted
        — no superfluous SetSourceRadiusCommand for a no-op."""
        source = self._make_source(scene, radius=3.0)
        dialog = self._open_dialog(source, scene.sketch)
        # Don't touch radius input; change flux only
        dialog.in_flux.set_value(2.0)
        self._apply(scene, source, dialog)

        # One undo restores the property change
        scene.undo()
        assert source.properties.flux == 0.5
        assert source.radius == 3.0  # unchanged throughout

    # ---- cancel paths ------------------------------------------------------

    def test_cancel_via_apply_flag_makes_no_changes(self, scene):
        """When dialog.apply is False, the source state must be untouched even
        if dialog.done is True (this is the click-outside-dismiss path)."""
        source = self._make_source(scene, radius=3.0, material_name='Water',
                                   flux=0.5, temperature=1.0)
        dialog = self._open_dialog(source, scene.sketch)
        self._set_dropdown(dialog.dropdown, 'Mercury')
        dialog.in_radius.set_value(99.0)
        dialog.in_flux.set_value(99.0)

        # Simulate Cancel button: apply=False, done=True
        dialog.apply = False
        dialog.cancelled = True
        dialog.done = True
        ctrl = _make_app_controller_stub(scene)
        ctrl.apply_source_properties_from_dialog(dialog)

        # All four pre-dialog values are intact
        assert source.radius == 3.0
        assert source.properties.material_name == 'Water'
        assert source.properties.flux == 0.5
        assert source.properties.temperature == 1.0

    def test_apply_then_cancel_subsequent_dialog_independent(self, scene):
        """Two dialog sessions in sequence: apply the first, cancel the second.
        Second cancellation must not undo the first apply."""
        source = self._make_source(scene, flux=0.5)

        # Dialog 1: apply flux=2.0
        d1 = self._open_dialog(source, scene.sketch)
        d1.in_flux.set_value(2.0)
        self._apply(scene, source, d1)
        assert source.properties.flux == 2.0

        # Dialog 2: change flux, then cancel
        d2 = self._open_dialog(source, scene.sketch)
        d2.in_flux.set_value(99.0)
        d2.apply = False
        d2.done = True
        ctrl = _make_app_controller_stub(scene)
        ctrl.apply_source_properties_from_dialog(d2)

        # First apply still in effect
        assert source.properties.flux == 2.0

    # ---- behavioral effect of apply ---------------------------------------

    def test_apply_material_change_affects_subsequent_spawned_particles(self, scene):
        """The whole point: switching material in the dialog must change what
        new particles look like physically. Verify via emitted-particle sigma."""
        source = self._make_source(scene, material_name='Water', flux=100.0)
        scene.simulation.world_size = 50.0

        dialog = self._open_dialog(source, scene.sketch)
        self._set_dropdown(dialog.dropdown, 'Mercury')  # sigma=0.8
        self._apply(scene, source, dialog)

        source.execute(scene.simulation, dt=1.0)
        assert scene.simulation.count > 0
        mercury_sigma = scene.sketch.get_material('Mercury').sigma
        # All spawned dynamic particles carry Mercury's sigma now
        for i in range(scene.simulation.count):
            if scene.simulation.is_static[i] == 0:
                assert abs(float(scene.simulation.atom_sigma[i]) - mercury_sigma) < 1e-5

    def test_flux_is_intensive_effective_rate_scales_with_area(self, scene):
        """Core conceptual win: flux is intensive — same flux gives more
        target throughput on a bigger source, like pressure or chemical
        potential. The effective spawn rate is `flux · π · r²` so doubling
        the radius quadruples target throughput at fixed flux.

        Tested at two layers:

        1. *Target rate* (deterministic): the accumulator grows by exactly
           `flux · π · r² · dt` each call. Verifies the math directly,
           bypassing the per-frame attempt cap and rejection-sampling noise.
        2. *Delivered count* (statistical): over 20 frames, the large source
           delivers >3× the spawns of the small source — directional
           confirmation that the throughput difference survives the cap.
        """
        import math as m, random, numpy as np

        # ---- Layer 1: target effective rate is exactly intensive ----
        s_small = Source((0, 0), 2.0, SourceProperties(flux=2.0))
        s_large = Source((0, 0), 6.0, SourceProperties(flux=2.0))
        small_eff = s_small.properties.flux * m.pi * s_small.radius ** 2
        large_eff = s_large.properties.flux * m.pi * s_large.radius ** 2
        # 9× area ratio (r²=4 vs 36) ⇒ exactly 9× effective rate
        assert large_eff == pytest.approx(9 * small_eff)

        # ---- Layer 2: 20-frame delivered count reflects the difference ----
        scene.simulation.world_size = 200.0
        random.seed(7); np.random.seed(7)

        s_small = Source((50, 50), 2.0, SourceProperties(flux=2.0))
        scene.add_process_object(s_small)
        for _ in range(20):
            s_small.execute(scene.simulation, dt=0.05)
        small_count = scene.simulation.count
        scene.simulation.clear()
        scene.remove_process_object(s_small)

        s_large = Source((150, 50), 6.0, SourceProperties(flux=2.0))
        scene.add_process_object(s_large)
        for _ in range(20):
            s_large.execute(scene.simulation, dt=0.05)
        large_count = scene.simulation.count

        assert large_count > 3 * small_count, (
            f"flux=2.0 r=6 produced {large_count} particles, "
            f"flux=2.0 r=2 produced {small_count} — intensive flux should "
            f"scale total throughput by ~9× area ratio (per-frame attempt "
            f"cap brings the observable ratio below 9×, but well above 3×)"
        )

    def test_apply_temperature_change_affects_velocity_sampling(self, scene):
        """Doubling temperature via the dialog must roughly sqrt(2) the
        velocity spread of spawned particles (Maxwell-Boltzmann: std ∝ sqrt(T/m)).
        Verified by comparing mean speeds with the same seed."""
        import numpy as np
        import random
        scene.simulation.world_size = 200.0

        # Cold Source
        random.seed(123); np.random.seed(123)
        s = self._make_source(scene, center=(100, 100), radius=20.0,
                              flux=2.0, temperature=1.0)
        for _ in range(3):
            s.execute(scene.simulation, dt=0.5)
        cold_speeds = np.hypot(scene.simulation.vel_x[:scene.simulation.count],
                                scene.simulation.vel_y[:scene.simulation.count])
        cold_mean = float(np.mean(cold_speeds))

        # Bump temperature to 4.0 via dialog (expected speed scale: 2x cold).
        # Reset everything else for a fair second measurement.
        scene.simulation.clear()
        s._spawn_accumulator = 0.0
        s._recent_spawns = []
        random.seed(123); np.random.seed(123)

        dialog = self._open_dialog(s, scene.sketch)
        dialog.in_temp.set_value(4.0)
        self._apply(scene, s, dialog)

        for _ in range(3):
            s.execute(scene.simulation, dt=0.5)
        hot_speeds = np.hypot(scene.simulation.vel_x[:scene.simulation.count],
                               scene.simulation.vel_y[:scene.simulation.count])
        hot_mean = float(np.mean(hot_speeds))

        # T=4 vs T=1 means speed std ratio ≈ 2.0. Loose bound (>1.5)
        # accommodates sampling noise even with seeded randomness.
        assert hot_mean > 1.5 * cold_mean, (
            f"hot mean speed {hot_mean:.2f} should be >1.5× cold {cold_mean:.2f}"
        )


class TestSceneClearAndNew:
    def test_clear_removes_everything(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.add_process_object(Source((10, 10), 2.0, SourceProperties()))
        scene.clear()
        assert scene.sketch.entities == []
        assert scene.simulation.count == 0
        assert scene.process_objects == []

    def test_new_resets_to_empty_state(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.new()
        assert scene.sketch.entities == []
        assert scene.simulation.count == 0
        assert scene.commands.can_undo() is False


class TestBrushDelegation:
    def test_paint_particles_delegates_to_brush(self, scene):
        added = scene.paint_particles(25.0, 25.0, radius=2.0)
        assert added > 0
        assert scene.simulation.count == added

    def test_erase_particles_delegates_to_brush(self, scene):
        scene.paint_particles(25.0, 25.0, radius=2.0)
        before = scene.simulation.count
        removed = scene.erase_particles(25.0, 25.0, radius=2.0)
        assert removed > 0
        assert scene.simulation.count < before


# ----- Update loop ordering / dirty flag transitions ----------------------

class TestUpdateLoopOrdering:
    def test_topology_rebuild_clears_geometry_dirty(self, scene):
        """Topology rebuild covers sync, so _geometry_dirty should also clear."""
        scene._topology_dirty = True
        scene._geometry_dirty = True
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        assert scene._topology_dirty is False
        assert scene._geometry_dirty is False

    def test_geometry_only_path_does_not_call_rebuild(self, scene):
        """When only _geometry_dirty is set, Compiler.rebuild must NOT be called.
        That's the whole point of the fast-path optimization."""
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        # First update consumes the topology-dirty
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        # Now move (geometry-only)
        scene.execute(MoveEntityCommand(scene.sketch, 0, 1.0, 0.0))
        assert scene._topology_dirty is False
        assert scene._geometry_dirty is True

        rebuild_calls = []
        original_rebuild = scene.compiler.rebuild
        def spy(*args, **kwargs):
            rebuild_calls.append(1)
            return original_rebuild(*args, **kwargs)
        scene.compiler.rebuild = spy
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        # Rebuild should NOT have been called on the fast path
        assert rebuild_calls == []

    def test_interaction_data_alone_triggers_solve(self, scene):
        """Even with no constraints, an active User Servo must trigger solve().

        Per TU-UI cross-domain note: this test is a Scene-level orchestration
        check (the Scene is allowed to mutate sketch.interaction_data — the Air
        Gap restriction is on tools, not on Scene). The corresponding
        tool-level facade contract is exercised in test_tool_context.py
        (set_interaction_data writes the four documented keys).
        """
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0)))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        scene.sketch.interaction_data = {
            "entity_idx": 0, "point_idx": 1, "handle_t": None, "target": (5.0, 5.0),
        }
        end_before = scene.sketch.entities[0].end.copy()
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        assert tuple(scene.sketch.entities[0].end) != tuple(end_before)


class TestSceneExecuteOnFailure:
    def test_failing_command_does_not_mark_dirty_flags(self, scene):
        """Scene.execute(failing_cmd) must leave dirty flags untouched."""
        scene._topology_dirty = False
        scene._geometry_dirty = False

        # A command that returns False from execute()
        from core.command_base import Command

        class FailingCommand(Command):
            changes_topology = True
            def execute(self): return False
            def undo(self): pass

        result = scene.execute(FailingCommand())
        assert result is False
        # Flags must remain False
        assert scene._topology_dirty is False
        assert scene._geometry_dirty is False


# ----- Scene undo/redo causes immediate rebuild ----------------------------

class TestUndoRedoEagerRebuild:
    def test_undo_eagerly_rebuilds_atoms(self, scene):
        """Scene.undo() calls rebuild() before returning — even before update()."""
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        # Confirm atoms were emitted
        before_count = scene.simulation.count
        assert before_count > 0

        scene.undo()
        # After undo, the entity is gone; rebuild was called eagerly so atoms are gone too
        assert scene.simulation.count == 0

    def test_redo_eagerly_rebuilds_atoms(self, scene):
        """TU-SCENE refinement: assert the intermediate state (count == 0 after undo)
        so we isolate eager rebuild from incidental atom-leftover scenarios."""
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0), physical=True))
        scene.update(dt=0.016, geo_time=0.0, run_physics=False)
        atom_count = scene.simulation.count
        scene.undo()
        # Snapshot intermediate: undo's eager rebuild cleared all atoms
        assert scene.simulation.count == 0
        scene.redo()
        # Redo's eager rebuild should restore the atoms
        assert scene.simulation.count == atom_count

    def test_discard_cannot_be_redone(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (10, 0)))
        scene.discard()
        assert scene.can_redo() is False


# ----- ProcessObject lifecycle through clear / new --------------------------

class TestProcessObjectLifecycle:
    def test_clear_then_add_process_object_works(self, scene):
        """After clear(), adding a fresh ProcessObject should succeed."""
        scene.add_process_object(Source((25, 25), 3.0, SourceProperties()))
        scene.clear()
        # Add a new one — should not have stale state
        new_source = Source((10, 10), 2.0, SourceProperties())
        scene.add_process_object(new_source)
        assert new_source in scene.process_objects
        assert new_source._owner_scene is scene

    def test_new_clears_redo_stack(self, scene):
        scene.execute(AddLineCommand(scene.sketch, (0, 0), (1, 0)))
        scene.undo()
        assert scene.can_redo() is True
        scene.new()
        assert scene.can_redo() is False

    def test_removing_process_object_does_not_remove_unrelated_points(self, scene):
        """When a ProcessObject is removed, only its handles should leave the
        sketch — hand-placed Points must survive."""
        scene.sketch.entities.append(Point(5, 5))
        source = Source((25, 25), 3.0, SourceProperties())
        scene.add_process_object(source)
        # Now sketch has: hand-placed Point, source.handle.center
        scene.remove_process_object(source)
        # Hand-placed Point must survive
        assert any(isinstance(e, Point) and not getattr(e, "is_handle", False)
                   for e in scene.sketch.entities)


# ----- Two-way coupling kernel (TU-SIM contributions to scene tests) -------

class TestTwoWayCoupling:
    def test_tether_force_pulls_atom_toward_anchor(self):
        """A tethered atom displaced from its anchor must feel a force pulling
        it back; the parent entity must accumulate an equal-and-opposite reaction."""
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.sketch.entities[0].dynamic = True
        scene.rebuild()

        # Pick a tethered atom and displace it perpendicular to the line
        sim = scene.simulation
        idx = next(i for i in range(sim.count) if sim.is_static[i] == 3)
        sim.pos_y[idx] += 1.0  # off the line in +y direction

        sim.clear_entity_forces()
        sim.apply_tether_forces()

        # Force on atom should pull -y (back to anchor)
        assert sim.force_y[idx] < 0
        # Reaction on entity should be +y (Newton's third law)
        forces = sim.get_entity_forces()
        assert forces[0, 1] > 0

    def test_tether_force_on_undisplaced_atom_is_zero(self):
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.sketch.entities[0].dynamic = True
        scene.rebuild()

        sim = scene.simulation
        sim.clear_entity_forces()
        sim.apply_tether_forces()

        # No displacement → no force
        idx = next(i for i in range(sim.count) if sim.is_static[i] == 3)
        assert abs(sim.force_x[idx]) < 1e-3
        assert abs(sim.force_y[idx]) < 1e-3

    def test_tether_torque_sign_on_dynamic_line(self):
        """Push a tethered atom at a known parametric position perpendicular
        to the line and verify the entity's torque accumulator has the predicted
        sign per right-hand rule. TU-SIM refinement: assert the *sign*, not just
        nonzero, so a flipped cross-product convention regression is caught."""
        from core.scene import Scene
        import math
        scene = Scene(skip_warmup=True)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.sketch.entities[0].dynamic = True
        scene.rebuild()

        sim = scene.simulation
        # Find an atom near the right end (high t value) — lever arm is +x relative to COM
        candidates = [i for i in range(sim.count)
                      if sim.is_static[i] == 3 and sim.tether_local_pos[i, 0] > 0.7]
        assert candidates, "Need a tethered atom near the right end"
        idx = candidates[0]

        # Capture lever arm before displacement: r = anchor - COM
        com = scene.sketch.entities[0].get_center_of_mass()
        anchor_x = float(sim.pos_x[idx])
        anchor_y = float(sim.pos_y[idx])
        rx = anchor_x - com[0]
        ry = anchor_y - com[1]

        # Displace in +y → restoring force on atom is -y → reaction on entity is +y
        # τ = r × F_reaction; with r = (+x, 0) and F = (0, +y), τ = +x * y - 0 = positive
        sim.pos_y[idx] += 0.5

        sim.clear_entity_forces()
        sim.apply_tether_forces()
        forces = sim.get_entity_forces()
        torque = float(forces[0, 2])
        # The predicted sign for a +x lever arm with +y displacement is positive
        # (atom force is -y, entity reaction is +y, torque is +x*+y = +z).
        # Sign convention: rx > 0 → torque has same sign as the y-displacement direction
        # of the entity reaction (which equals the y-displacement of the atom).
        assert torque > 0, f"Predicted positive torque (rx={rx:.2f}, +y displacement); got {torque}"


class TestMaxTetherForceClampNoNaN:
    def test_extreme_displacement_is_finite(self):
        """A tethered atom at a huge displacement must not produce NaN or Inf."""
        from core.scene import Scene
        scene = Scene(skip_warmup=True)
        scene.sketch.add_line((0, 0), (10, 0))
        scene.sketch.entities[0].physical = True
        scene.sketch.entities[0].dynamic = True
        scene.rebuild()

        sim = scene.simulation
        idx = next(i for i in range(sim.count) if sim.is_static[i] == 3)
        sim.pos_x[idx] = 1e6
        sim.pos_y[idx] = 1e6

        sim.clear_entity_forces()
        sim.apply_tether_forces()
        # Forces must remain finite
        assert np.all(np.isfinite(sim.force_x[:sim.count]))
        assert np.all(np.isfinite(sim.force_y[:sim.count]))
        forces = sim.get_entity_forces()
        assert np.all(np.isfinite(forces))


class TestSimulationSnapshotUndo:
    """Renamed per TU-SIM: this exercises raw snapshot/undo on the Simulation,
    not an integration with the BrushTool path. The Brush tool's snapshot
    integration is tested separately in test_tools.py."""

    def test_manual_snapshot_then_paint_then_undo_restores_count(self, scene):
        sim = scene.simulation
        sim.snapshot()
        scene.paint_particles(25.0, 25.0, radius=2.0)
        assert sim.count > 0
        sim.undo()
        # After undo, particle count returns to zero
        assert sim.count == 0


# ----- CompositeCommand topology declaration -------------------------------

class TestCompositeCommandTopology:
    def test_generic_composite_inherits_false_changes_topology(self, scene):
        """Generic CompositeCommand inherits changes_topology=False from the
        Command base class; Scene.execute treats it as a geometry-only change.
        Documents the contract: topology declarations must be explicitly set
        on subclasses (like AddRectangleCommand) to opt into rebuild."""
        from core.commands import CompositeCommand, AddLineCommand
        cmds = [AddLineCommand(scene.sketch, (0, 0), (1, 0), historize=False)]
        comp = CompositeCommand(cmds, "test")
        # Inherits Command.changes_topology = False
        assert comp.changes_topology is False
        scene.execute(comp)
        # Without changes_topology, Scene marks geometry_dirty (not topology_dirty)
        assert scene._topology_dirty is False
        assert scene._geometry_dirty is True
