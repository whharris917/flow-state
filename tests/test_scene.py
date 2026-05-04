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
