"""Tests for engine/compiler.py — Sketch → Simulation atom emission."""

import math
import numpy as np
import pytest

from model.geometry import Line, Circle, Point
from model.constraints import Coincident


def _count_static(sim):
    return int(np.sum(sim.is_static[:sim.count] == 1))


def _count_tethered(sim):
    return int(np.sum(sim.is_static[:sim.count] == 3))


class TestEmptyCompile:
    def test_empty_sketch_emits_no_atoms(self, sketch, simulation, compiler):
        compiler.rebuild()
        assert simulation.count == 0


class TestPhysicalGate:
    def test_non_physical_entity_emits_nothing(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        # physical defaults to False
        compiler.rebuild()
        assert simulation.count == 0

    def test_non_physical_material_emits_nothing(self, sketch, simulation, compiler):
        from model.properties import Material
        sketch.materials["Guide"] = Material("Guide", physical=False, color=(255, 0, 0))
        sketch.add_line((0, 0), (10, 0), material_id="Guide")
        sketch.entities[0].physical = True
        compiler.rebuild()
        # Material has physical=False — atomization is suppressed even with entity.physical=True
        assert simulation.count == 0

    def test_reference_line_emits_nothing(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0), is_ref=True)
        sketch.entities[0].physical = True  # Even with physical, ref lines skip
        compiler.rebuild()
        assert simulation.count == 0

    def test_handle_point_emits_nothing(self, sketch, simulation, compiler):
        p = Point(5.0, 5.0)
        p.is_handle = True
        p.physical = True
        sketch.entities.append(p)
        compiler.rebuild()
        assert simulation.count == 0


class TestStaticAtomEmission:
    def test_static_line_produces_static_atoms(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        assert _count_static(simulation) > 0
        assert _count_tethered(simulation) == 0

    def test_static_atom_count_scales_with_length(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        short_count = simulation.count

        sketch.entities[0].end[:] = [20, 0]
        compiler.rebuild()
        long_count = simulation.count
        assert long_count > short_count

    def test_static_atoms_lie_along_line(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        # All atoms should have y ~ 0 (along horizontal line) and x in [0, 10]
        for i in range(simulation.count):
            assert simulation.pos_y[i] == pytest.approx(0.0, abs=0.01)
            assert 0.0 <= simulation.pos_x[i] <= 10.001

    def test_static_atom_records_tether_entity_idx(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        for i in range(simulation.count):
            # Static atoms still record entity_idx for sync_static_atoms_to_geometry
            assert simulation.tether_entity_idx[i] == 0


class TestTetheredAtomEmission:
    def test_dynamic_line_produces_tethered_atoms(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        sketch.entities[0].dynamic = True
        compiler.rebuild()
        assert _count_tethered(simulation) > 0
        assert _count_static(simulation) == 0

    def test_tethered_atoms_have_stiffness(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        sketch.entities[0].dynamic = True
        compiler.rebuild()
        for i in range(simulation.count):
            if simulation.is_static[i] == 3:
                assert simulation.tether_stiffness[i] > 0

    def test_dynamic_entity_mass_auto_scales_to_atom_count(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        line = sketch.entities[0]
        line.physical = True
        line.dynamic = True
        line.mass = 1.0  # Will be auto-bumped if too low for atom count
        compiler.rebuild()
        # Compiler enforces entity.mass >= num_atoms * ATOM_MASS * ENTITY_MASS_MULTIPLIER
        # so the entity is heavier than the atoms it carries
        assert line.mass > 1.0


class TestCircleEmission:
    def test_physical_circle_produces_atoms(self, sketch, simulation, compiler):
        sketch.add_circle((10, 10), 5.0)
        sketch.entities[0].physical = True
        compiler.rebuild()
        assert simulation.count >= 3  # min 3 atoms enforced by Compiler

    def test_circle_atoms_lie_on_circumference(self, sketch, simulation, compiler):
        cx, cy, r = 10.0, 10.0, 5.0
        sketch.add_circle((cx, cy), r)
        sketch.entities[0].physical = True
        compiler.rebuild()
        for i in range(simulation.count):
            dist = math.hypot(simulation.pos_x[i] - cx, simulation.pos_y[i] - cy)
            assert dist == pytest.approx(r, abs=0.01)


class TestRebuildIdempotence:
    def test_rebuild_preserves_dynamic_particles(self, sketch, simulation, compiler):
        # Manually add a dynamic (fluid) particle
        simulation._add_particle(5.0, 5.0, is_static=0)
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        # Fluid particle should survive — Compiler only compacts non-fluid atoms
        fluid_count = int(np.sum(simulation.is_static[:simulation.count] == 0))
        assert fluid_count == 1

    def test_rebuild_replaces_existing_static_atoms(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        first_count = simulation.count
        # Rebuild again — should produce same number of static atoms, not double
        compiler.rebuild()
        assert simulation.count == first_count


class TestCoincidentJointIDs:
    def test_coincident_atoms_share_joint_id(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (5, 0))
        sketch.add_line((5, 0), (10, 0))
        sketch.entities[0].physical = True
        sketch.entities[1].physical = True
        # Mark line 0's end coincident with line 1's start
        sketch.add_constraint_object(Coincident(0, 1, 1, 0), solve=False)
        compiler.rebuild()
        # Find atoms that should be at the joint (5, 0)
        joint_ids = set()
        for i in range(simulation.count):
            if simulation.joint_ids[i] != 0:
                joint_ids.add(int(simulation.joint_ids[i]))
        # At least one shared joint id should have been assigned
        assert len(joint_ids) >= 1

    def test_no_coincident_constraint_means_no_joint_ids(self, sketch, simulation, compiler):
        sketch.add_line((0, 0), (10, 0))
        sketch.entities[0].physical = True
        compiler.rebuild()
        # Without coincidents, no atom should have a non-zero joint_id
        assert int(np.sum(simulation.joint_ids[:simulation.count] != 0)) == 0
