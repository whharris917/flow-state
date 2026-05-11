"""Tests for the Molecule data model (model/molecule.py), Sketch palette,
bond_defaults table, and AddMoleculeCommand.

Engine bond plumbing is covered separately in test_simulation.py
(TestBondArrayConstruction, TestSpringForceKernel, ...). These tests are
the data-model and command layer atop that engine.
"""

import math

import numpy as np
import pytest

from model.molecule import (
    MoleculeAtom, MoleculeBond, MoleculeTemplate,
    make_diatom, make_water,
)
from model.sketch import Sketch
from core.scene import Scene
from core.molecule_commands import AddMoleculeCommand


# =============================================================================
# Data model round-trip and validation
# =============================================================================

class TestMoleculeAtomRoundTrip:
    def test_to_from_dict(self):
        a = MoleculeAtom(x=1.5, y=-2.5, material_name="Mercury")
        d = a.to_dict()
        b = MoleculeAtom.from_dict(d)
        assert b.x == pytest.approx(1.5)
        assert b.y == pytest.approx(-2.5)
        assert b.material_name == "Mercury"


class TestMoleculeBondRoundTrip:
    def test_to_from_dict(self):
        bond = MoleculeBond(atom_a=0, atom_b=2, k=250.0, r_eq=1.2)
        d = bond.to_dict()
        b2 = MoleculeBond.from_dict(d)
        assert b2.atom_a == 0
        assert b2.atom_b == 2
        assert b2.k == pytest.approx(250.0)
        assert b2.r_eq == pytest.approx(1.2)


class TestMoleculeTemplateRoundTrip:
    def test_to_from_dict_preserves_structure(self):
        tpl = make_water()
        d = tpl.to_dict()
        tpl2 = MoleculeTemplate.from_dict(d)
        assert tpl2.name == tpl.name
        assert len(tpl2.atoms) == len(tpl.atoms)
        assert len(tpl2.bonds) == len(tpl.bonds)
        for a, a2 in zip(tpl.atoms, tpl2.atoms):
            assert a.x == pytest.approx(a2.x)
            assert a.y == pytest.approx(a2.y)
            assert a.material_name == a2.material_name
        for b, b2 in zip(tpl.bonds, tpl2.bonds):
            assert b.atom_a == b2.atom_a
            assert b.atom_b == b2.atom_b
            assert b.k == pytest.approx(b2.k)
            assert b.r_eq == pytest.approx(b2.r_eq)


class TestMoleculeTemplateValidate:
    def test_diatom_validates(self):
        assert make_diatom().validate() is True

    def test_water_validates(self):
        assert make_water().validate() is True

    def test_out_of_range_atom_index_fails(self):
        tpl = MoleculeTemplate(
            name="Broken",
            atoms=[MoleculeAtom(0.0, 0.0, "Water")],
            bonds=[MoleculeBond(atom_a=0, atom_b=5, k=100.0, r_eq=1.0)],
        )
        assert tpl.validate() is False

    def test_self_bond_fails(self):
        tpl = MoleculeTemplate(
            name="Self",
            atoms=[MoleculeAtom(0.0, 0.0, "Water"),
                   MoleculeAtom(1.0, 0.0, "Water")],
            bonds=[MoleculeBond(atom_a=0, atom_b=0, k=100.0, r_eq=1.0)],
        )
        assert tpl.validate() is False


class TestStarterMolecules:
    """The seeded starter molecules must produce physically sensible templates."""

    def test_make_diatom_bond_length_matches_r_eq(self):
        tpl = make_diatom(r_eq=1.5)
        # Atoms placed at (-0.75, 0) and (0.75, 0) — distance 1.5
        dx = tpl.atoms[1].x - tpl.atoms[0].x
        dy = tpl.atoms[1].y - tpl.atoms[0].y
        d = math.hypot(dx, dy)
        assert d == pytest.approx(1.5)
        # Bond carries the same r_eq
        assert tpl.bonds[0].r_eq == pytest.approx(1.5)

    def test_make_water_central_atom_is_distinct_material(self):
        tpl = make_water()
        # Central atom (index 0) should be the o_material; the two H atoms
        # should share the h_material (and be distinct from the central).
        assert tpl.atoms[0].material_name != tpl.atoms[1].material_name
        assert tpl.atoms[1].material_name == tpl.atoms[2].material_name

    def test_make_water_has_two_bonds_from_central(self):
        tpl = make_water()
        # Both bonds emanate from the central (atom 0)
        assert len(tpl.bonds) == 2
        assert tpl.bonds[0].atom_a == 0 or tpl.bonds[0].atom_b == 0
        assert tpl.bonds[1].atom_a == 0 or tpl.bonds[1].atom_b == 0


# =============================================================================
# Sketch palette + bond defaults
# =============================================================================

class TestSketchMoleculePalette:
    def test_defaults_seeded(self, sketch):
        # Seeded starter palette: 'Diatom' and 'Water-mol' (renamed to avoid
        # collision with the 'Water' material).
        assert 'Diatom' in sketch.molecules
        # The second seed is whatever make_water named itself
        assert any(name != 'Diatom' for name in sketch.molecules)

    def test_add_molecule_inserts(self, sketch):
        tpl = make_diatom(name="Custom", material_name="Mercury")
        sketch.add_molecule(tpl)
        assert sketch.get_molecule("Custom") is tpl

    def test_add_molecule_overwrites_same_name(self, sketch):
        tpl1 = make_diatom(name="X", r_eq=1.0)
        tpl2 = make_diatom(name="X", r_eq=2.0)
        sketch.add_molecule(tpl1)
        sketch.add_molecule(tpl2)
        assert sketch.get_molecule("X").bonds[0].r_eq == pytest.approx(2.0)

    def test_remove_molecule(self, sketch):
        sketch.remove_molecule("Diatom")
        assert sketch.get_molecule("Diatom") is None

    def test_get_unknown_molecule_returns_none(self, sketch):
        assert sketch.get_molecule("Nonexistent") is None


class TestBondDefaultsTable:
    def test_fallback_uses_sigma_average(self, sketch):
        # Water has sigma=1.0, Mercury has sigma=0.8. Average = 0.9.
        k, r_eq = sketch.get_bond_default("Water", "Mercury")
        assert r_eq == pytest.approx(0.9, rel=1e-3)
        assert k == pytest.approx(200.0)

    def test_same_material_pair_uses_self_sigma(self, sketch):
        # Both atoms are Water (sigma=1.0). Average sigma = 1.0.
        k, r_eq = sketch.get_bond_default("Water", "Water")
        assert r_eq == pytest.approx(1.0)

    def test_unknown_material_falls_back_via_get_material(self, sketch):
        # sketch.get_material falls back to Water/Wall for unknown names →
        # we should still get back a numeric (k, r_eq) without raising.
        k, r_eq = sketch.get_bond_default("UnknownMat", "Water")
        assert isinstance(k, float)
        assert isinstance(r_eq, float)
        assert r_eq > 0

    def test_set_bond_default_overrides_fallback(self, sketch):
        sketch.set_bond_default("Water", "Mercury", k=999.0, r_eq=1.7)
        k, r_eq = sketch.get_bond_default("Water", "Mercury")
        assert k == pytest.approx(999.0)
        assert r_eq == pytest.approx(1.7)

    def test_set_bond_default_is_order_independent(self, sketch):
        sketch.set_bond_default("Water", "Mercury", k=999.0, r_eq=1.7)
        # Look up with the names swapped
        k, r_eq = sketch.get_bond_default("Mercury", "Water")
        assert k == pytest.approx(999.0)
        assert r_eq == pytest.approx(1.7)


class TestSketchMoleculeSerialization:
    def test_round_trip_preserves_palette(self, sketch):
        # Add a custom molecule + a bond default override
        custom = make_diatom(name="Custom", material_name="Mercury",
                             r_eq=1.4, k=350.0)
        sketch.add_molecule(custom)
        sketch.set_bond_default("Water", "Mercury", k=999.0, r_eq=1.7)

        d = sketch.to_dict()
        sk2 = Sketch()
        sk2.restore(d)

        assert "Custom" in sk2.molecules
        tpl = sk2.molecules["Custom"]
        assert tpl.bonds[0].k == pytest.approx(350.0)
        assert tpl.bonds[0].r_eq == pytest.approx(1.4)
        assert tpl.atoms[0].material_name == "Mercury"

        k, r_eq = sk2.get_bond_default("Water", "Mercury")
        assert k == pytest.approx(999.0)
        assert r_eq == pytest.approx(1.7)

    def test_pre_r2_save_reseeds_starter_palette(self, sketch):
        # A save dict without a 'molecules' key (pre-R2) should re-seed
        # the starter molecules so the user isn't left with an empty palette.
        legacy = {
            'entities': [],
            'constraints': [],
            'materials': {k: v.to_dict() for k, v in sketch.materials.items()},
        }
        sk2 = Sketch()
        sk2.restore(legacy)
        assert 'Diatom' in sk2.molecules


# =============================================================================
# AddMoleculeCommand — placement, undo, redo, transforms
# =============================================================================

class TestAddMoleculeCommand:
    def test_places_atoms_and_bonds(self, scene):
        tpl = make_diatom(material_name="Water", r_eq=1.0)
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        ok = cmd.execute()
        assert ok is True
        sim = scene.simulation
        assert sim.count == 2
        assert sim.bond_count == 1
        # Atoms placed symmetrically about (10, 10) along local x
        ax = float(sim.pos_x[0])
        bx = float(sim.pos_x[1])
        assert ax == pytest.approx(9.5)
        assert bx == pytest.approx(10.5)
        assert float(sim.bond_r_eq[0]) == pytest.approx(1.0)

    def test_spawned_atoms_carry_material_properties(self, scene):
        tpl = make_diatom(material_name="Mercury", r_eq=1.0)
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(20.0, 20.0))
        cmd.execute()
        sim = scene.simulation
        sketch = scene.sketch
        mat = sketch.get_material("Mercury")
        # Mercury: sigma=0.8, epsilon=2.0, mass=13.5
        assert float(sim.atom_sigma[0]) == pytest.approx(mat.sigma)
        assert float(sim.atom_mass[0]) == pytest.approx(mat.mass)
        # atom_eps_sqrt stores sqrt(epsilon)
        assert float(sim.atom_eps_sqrt[0]) == pytest.approx(math.sqrt(mat.epsilon))
        # Color taken from material
        for i in range(3):
            assert int(sim.atom_color[0, i]) == mat.color[i]

    def test_rotation_rotates_atom_positions(self, scene):
        # Diatom with atoms at (-0.5, 0) and (0.5, 0). Rotate by 90° about
        # the placement center → atoms should be at (0, -0.5) and (0, 0.5).
        tpl = make_diatom(material_name="Water", r_eq=1.0)
        cmd = AddMoleculeCommand(
            scene, tpl, world_pos=(10.0, 10.0),
            rotation=math.pi / 2,
        )
        cmd.execute()
        sim = scene.simulation
        # Atom 0 (was at -0.5, 0 locally) → (10, 10) + R(90°)*(-0.5, 0)
        #     = (10, 10) + (0, -0.5) = (10, 9.5)
        assert float(sim.pos_x[0]) == pytest.approx(10.0, abs=1e-4)
        assert float(sim.pos_y[0]) == pytest.approx(9.5, abs=1e-4)
        # Atom 1 (was at +0.5, 0 locally) → (10, 10) + (0, +0.5) = (10, 10.5)
        assert float(sim.pos_x[1]) == pytest.approx(10.0, abs=1e-4)
        assert float(sim.pos_y[1]) == pytest.approx(10.5, abs=1e-4)

    def test_undo_truncates_to_pre_state(self, scene):
        sim = scene.simulation
        # Pre-existing brush atom (simulating prior work)
        sim._add_particle(5.0, 5.0)
        pre_count = sim.count

        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()
        assert sim.count == pre_count + 2
        assert sim.bond_count == 1

        cmd.undo()
        assert sim.count == pre_count
        assert sim.bond_count == 0

    def test_redo_restores_atoms_and_bonds(self, scene):
        tpl = make_water()
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(15.0, 15.0))
        cmd.execute()
        n_atoms_first = scene.simulation.count
        n_bonds_first = scene.simulation.bond_count
        assert n_atoms_first == 3
        assert n_bonds_first == 2

        cmd.undo()
        assert scene.simulation.count == 0
        assert scene.simulation.bond_count == 0

        cmd.redo()
        assert scene.simulation.count == n_atoms_first
        assert scene.simulation.bond_count == n_bonds_first

    def test_invalid_template_returns_false_no_mutation(self, scene):
        bad = MoleculeTemplate(
            name="Bad",
            atoms=[MoleculeAtom(0.0, 0.0, "Water")],
            bonds=[MoleculeBond(atom_a=0, atom_b=99, k=100.0, r_eq=1.0)],
        )
        sim = scene.simulation
        pre_count = sim.count
        pre_bond_count = sim.bond_count
        cmd = AddMoleculeCommand(scene, bad, world_pos=(10.0, 10.0))
        assert cmd.execute() is False
        # No partial placement
        assert sim.count == pre_count
        assert sim.bond_count == pre_bond_count

    def test_bonds_link_correct_atom_indices(self, scene):
        """When placement adds atoms at base offset (sim.count = N before),
        bonds must reference the NEW indices, not the template-local indices."""
        sim = scene.simulation
        # Pre-existing atoms shift the new molecule's atom indices.
        for k in range(3):
            sim._add_particle(float(k), 0.0)
        pre_count = sim.count  # 3

        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()

        # Molecule's two atoms got indices 3 and 4.
        # The bond should reference atom indices 3 and 4, NOT 0 and 1.
        assert int(sim.bond_i[0]) == pre_count
        assert int(sim.bond_j[0]) == pre_count + 1


class TestMoleculeIntegrationWithScene:
    """End-to-end: place a molecule via the Scene's CommandQueue and verify
    the bond actually exerts force during stepping."""

    def test_placed_molecule_atoms_oscillate(self, scene):
        sim = scene.simulation
        sim.world_size = 200.0
        sim.gravity = 0.0
        sim.paused = False
        sim.dt = 0.001
        sim.damping = 1.0
        sim.use_boundaries = False

        # Make a stretched diatom: atoms placed 2 units apart but r_eq=1.
        tpl = MoleculeTemplate(
            name="Stretched",
            atoms=[
                MoleculeAtom(x=-1.0, y=0.0, material_name="Water"),
                MoleculeAtom(x=+1.0, y=0.0, material_name="Water"),
            ],
            bonds=[MoleculeBond(atom_a=0, atom_b=1, k=500.0, r_eq=1.0)],
        )
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(100.0, 100.0))
        scene.execute(cmd)

        # Atoms should be pulled together when we step the simulation
        initial_separation = float(np.hypot(
            sim.pos_x[1] - sim.pos_x[0],
            sim.pos_y[1] - sim.pos_y[0],
        ))
        assert initial_separation == pytest.approx(2.0, abs=1e-3)

        for _ in range(50):
            sim.step(steps_to_run=1)

        final_separation = float(np.hypot(
            sim.pos_x[1] - sim.pos_x[0],
            sim.pos_y[1] - sim.pos_y[0],
        ))
        # The bond should have pulled atoms toward equilibrium (r_eq=1.0)
        assert final_separation < initial_separation

    def test_undo_through_scene_removes_molecule(self, scene):
        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        scene.execute(cmd)
        assert scene.simulation.count == 2
        assert scene.simulation.bond_count == 1

        scene.undo()
        assert scene.simulation.count == 0
        assert scene.simulation.bond_count == 0


class TestMaxwellBoltzmannPlacementVelocity:
    """At placement time AddMoleculeCommand seeds a Maxwell-Boltzmann
    centre-of-mass velocity drawn from N(0, sqrt(target_temp / M_molecule)).

    The Berendsen thermostat (apply_thermostat) is a multiplicative
    rescaler with an `if current_T <= 1e-6: return` early-out — it can
    only scale existing motion. Without this kick the placed molecule
    sits at LJ + bond equilibrium with zero velocity → zero KE → the
    thermostat does nothing and the molecule never translates.
    """

    def _seed(self):
        # Seed Python's random so the COM velocity sample is reproducible.
        import random
        random.seed(12345)

    def test_zero_target_temp_yields_zero_velocity(self, scene):
        self._seed()
        scene.simulation.target_temp = 0.0
        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()
        sim = scene.simulation
        assert float(sim.vel_x[0]) == 0.0
        assert float(sim.vel_y[0]) == 0.0
        assert float(sim.vel_x[1]) == 0.0
        assert float(sim.vel_y[1]) == 0.0

    def test_nonzero_target_temp_yields_nonzero_velocity(self, scene):
        self._seed()
        scene.simulation.target_temp = 1.0
        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()
        sim = scene.simulation
        # At least one component should be nonzero (with seed 12345, a draw
        # from N(0, std) is extremely unlikely to land at exactly 0 on
        # both axes simultaneously).
        assert (float(sim.vel_x[0]) != 0.0 or float(sim.vel_y[0]) != 0.0)

    def test_all_atoms_share_com_velocity(self, scene):
        """The kick is centre-of-mass: every atom in the molecule receives
        the SAME (vx_cm, vy_cm). Internal vibration is NOT seeded — that
        keeps the bond at r_eq instead of stretching it on placement."""
        self._seed()
        scene.simulation.target_temp = 1.0
        tpl = make_diatom(material_name="Water")
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()
        sim = scene.simulation
        assert float(sim.vel_x[0]) == pytest.approx(float(sim.vel_x[1]))
        assert float(sim.vel_y[0]) == pytest.approx(float(sim.vel_y[1]))

    def test_water_molecule_three_atoms_share_velocity(self, scene):
        self._seed()
        scene.simulation.target_temp = 1.0
        tpl = make_water()
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0))
        cmd.execute()
        sim = scene.simulation
        v0x = float(sim.vel_x[0])
        v0y = float(sim.vel_y[0])
        for i in range(1, 3):
            assert float(sim.vel_x[i]) == pytest.approx(v0x)
            assert float(sim.vel_y[i]) == pytest.approx(v0y)

    def test_heavier_molecule_has_smaller_velocity_variance(self, scene):
        """At the same target_temp, the heavier molecule's M is larger so
        std = sqrt(target_temp / M) is smaller. Average |v| over many
        placements should follow."""
        import random
        # Light molecule: 2 × Water (m=1 each) → M=2
        # Heavy molecule: 2 × Mercury (m=13.5 each) → M=27
        scene.simulation.target_temp = 1.0
        light = make_diatom(material_name="Water")
        heavy = make_diatom(material_name="Mercury")

        N = 200
        light_speeds = []
        random.seed(1)
        for _ in range(N):
            scene.simulation.count = 0
            scene.simulation.bond_count = 0
            cmd = AddMoleculeCommand(scene, light, world_pos=(10.0, 10.0))
            cmd.execute()
            sp = math.hypot(float(scene.simulation.vel_x[0]),
                            float(scene.simulation.vel_y[0]))
            light_speeds.append(sp)

        heavy_speeds = []
        random.seed(1)
        for _ in range(N):
            scene.simulation.count = 0
            scene.simulation.bond_count = 0
            cmd = AddMoleculeCommand(scene, heavy, world_pos=(10.0, 10.0))
            cmd.execute()
            sp = math.hypot(float(scene.simulation.vel_x[0]),
                            float(scene.simulation.vel_y[0]))
            heavy_speeds.append(sp)

        mean_light = sum(light_speeds) / N
        mean_heavy = sum(heavy_speeds) / N
        # Theoretical ratio: sqrt(M_heavy / M_light) = sqrt(27/2) ≈ 3.67
        # Loose bound — sampling noise at N=200 → allow ±25%
        ratio = mean_light / mean_heavy
        assert 2.7 < ratio < 5.0

    def test_placed_molecule_translates_under_thermostat(self, scene):
        """End-to-end sanity check that pins the original bug: with the
        thermostat on, a placed water molecule should drift (its centre
        of mass should move) over a few-hundred-substep window."""
        import random
        random.seed(7)
        sim = scene.simulation
        sim.world_size = 200.0
        sim.gravity = 0.0
        sim.use_boundaries = False
        sim.use_thermostat = True
        sim.target_temp = 1.0
        sim.damping = 1.0  # no medium drag — pure thermal motion
        sim.paused = False
        sim.dt = 0.001

        tpl = make_water()
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(100.0, 100.0))
        scene.execute(cmd)

        # Initial centre of mass
        cx0 = float(sum(sim.pos_x[:sim.count]) / sim.count)
        cy0 = float(sum(sim.pos_y[:sim.count]) / sim.count)

        for _ in range(200):
            sim.step(steps_to_run=1)

        cx1 = float(sum(sim.pos_x[:sim.count]) / sim.count)
        cy1 = float(sum(sim.pos_y[:sim.count]) / sim.count)
        drift = math.hypot(cx1 - cx0, cy1 - cy0)

        # The molecule should have moved measurably (>0.05 simulation units
        # over 200 substeps at dt=0.001 → 0.2 sim-time at thermal speeds
        # ≈ sqrt(1/15.5) ≈ 0.25 sim/units per sim/time).
        assert drift > 0.05, (
            f"Molecule didn't translate (drift={drift:.4f}). Pre-fix this "
            "was zero — placed molecules had vx=vy=0 and the Berendsen "
            "thermostat couldn't heat from absolute zero."
        )
