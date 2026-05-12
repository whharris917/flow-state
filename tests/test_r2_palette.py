"""Tests for the R2 emergent-phenomena palette additions.

Three layers, each tested in isolation so failures point cleanly at the
breaking surface:
1. PRESET_MATERIALS additions (Polar, Nonpolar, Heavy, LightGas)
2. Sketch.lj_cross_overrides seeded defaults
3. New molecule helpers: make_surfactant, make_lipid, make_polymer

End-to-end micelle / bilayer phenomenology is deferred to R3 (which adds the
demo presets). R2's contract is "the palette exists, has the right shape,
and the ε defaults are seeded correctly" — those are verified here.
"""

import math

import numpy as np
import pytest

from core.molecule_commands import AddMoleculeCommand
from core.scene import Scene
from engine.simulation import Simulation
from model.molecule import (
    MoleculeAtom, MoleculeBond, MoleculeAngle,
    make_surfactant, make_lipid, make_polymer,
)
from model.properties import PRESET_MATERIALS
from model.sketch import Sketch


# =============================================================================
# PRESET_MATERIALS — R2 additions
# =============================================================================

class TestNewMaterials:
    def test_polar_present_with_design_params(self):
        m = PRESET_MATERIALS['Polar']
        assert m.sigma == 1.0
        assert m.epsilon == 1.0
        assert m.mass == 1.0

    def test_nonpolar_present_with_design_params(self):
        m = PRESET_MATERIALS['Nonpolar']
        assert m.sigma == 1.0
        assert m.epsilon == 1.0
        assert m.mass == 1.0

    def test_heavy_present_with_design_params(self):
        m = PRESET_MATERIALS['Heavy']
        assert m.sigma == 1.2
        assert m.epsilon == 2.0
        assert m.mass == 3.0

    def test_light_gas_present_with_design_params(self):
        m = PRESET_MATERIALS['LightGas']
        assert m.sigma == 0.8
        assert m.epsilon == 0.3
        assert m.mass == 0.5

    def test_polar_nonpolar_share_sigma_for_clean_demixing(self):
        """The Polar/Nonpolar pair specifically shares σ so demixing can be
        attributed purely to cross-ε rather than size-mismatch packing."""
        assert PRESET_MATERIALS['Polar'].sigma == PRESET_MATERIALS['Nonpolar'].sigma

    def test_legacy_materials_still_present(self):
        for name in ('Water', 'Oil', 'Mercury', 'Honey', 'Wall'):
            assert name in PRESET_MATERIALS


class TestSketchSeedsR2Materials:
    def test_fresh_sketch_carries_all_r2_species(self):
        s = Sketch()
        for name in ('Polar', 'Nonpolar', 'Heavy', 'LightGas'):
            assert name in s.materials
        # Stable id present too
        for name in ('Polar', 'Nonpolar', 'Heavy', 'LightGas'):
            assert s.get_material_index(name) >= 0


# =============================================================================
# Seeded cross-ε overrides
# =============================================================================

class TestSeededLjOverrides:
    def test_polar_nonpolar_seeded_immiscible(self):
        s = Sketch()
        assert s.get_lj_epsilon("Polar", "Nonpolar") == pytest.approx(0.25)
        # Compare to L-B baseline: √(1·1) = 1.0. Override is 4× weaker.
        assert s.get_lj_epsilon("Polar", "Nonpolar") < 0.5

    def test_polar_light_gas_seeded_gas_insoluble(self):
        s = Sketch()
        assert s.get_lj_epsilon("Polar", "LightGas") == pytest.approx(0.15)

    def test_heavy_light_gas_seeded(self):
        s = Sketch()
        assert s.get_lj_epsilon("Heavy", "LightGas") == pytest.approx(0.20)

    def test_nonpolar_heavy_seeded_above_lb(self):
        """ε_Nonpolar-Heavy override > L-B (√(1·2) ≈ 1.414) — "oil dissolves
        grease" — the override makes the pair more attractive than mixing
        would predict."""
        s = Sketch()
        assert s.get_lj_epsilon("Nonpolar", "Heavy") == pytest.approx(1.5)
        # Above the L-B prediction
        assert s.get_lj_epsilon("Nonpolar", "Heavy") > math.sqrt(1.0 * 2.0)

    def test_unrelated_pairs_use_lb(self):
        """Pairs not in the seed table fall back to L-B (no surprise overrides)."""
        s = Sketch()
        # Water-Oil: no override → L-B
        assert s.get_lj_epsilon("Water", "Oil") == pytest.approx(math.sqrt(1.0 * 0.8))
        # Polar-Heavy: no override → L-B
        assert s.get_lj_epsilon("Polar", "Heavy") == pytest.approx(math.sqrt(1.0 * 2.0))

    def test_restore_clears_seeded_overrides(self):
        """Loading a save replaces the lj_cross_overrides dict with whatever
        the save contained. Pre-R2 saves restore as empty (no surprise
        overrides on legacy scenes)."""
        s = Sketch()
        data = s.to_dict()
        # Strip the key as if from a pre-R2 save
        del data['lj_cross_overrides']
        s2 = Sketch()  # has R2 defaults seeded
        s2.restore(data)
        assert s2.lj_cross_overrides == {}
        # Polar-Nonpolar now falls back to L-B (1.0), no longer the seeded 0.25
        assert s2.get_lj_epsilon("Polar", "Nonpolar") == pytest.approx(1.0)


# =============================================================================
# make_surfactant
# =============================================================================

class TestMakeSurfactant:
    def test_default_5_atoms_polar_head_nonpolar_tails(self):
        tpl = make_surfactant()
        assert len(tpl.atoms) == 5
        assert tpl.atoms[0].material_name == "Polar"
        for a in tpl.atoms[1:]:
            assert a.material_name == "Nonpolar"

    def test_atoms_collinear_along_x_axis(self):
        tpl = make_surfactant()
        for a in tpl.atoms:
            assert a.y == 0.0
        # x positions strictly increasing
        xs = [a.x for a in tpl.atoms]
        for i in range(len(xs) - 1):
            assert xs[i + 1] > xs[i]

    def test_atoms_centered_at_origin(self):
        tpl = make_surfactant()
        xs = [a.x for a in tpl.atoms]
        # Mean x ≈ 0 (centred chain)
        assert abs(sum(xs) / len(xs)) < 1e-6

    def test_n_minus_1_bonds_in_series(self):
        tpl = make_surfactant(n_tail=4)
        assert len(tpl.bonds) == 4  # head + 4 tails → 4 bonds
        for i, b in enumerate(tpl.bonds):
            assert b.atom_a == i
            assert b.atom_b == i + 1

    def test_interior_angles_straighten_chain(self):
        tpl = make_surfactant(n_tail=4)
        # n_atoms=5, interior atoms 1, 2, 3 each carry a 180° angle constraint
        assert len(tpl.angles) == 3
        for ang in tpl.angles:
            assert ang.theta_eq == pytest.approx(math.pi)

    def test_custom_n_tail_scales_atom_count(self):
        tpl = make_surfactant(n_tail=7)
        assert len(tpl.atoms) == 8  # 1 head + 7 tails
        assert len(tpl.bonds) == 7  # 7 chain bonds
        assert len(tpl.angles) == 6  # interior 1..6

    def test_validate(self):
        tpl = make_surfactant()
        assert tpl.validate() is True


# =============================================================================
# make_lipid
# =============================================================================

class TestMakeLipid:
    def test_six_atoms_two_polar_four_nonpolar(self):
        tpl = make_lipid()
        assert len(tpl.atoms) == 6
        # head, neck = Polar; four tail atoms = Nonpolar
        assert tpl.atoms[0].material_name == "Polar"
        assert tpl.atoms[1].material_name == "Polar"
        for i in (2, 3, 4, 5):
            assert tpl.atoms[i].material_name == "Nonpolar"

    def test_head_above_neck(self):
        tpl = make_lipid()
        assert tpl.atoms[0].y > tpl.atoms[1].y

    def test_tails_below_neck(self):
        tpl = make_lipid()
        for i in (2, 3, 4, 5):
            assert tpl.atoms[i].y < tpl.atoms[1].y

    def test_tails_splay_apart(self):
        """tail1a (idx 2) has x<0, tail2a (idx 4) has x>0 — the V-shape."""
        tpl = make_lipid()
        assert tpl.atoms[2].x < 0
        assert tpl.atoms[4].x > 0
        assert tpl.atoms[3].x < 0  # tail1b continues left
        assert tpl.atoms[5].x > 0  # tail2b continues right

    def test_five_bonds(self):
        tpl = make_lipid()
        assert len(tpl.bonds) == 5
        # head-neck, neck-tail1a, tail1a-tail1b, neck-tail2a, tail2a-tail2b
        expected_pairs = {(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)}
        actual_pairs = {(b.atom_a, b.atom_b) for b in tpl.bonds}
        assert actual_pairs == expected_pairs

    def test_five_angles(self):
        tpl = make_lipid()
        assert len(tpl.angles) == 5

    def test_validate(self):
        tpl = make_lipid()
        assert tpl.validate() is True

    def test_tail_branch_angle_setting(self):
        """tail1a–neck–tail2a angle should match the requested branch angle."""
        tpl = make_lipid(tail_branch_angle_deg=80.0)
        # Find the angle with apex=1 (neck) and legs 2, 4
        target = None
        for ang in tpl.angles:
            if ang.atom_b == 1 and {ang.atom_a, ang.atom_c} == {2, 4}:
                target = ang
                break
        assert target is not None
        assert target.theta_eq == pytest.approx(math.radians(80.0))


# =============================================================================
# make_polymer
# =============================================================================

class TestMakePolymer:
    def test_default_15_atom_chain(self):
        tpl = make_polymer()
        assert len(tpl.atoms) == 15
        for a in tpl.atoms:
            assert a.material_name == "Nonpolar"

    def test_atoms_collinear_centered(self):
        tpl = make_polymer()
        for a in tpl.atoms:
            assert a.y == 0.0
        xs = [a.x for a in tpl.atoms]
        assert abs(sum(xs) / len(xs)) < 1e-6

    def test_n_minus_1_bonds_in_series(self):
        tpl = make_polymer(n=10)
        assert len(tpl.bonds) == 9
        for i, b in enumerate(tpl.bonds):
            assert b.atom_a == i
            assert b.atom_b == i + 1

    def test_interior_angles_at_180(self):
        tpl = make_polymer(n=8)
        # Interior atoms 1..6 → 6 angles, all at π
        assert len(tpl.angles) == 6
        for ang in tpl.angles:
            assert ang.theta_eq == pytest.approx(math.pi)

    def test_custom_material(self):
        tpl = make_polymer(material="Polar")
        for a in tpl.atoms:
            assert a.material_name == "Polar"

    def test_validate(self):
        tpl = make_polymer()
        assert tpl.validate() is True


# =============================================================================
# Sketch palette integration — fresh Sketch ships with R2 molecules
# =============================================================================

class TestSketchSeedsR2Molecules:
    def test_palette_contains_surfactant_lipid_polymer(self):
        s = Sketch()
        for name in ('Surfactant', 'Lipid', 'Polymer'):
            assert name in s.molecules, f"{name} missing from sketch.molecules"

    def test_legacy_molecules_still_present(self):
        s = Sketch()
        for name in ('Diatom', 'Water-mol', 'CO2', 'Ammonia',
                     'Methane', 'Benzene'):
            assert name in s.molecules

    def test_total_molecule_count_nine(self):
        """6 legacy + 3 R2 = 9. Sentinel to catch accidental seed drift."""
        s = Sketch()
        assert len(s.molecules) == 9


# =============================================================================
# End-to-end: placing an R2 molecule registers material_ids matching the
# seeded palette so the LJ kernel sees overrides immediately
# =============================================================================

class TestR2MoleculePlacementUsesOverrides:
    def test_placed_surfactant_atoms_carry_polar_or_nonpolar_material_id(self):
        scene = Scene(skip_warmup=True)
        sketch = scene.sketch
        sim = scene.simulation

        tpl = sketch.molecules['Surfactant']
        cmd = AddMoleculeCommand(scene, tpl, world_pos=(10.0, 10.0), rotation=0.0)
        ok = cmd.execute()
        assert ok

        # First atom is head (Polar), rest are tails (Nonpolar)
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        assert polar_id >= 0
        assert nonpolar_id >= 0
        N = sim.count
        # head atom id
        assert sim.atom_material_id[0] == polar_id
        # tail atoms
        for i in range(1, N):
            assert sim.atom_material_id[i] == nonpolar_id

    def test_placed_lipid_has_two_polar_four_nonpolar(self):
        scene = Scene(skip_warmup=True)
        sketch = scene.sketch
        sim = scene.simulation
        cmd = AddMoleculeCommand(scene, sketch.molecules['Lipid'],
                                 world_pos=(10.0, 10.0), rotation=0.0)
        assert cmd.execute()
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        n_polar = int(np.sum(sim.atom_material_id[:sim.count] == polar_id))
        n_nonpolar = int(np.sum(sim.atom_material_id[:sim.count] == nonpolar_id))
        assert n_polar == 2
        assert n_nonpolar == 4

    def test_eps_matrix_has_polar_nonpolar_override_for_placed_atoms(self):
        """After Scene.__init__ pushes the matrix, the (Polar, Nonpolar)
        entry should be the seeded 0.25, not the L-B 1.0. This is the
        end-to-end smoke check that R1 + R2 integrate correctly."""
        scene = Scene(skip_warmup=True)
        sketch = scene.sketch
        polar_id = sketch.get_material_index("Polar")
        nonpolar_id = sketch.get_material_index("Nonpolar")
        # The matrix was pushed in Scene.__init__
        assert scene.simulation.eps_ij_matrix[polar_id, nonpolar_id] == pytest.approx(0.25)
        # And the diagonal entries are the materials' own ε
        assert scene.simulation.eps_ij_matrix[polar_id, polar_id] == pytest.approx(1.0)
        assert scene.simulation.eps_ij_matrix[nonpolar_id, nonpolar_id] == pytest.approx(1.0)
