"""Tests for the Molecule Builder modal dialog (ui/molecule_builder_dialog.py).

Drives the dialog through its programmatic API rather than synthesizing
pygame events — the API exists for testability (and also feeds the event
handler internally, so behavior tested here exercises the same paths).
"""

import pytest

from model.molecule import MoleculeAtom, MoleculeBond, MoleculeTemplate, make_diatom
from model.sketch import Sketch
from ui.molecule_builder_dialog import MoleculeBuilderDialog


@pytest.fixture
def dialog(sketch):
    """Dialog at fixed position, fresh template."""
    return MoleculeBuilderDialog(x=100, y=100, sketch=sketch, template=None)


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_fresh_template_starts_empty(self, dialog):
        assert dialog.template.atoms == []
        assert dialog.template.bonds == []
        assert dialog.template.name == "NewMolecule"

    def test_default_mode_is_atom(self, dialog):
        assert dialog.mode == 'atom'

    def test_with_existing_template_deep_copies(self, sketch):
        original = make_diatom(name="Original")
        dialog = MoleculeBuilderDialog(100, 100, sketch, template=original)
        # Template is a copy — mutating the dialog's template doesn't touch original
        dialog.template.atoms[0].x = 999.0
        assert original.atoms[0].x != 999.0

    def test_seeded_dialog_carries_atoms_and_bonds(self, sketch):
        original = make_diatom()
        dialog = MoleculeBuilderDialog(100, 100, sketch, template=original)
        assert len(dialog.template.atoms) == 2
        assert len(dialog.template.bonds) == 1


# =============================================================================
# Mode toggle
# =============================================================================

class TestModeToggle:
    def test_set_mode_atom(self, dialog):
        dialog.set_mode('bond')
        dialog.set_mode('atom')
        assert dialog.mode == 'atom'
        assert dialog.btn_atom.is_active is True
        assert dialog.btn_bond.is_active is False

    def test_set_mode_bond(self, dialog):
        dialog.set_mode('bond')
        assert dialog.mode == 'bond'
        assert dialog.btn_atom.is_active is False
        assert dialog.btn_bond.is_active is True

    def test_invalid_mode_ignored(self, dialog):
        dialog.set_mode('invalid')
        assert dialog.mode == 'atom'  # Unchanged

    def test_mode_switch_clears_pending_bond(self, dialog):
        dialog.add_atom_local(0.0, 0.0)
        dialog.add_atom_local(1.0, 0.0)
        dialog.set_mode('bond')
        dialog.begin_bond(0)
        assert dialog.pending_bond_first == 0
        dialog.set_mode('atom')
        assert dialog.pending_bond_first is None


# =============================================================================
# Atom placement
# =============================================================================

class TestAtomPlacement:
    def test_add_atom_local_appends(self, dialog):
        idx = dialog.add_atom_local(0.0, 0.0)
        assert idx == 0
        assert len(dialog.template.atoms) == 1
        atom = dialog.template.atoms[0]
        assert atom.x == pytest.approx(0.0)
        assert atom.y == pytest.approx(0.0)

    def test_atom_carries_dropdown_material(self, dialog):
        # Switch the dropdown to Mercury (3rd preset, index varies — find it)
        names = list(dialog.sketch.materials.keys())
        if 'Mercury' in names:
            dialog.dropdown.selected_index = names.index('Mercury')
        dialog.add_atom_local(1.0, 1.0)
        assert dialog.template.atoms[0].material_name == dialog.dropdown.get_selected()

    def test_explicit_material_override(self, dialog):
        dialog.add_atom_local(0.0, 0.0, material_name="Mercury")
        assert dialog.template.atoms[0].material_name == "Mercury"

    def test_overlap_rejected(self, dialog):
        """Two atoms at the same local position — second should be rejected."""
        dialog.add_atom_local(0.0, 0.0)
        result = dialog.add_atom_local(0.0, 0.0)
        assert result is None
        assert len(dialog.template.atoms) == 1

    def test_atoms_at_different_positions_both_added(self, dialog):
        # Place atoms far apart enough that the pixel-threshold doesn't reject.
        dialog.add_atom_local(-2.0, 0.0)
        dialog.add_atom_local(+2.0, 0.0)
        assert len(dialog.template.atoms) == 2


# =============================================================================
# Bond creation
# =============================================================================

class TestBondCreation:
    def test_begin_complete_creates_bond(self, dialog):
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.set_mode('bond')
        dialog.begin_bond(0)
        b_idx = dialog.complete_bond(1)
        assert b_idx == 0
        assert len(dialog.template.bonds) == 1
        b = dialog.template.bonds[0]
        assert b.atom_a == 0
        assert b.atom_b == 1

    def test_self_bond_rejected(self, dialog):
        dialog.add_atom_local(0.0, 0.0)
        dialog.begin_bond(0)
        b_idx = dialog.complete_bond(0)
        assert b_idx is None
        assert len(dialog.template.bonds) == 0

    def test_duplicate_bond_rejected(self, dialog):
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        # Try to add the same pair again (reversed direction)
        dialog.begin_bond(1)
        b_idx = dialog.complete_bond(0)
        assert b_idx is None
        assert len(dialog.template.bonds) == 1

    def test_bond_pulls_defaults_from_sketch(self, sketch):
        sketch.set_bond_default("Water", "Mercury", k=777.0, r_eq=1.23)
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0, material_name="Water")
        dialog.add_atom_local(+1.0, 0.0, material_name="Mercury")
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        b = dialog.template.bonds[0]
        assert b.k == pytest.approx(777.0)
        assert b.r_eq == pytest.approx(1.23)

    def test_cancel_pending_bond_drops_state(self, dialog):
        dialog.add_atom_local(0.0, 0.0)
        dialog.begin_bond(0)
        dialog.cancel_pending_bond()
        assert dialog.pending_bond_first is None

    def test_complete_without_begin_is_no_op(self, dialog):
        dialog.add_atom_local(0.0, 0.0)
        b_idx = dialog.complete_bond(0)
        assert b_idx is None
        assert len(dialog.template.bonds) == 0


# =============================================================================
# Bond editor — k / r_eq
# =============================================================================

class TestBondEditor:
    def _make_bonded_dialog(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        return dialog

    def test_new_bond_becomes_selected(self, sketch):
        dialog = self._make_bonded_dialog(sketch)
        assert dialog.selected_bond == 0

    def test_input_fields_populated_from_selected_bond(self, sketch):
        dialog = self._make_bonded_dialog(sketch)
        b = dialog.template.bonds[0]
        # InputField text matches the bond's values
        assert float(dialog.in_k.text) == pytest.approx(b.k, rel=1e-3)
        assert float(dialog.in_r_eq.text) == pytest.approx(b.r_eq, rel=1e-3)

    def test_apply_bond_input_edits_propagates_to_bond(self, sketch):
        dialog = self._make_bonded_dialog(sketch)
        dialog.in_k.set_value("888.0")
        dialog.in_r_eq.set_value("2.5")
        dialog.apply_bond_input_edits()
        b = dialog.template.bonds[0]
        assert b.k == pytest.approx(888.0)
        assert b.r_eq == pytest.approx(2.5)

    def test_non_positive_k_or_r_eq_rejected(self, sketch):
        dialog = self._make_bonded_dialog(sketch)
        orig_k = dialog.template.bonds[0].k
        orig_r_eq = dialog.template.bonds[0].r_eq
        dialog.in_k.set_value("-1.0")
        dialog.in_r_eq.set_value("0")
        dialog.apply_bond_input_edits()
        # Bond unchanged
        assert dialog.template.bonds[0].k == pytest.approx(orig_k)
        assert dialog.template.bonds[0].r_eq == pytest.approx(orig_r_eq)


# =============================================================================
# Clear
# =============================================================================

class TestClear:
    def test_clear_empties_template(self, dialog):
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        dialog.clear_template()
        assert dialog.template.atoms == []
        assert dialog.template.bonds == []
        assert dialog.selected_bond is None
        assert dialog.pending_bond_first is None


# =============================================================================
# Save / Cancel — interaction with sketch
# =============================================================================

class TestApply:
    def test_apply_to_sketch_uses_name_field(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        dialog.in_name.set_value("Custom-Diatom")
        dialog.apply_to_sketch(sketch)
        assert "Custom-Diatom" in sketch.molecules
        tpl = sketch.molecules["Custom-Diatom"]
        assert len(tpl.atoms) == 2
        assert len(tpl.bonds) == 1

    def test_apply_to_sketch_keeps_existing_name_when_blank(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.template.name = "Preset"
        dialog.add_atom_local(0.0, 0.0)
        dialog.in_name.set_value("")
        dialog.apply_to_sketch(sketch)
        assert "Preset" in sketch.molecules

    def test_apply_to_sketch_overwrites_collision(self, sketch):
        # Sketch already has 'Diatom' from the seed palette
        assert "Diatom" in sketch.molecules
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        # Author a single-atom 'Diatom' (would overwrite the seed)
        dialog.add_atom_local(0.0, 0.0)
        dialog.in_name.set_value("Diatom")
        dialog.apply_to_sketch(sketch)
        # Now the palette's 'Diatom' has 1 atom (overwritten)
        assert len(sketch.molecules["Diatom"].atoms) == 1

    def test_apply_pulls_pending_bond_input_edits(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0)
        dialog.complete_bond(1)
        dialog.in_k.set_value("555.0")
        dialog.in_r_eq.set_value("1.7")
        # Don't call apply_bond_input_edits explicitly — apply_to_sketch should
        dialog.apply_to_sketch(sketch)
        saved = sketch.molecules["NewMolecule"]
        assert saved.bonds[0].k == pytest.approx(555.0)
        assert saved.bonds[0].r_eq == pytest.approx(1.7)


# =============================================================================
# Coordinate transforms
# =============================================================================

class TestCoordinateTransforms:
    def test_canvas_center_corresponds_to_local_origin(self, dialog):
        cx, cy = dialog._canvas_center_px()
        # Round-trip: local (0, 0) → canvas (cx, cy) → local (0, 0)
        sx, sy = dialog.local_to_canvas(0.0, 0.0)
        assert sx == cx
        assert sy == cy
        local = dialog.canvas_to_local(sx, sy)
        assert local[0] == pytest.approx(0.0)
        assert local[1] == pytest.approx(0.0)

    def test_y_is_inverted(self, dialog):
        """Positive local-y should map to LOWER pixel-y (canvas convention
        has y growing downward; molecule convention has y growing upward)."""
        cx, cy = dialog._canvas_center_px()
        _, sy_pos = dialog.local_to_canvas(0.0, 1.0)
        _, sy_neg = dialog.local_to_canvas(0.0, -1.0)
        assert sy_pos < cy   # Positive y → above center
        assert sy_neg > cy   # Negative y → below center

    def test_pick_returns_nearest_atom(self, dialog):
        dialog.add_atom_local(-2.0, 0.0)
        dialog.add_atom_local(+2.0, 0.0)
        # Pixel coords of atom 1 (the +2.0 atom)
        px, py = dialog.local_to_canvas(2.0, 0.0)
        picked = dialog._pick_atom_at_pixel(px, py)
        assert picked == 1

    def test_pick_returns_none_when_far(self, dialog):
        dialog.add_atom_local(0.0, 0.0)
        # Pixel coords way outside the pick threshold
        picked = dialog._pick_atom_at_pixel(9999, 9999)
        assert picked is None


# =============================================================================
# Bond selection (multi-bond scenarios)
# =============================================================================

class TestBondSelection:
    def test_creating_second_bond_selects_it(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(0.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        # First bond: 0-1
        dialog.begin_bond(0); dialog.complete_bond(1)
        # Second bond: 1-2
        dialog.begin_bond(1); dialog.complete_bond(2)
        assert dialog.selected_bond == 1  # most recent

    def test_select_bond_changes_inputs(self, sketch):
        dialog = MoleculeBuilderDialog(100, 100, sketch)
        dialog.add_atom_local(-1.0, 0.0)
        dialog.add_atom_local(0.0, 0.0)
        dialog.add_atom_local(+1.0, 0.0)
        dialog.begin_bond(0); dialog.complete_bond(1)
        dialog.template.bonds[0].k = 111.0
        dialog.template.bonds[0].r_eq = 0.5
        dialog.begin_bond(1); dialog.complete_bond(2)
        dialog.template.bonds[1].k = 222.0
        dialog.template.bonds[1].r_eq = 1.5

        dialog.select_bond(0)
        assert float(dialog.in_k.text) == pytest.approx(111.0, rel=1e-3)
        assert float(dialog.in_r_eq.text) == pytest.approx(0.5, rel=1e-3)
