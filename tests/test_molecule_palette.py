"""Tests for the right-panel MoleculePaletteWidget and the AppController-
side dialog→sketch.molecules→palette refresh path.

Stubs the controller surface the widget needs (sketch, session.tools,
change_tool, actions). The widget is constructed directly — no UIManager
needed, since this is a unit test of widget behaviour, not UI integration.
"""

import pytest

import shared.config as config
from model.molecule import make_diatom, MoleculeTemplate, MoleculeAtom, MoleculeBond
from model.sketch import Sketch
from ui.molecule_palette_widget import MoleculePaletteWidget
from ui.molecule_tool import MoleculeTool
from core.tool_context import ToolContext
from core.scene import Scene
from core.session import Session


# =============================================================================
# Stub controller surface
# =============================================================================

class _StubActions:
    def __init__(self):
        self.opened_builder = False

    def open_molecule_builder_dialog(self, template=None):
        self.opened_builder = True


class _StubController:
    """Minimal controller surface for MoleculePaletteWidget.

    The widget reads:
    - self.controller.sketch
    - self.controller.session.tools  (tool registry: id → tool instance)
    - self.controller.change_tool(tool_id)
    - self.controller.actions.open_molecule_builder_dialog()
    """

    def __init__(self, sketch, session, layout):
        self.sketch = sketch
        self.session = session
        self.actions = _StubActions()
        self.last_tool_change = None

        # Build a real MoleculeTool so set_template / activate work
        from tests.conftest import FakeApp
        from core.scene import Scene
        fake = FakeApp(scene=Scene(skip_warmup=True), session=session, layout=layout)
        # Use the sketch from the test fixture rather than the FakeApp scene's
        fake.scene.sketch = sketch
        ctx = ToolContext(fake)
        self.molecule_tool = MoleculeTool(ctx)
        session.tools[config.TOOL_MOLECULE] = self.molecule_tool

    def change_tool(self, tool_id):
        self.last_tool_change = tool_id
        tool = self.session.tools.get(tool_id)
        if tool is not None and hasattr(tool, 'activate'):
            tool.activate()


@pytest.fixture
def controller(sketch, session, layout):
    return _StubController(sketch, session, layout)


@pytest.fixture
def palette(controller):
    return MoleculePaletteWidget(0, 0, 200, controller)


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:
    def test_dropdown_seeded_from_sketch(self, palette, sketch):
        """The dropdown options should match sketch.molecules at construct
        time (in iteration order)."""
        expected = list(sketch.molecules.keys())
        assert palette.dropdown.options == expected

    def test_builder_button_present(self, palette):
        assert palette.btn_builder is not None

    def test_empty_palette_shows_placeholder(self, sketch, session, layout):
        # Wipe the sketch palette before constructing the widget
        sketch.molecules.clear()
        controller = _StubController(sketch, session, layout)
        palette = MoleculePaletteWidget(0, 0, 200, controller)
        assert palette.dropdown.options == ["(no molecules)"]


# =============================================================================
# Refresh
# =============================================================================

class TestRefresh:
    def test_refresh_picks_up_new_molecules(self, palette, sketch):
        # Add a new molecule via the sketch
        custom = make_diatom(name="CustomMol")
        sketch.add_molecule(custom)
        palette.refresh()
        assert "CustomMol" in palette.dropdown.options

    def test_refresh_preserves_current_selection_if_name_remains(self, palette, sketch):
        # Select the second molecule
        if len(palette.dropdown.options) >= 2:
            palette.dropdown.selected_index = 1
            sel_name = palette.dropdown.get_selected()
            # Add another molecule (changes the option list)
            sketch.add_molecule(make_diatom(name="Extra"))
            palette.refresh()
            # Same name should still be selected
            assert palette.dropdown.get_selected() == sel_name

    def test_refresh_falls_back_to_zero_when_selected_removed(self, palette, sketch):
        # Remove the currently-selected name
        sel_name = palette.dropdown.get_selected()
        sketch.remove_molecule(sel_name)
        palette.refresh()
        # Index should be 0; the option there should NOT be the removed name
        assert palette.dropdown.selected_index == 0
        assert palette.dropdown.get_selected() != sel_name


# =============================================================================
# Selection callback
# =============================================================================

class TestSelectionCallback:
    def test_selecting_sets_molecule_tool_template(self, palette, controller):
        names = list(controller.sketch.molecules.keys())
        if not names:
            pytest.skip("No seeded molecules to test selection")
        name = names[0]
        palette._on_select_molecule(name)
        assert controller.molecule_tool.template is controller.sketch.get_molecule(name)

    def test_selecting_activates_molecule_tool(self, palette, controller):
        names = list(controller.sketch.molecules.keys())
        if not names:
            pytest.skip("No seeded molecules to test activation")
        palette._on_select_molecule(names[0])
        assert controller.last_tool_change == config.TOOL_MOLECULE

    def test_select_placeholder_is_no_op(self, palette, controller):
        palette._on_select_molecule("(no molecules)")
        assert controller.last_tool_change is None

    def test_select_unknown_name_is_no_op(self, palette, controller):
        palette._on_select_molecule("NoSuchMolecule")
        assert controller.last_tool_change is None


# =============================================================================
# Builder button → controller.actions.open_molecule_builder_dialog
# =============================================================================

class TestBuilderButton:
    def test_button_dispatch_via_handle_event(self, palette, controller):
        """Simulate a Button click event reaching the widget's handler.
        We don't synthesize pygame events; instead we directly invoke the
        click path via the button's `is_clicked` state (matches how the
        widget routes events)."""
        # The widget routes through btn_builder.handle_event → True triggers
        # the open dialog action. Mock that directly.
        import pygame
        ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
            'pos': (palette.btn_builder.rect.centerx,
                    palette.btn_builder.rect.centery),
            'button': 1,
        })
        # Buttons handle MOUSEBUTTONDOWN+MOUSEBUTTONUP; ensure the down
        # registers as a click via the button's standard event flow.
        consumed = palette.handle_event(ev)
        ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, {
            'pos': (palette.btn_builder.rect.centerx,
                    palette.btn_builder.rect.centery),
            'button': 1,
        })
        # Whether DOWN or UP triggers the click depends on Button impl;
        # try both and assert SOMETHING triggered the open action.
        palette.handle_event(ev_up)
        assert controller.actions.opened_builder is True


# =============================================================================
# AppController.apply_molecule_builder_dialog refreshes palette
# =============================================================================

class TestApplyRefreshesPalette:
    """The controller's apply_molecule_builder_dialog should call
    palette.refresh() after a successful save so the new molecule appears
    in the dropdown."""

    def test_apply_calls_refresh(self, scene, sketch, session, layout):
        from app.app_controller import AppController
        from core.status_bar import StatusBar
        from core.sound_manager import SoundManager

        # Stand up a minimal AppController bound to a real scene + sketch,
        # with a real UIManager-less stub whose only job is to expose
        # `ui.molecule_palette.refresh` for the test.
        class _StubApp:
            pass

        class _StubSession:
            def __init__(self):
                self.status = StatusBar()
                self.tools = {}

        class _SpyPalette:
            def __init__(self):
                self.refresh_count = 0

            def refresh(self):
                self.refresh_count += 1

        class _StubUI:
            def __init__(self):
                self.molecule_palette = _SpyPalette()

        # Reuse the existing scene+sketch (fixture creates both)
        scene.sketch = sketch

        app = _StubApp()
        app.scene = scene
        app.session = _StubSession()
        app.ui = _StubUI()
        app.input_handler = None

        ctrl = AppController.__new__(AppController)
        ctrl.app = app
        ctrl.sound_manager = SoundManager.get()
        ctrl._modal_stack = []
        ctrl.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}

        # Stand up a dialog that has been "Save"d
        from ui.molecule_builder_dialog import MoleculeBuilderDialog
        dlg = MoleculeBuilderDialog(0, 0, sketch)
        dlg.add_atom_local(0.0, 0.0)
        dlg.in_name.set_value("ApplySpyMol")
        dlg.apply = True
        dlg.done = True

        ctrl.apply_molecule_builder_dialog(dlg)

        assert "ApplySpyMol" in sketch.molecules
        assert app.ui.molecule_palette.refresh_count == 1

    def test_cancel_does_not_save_or_refresh(self, scene, sketch):
        from app.app_controller import AppController
        from core.status_bar import StatusBar
        from core.sound_manager import SoundManager

        class _StubApp: pass
        class _StubSession:
            def __init__(self):
                self.status = StatusBar()
                self.tools = {}
        class _SpyPalette:
            def __init__(self):
                self.refresh_count = 0
            def refresh(self):
                self.refresh_count += 1
        class _StubUI:
            def __init__(self):
                self.molecule_palette = _SpyPalette()

        scene.sketch = sketch
        app = _StubApp()
        app.scene = scene
        app.session = _StubSession()
        app.ui = _StubUI()
        app.input_handler = None

        ctrl = AppController.__new__(AppController)
        ctrl.app = app
        ctrl.sound_manager = SoundManager.get()
        ctrl._modal_stack = []
        ctrl.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}

        from ui.molecule_builder_dialog import MoleculeBuilderDialog
        dlg = MoleculeBuilderDialog(0, 0, sketch)
        dlg.add_atom_local(0.0, 0.0)
        dlg.in_name.set_value("CancelledMol")
        dlg.apply = False  # Cancelled
        dlg.done = True

        pre_palette_size = len(sketch.molecules)
        ctrl.apply_molecule_builder_dialog(dlg)

        assert "CancelledMol" not in sketch.molecules
        assert len(sketch.molecules) == pre_palette_size
        assert app.ui.molecule_palette.refresh_count == 0


# =============================================================================
# End-to-end: dialog → sketch → palette → MoleculeTool → AddMoleculeCommand
# =============================================================================

class TestEndToEnd:
    """Tie all four rounds together: author a molecule in the dialog, save
    it to the sketch, select it via the palette, place it via MoleculeTool,
    verify atoms+bonds in the Simulation."""

    def test_full_round_trip_dialog_to_simulation(self, scene, session, layout):
        sketch = scene.sketch  # Reuse the scene's sketch

        # --- R3: Author a triangle molecule in the dialog ---
        from ui.molecule_builder_dialog import MoleculeBuilderDialog
        dlg = MoleculeBuilderDialog(0, 0, sketch)
        dlg.add_atom_local(-1.0, 0.0)
        dlg.add_atom_local(+1.0, 0.0)
        dlg.add_atom_local(0.0, 1.0)
        dlg.begin_bond(0); dlg.complete_bond(1)
        dlg.begin_bond(1); dlg.complete_bond(2)
        dlg.begin_bond(2); dlg.complete_bond(0)
        dlg.in_name.set_value("Triangle")
        dlg.apply_to_sketch(sketch)

        assert "Triangle" in sketch.molecules
        assert len(sketch.molecules["Triangle"].atoms) == 3
        assert len(sketch.molecules["Triangle"].bonds) == 3

        # --- R4: Select the new molecule via the palette ---
        controller = _StubController(sketch, session, layout)
        palette = MoleculePaletteWidget(0, 0, 200, controller)
        # The dropdown was built before "Triangle" was added; refresh()
        # rebuilds the option list.
        palette.refresh()
        assert "Triangle" in palette.dropdown.options

        palette._on_select_molecule("Triangle")
        assert controller.molecule_tool.template.name == "Triangle"

        # --- R2: Place via AddMoleculeCommand ---
        from core.molecule_commands import AddMoleculeCommand
        cmd = AddMoleculeCommand(
            scene, controller.molecule_tool.template,
            world_pos=(25.0, 25.0),
        )
        scene.execute(cmd)

        # --- R1: Simulation has 3 atoms + 3 bonds ---
        sim = scene.simulation
        assert sim.count == 3
        assert sim.bond_count == 3
