"""Tests for the menu→controller→action wiring added with the R3 UI work.

Verifies:
- AppController.run_demo_preset dispatches to the right preset function
- AppController.open_lj_override_dialog pushes the right modal onto the stack
- AppController.apply_lj_override_dialog re-pushes the eps_ij_matrix to sim
"""

import numpy as np
import pytest

from app.app_controller import AppController
from core.scene import Scene
from core.session import Session
from core.sound_manager import SoundManager


def _make_controller_stub(scene):
    """Build a minimal AppController bound to `scene`.

    Stubs in just the attrs `run_demo_preset` and `open_lj_override_dialog`
    actually touch: app.scene, app.session, app.layout, app.switch_mode,
    sound_manager, ctx_vars, modal stack.
    """
    class _StubApp:
        pass

    app = _StubApp()
    app.scene = scene
    app.session = Session()
    # Layout fixture for dialog placement (centred at (W/2 - 240, H/2 - 220))
    app.layout = {'W': 1200, 'H': 800}
    # switch_mode lives on FlowStateApp; stub it as a no-op that records calls.
    app.switch_mode_calls = []
    app.switch_mode = lambda mode: app.switch_mode_calls.append(mode)

    ctrl = AppController.__new__(AppController)
    ctrl.app = app
    ctrl.sound_manager = SoundManager.get()
    ctrl._modal_stack = []
    ctrl.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}
    return ctrl, app


# =============================================================================
# Demo preset dispatch
# =============================================================================

class TestRunDemoPreset:
    def test_demixing_preset_populates_simulation(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('demixing')
        # Default n_per_species=80 → 160 atoms total
        assert scene.simulation.count == 160

    def test_micelles_preset_populates_simulation(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('micelles')
        # Default 100 solvent + 12 surfactants × 5 atoms = 100 + 60 = 160
        assert scene.simulation.count == 160

    def test_crystal_anneal_preset_populates_simulation(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('crystal_anneal')
        # Default n_atoms=120
        assert scene.simulation.count == 120

    def test_unknown_preset_is_safe_noop(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('not_a_real_preset')
        # No atoms placed, no crash
        assert scene.simulation.count == 0

    def test_preset_switches_to_sim_mode_if_needed(self):
        scene = Scene(skip_warmup=True)
        ctrl, app = _make_controller_stub(scene)
        import core.config as config
        app.session.mode = config.MODE_EDITOR  # start in editor mode
        ctrl.run_demo_preset('demixing')
        # switch_mode should have been called once with MODE_SIM
        assert config.MODE_SIM in app.switch_mode_calls


# =============================================================================
# LJ override dialog wiring
# =============================================================================

class TestOpenLjOverrideDialog:
    def test_open_pushes_modal(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.open_lj_override_dialog()
        assert len(ctrl._modal_stack) == 1
        # Top of stack should carry the right type tag
        assert ctrl._modal_stack[-1]['type'] == 'lj_override_dialog'

    def test_apply_pushes_matrix_to_sim_on_modified_dialog(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.open_lj_override_dialog()
        dialog = ctrl._modal_stack[-1]['modal']

        # Set an override via the dialog's internal helper
        names = list(scene.sketch.materials.keys())
        dialog.dropdown_a.selected_index = names.index("Water")
        dialog.dropdown_b.selected_index = names.index("Oil")
        dialog.in_eps.set_value(0.42)
        dialog._do_apply()
        # Dialog should now report it was modified
        assert dialog.apply is True

        # Controller pushes the new matrix to sim
        ctrl.apply_lj_override_dialog(dialog)
        polar_idx = scene.sketch.get_material_index("Water")
        oil_idx = scene.sketch.get_material_index("Oil")
        m = scene.simulation.eps_ij_matrix
        assert m[polar_idx, oil_idx] == pytest.approx(0.42, abs=0.01)

    def test_apply_does_nothing_if_dialog_not_modified(self):
        """Open the dialog, don't change anything, hit Done — sim should be
        unchanged."""
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.open_lj_override_dialog()
        dialog = ctrl._modal_stack[-1]['modal']

        # apply flag is False — no mutations occurred
        assert dialog.apply is False
        # Snapshot the matrix
        before = np.copy(scene.simulation.eps_ij_matrix)
        ctrl.apply_lj_override_dialog(dialog)
        # No change — apply_lj_override_dialog early-returns when apply is False
        after = scene.simulation.eps_ij_matrix
        assert np.array_equal(before, after)
