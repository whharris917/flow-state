"""Tests for the menu→controller→action wiring added with the R3 UI work.

Verifies:
- AppController.run_demo_preset dispatches to the right preset function
- AppController.run_demo_preset syncs sim physics knobs out to the UI
  sliders/buttons so the per-frame UI→sim read doesn't clobber them
- AppController.open_lj_override_dialog pushes the right modal onto the stack
- AppController.apply_lj_override_dialog re-pushes the eps_ij_matrix to sim
"""

import numpy as np
import pytest

from app.app_controller import AppController
from core.scene import Scene
from core.session import Session
from core.sound_manager import SoundManager


class _FakeSlider:
    """Minimal SmartSlider stand-in for ui-sync tests. Captures the value
    that the controller writes via set_val()."""
    def __init__(self, initial=0.0):
        self.val = float(initial)

    def set_val(self, new_val):
        self.val = float(new_val)


class _FakeButton:
    """Minimal Button stand-in. Has the .active bool that the controller
    flips when syncing sim state out to the UI."""
    def __init__(self, active=False):
        self.active = bool(active)


class _FakeUi:
    """Minimal UI stand-in providing the sliders/buttons dicts the
    controller's _sync_sim_controls_to_ui helper expects."""
    def __init__(self):
        self.sliders = {
            'gravity': _FakeSlider(9.81),
            'temp': _FakeSlider(0.5),
            'damping': _FakeSlider(0.99),
            'dt': _FakeSlider(0.002),
            'skin': _FakeSlider(0.3),
        }
        self.buttons = {
            'thermostat': _FakeButton(active=False),
            'boundaries': _FakeButton(active=False),
        }


def _make_controller_stub(scene, with_ui=False):
    """Build a minimal AppController bound to `scene`.

    Stubs in just the attrs `run_demo_preset` and `open_lj_override_dialog`
    actually touch: app.scene, app.session, app.layout, app.switch_mode,
    sound_manager, ctx_vars, modal stack. Pass with_ui=True to also attach
    a _FakeUi with the sliders/buttons the sim-sync helper writes to.
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
    if with_ui:
        app.ui = _FakeUi()

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
        # Default n_per_species=600 → 1200 atoms total filling the
        # periodic box at ρ* ≈ 0.48.
        assert scene.simulation.count == 1200

    def test_micelles_preset_populates_simulation(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('micelles')
        # Default 24 surfactants × 5 = 120 surfactant atoms, in 4 pre-organized
        # micelles of 6. Solvent target 1000 but ~100 grid cells fall within
        # 1σ of a planned surfactant atom under the default geometry, so the
        # actually-placed solvent count is around 895-905. Total in the band
        # of 1015-1025. Assert a permissive range — the exact count depends
        # on which solvent grid cells happen to fall inside exclusion zones,
        # which is deterministic but coupled to the geometry constants.
        N = scene.simulation.count
        assert 1000 <= N <= 1130, f"Total atom count out of expected band: {N}"

    def test_crystal_anneal_preset_populates_simulation(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('crystal_anneal')
        # Default n_atoms=1600 (= 40²; PBC grid tiles exactly).
        assert scene.simulation.count == 1600

    def test_unknown_preset_is_safe_noop(self):
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('not_a_real_preset')
        # No atoms placed, no crash
        assert scene.simulation.count == 0

    def test_pbc_sparse_sets_periodic_boundary(self):
        """All four PBC presets should set boundary_mode = PERIODIC (==2).
        Sparse is the smallest of the four; if it doesn't set PBC, none of
        them will."""
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene)
        ctrl.run_demo_preset('pbc_sparse')
        assert scene.simulation.boundary_mode == 2

    def test_pbc_presets_atom_counts_monotonic_in_density(self):
        """The controller wraps preset_pbc_liquid with density 0.2/0.5/0.8/1.0
        — the resulting atom counts should be strictly increasing. Verifies
        the controller is passing distinct density args to the preset."""
        names = ['pbc_sparse', 'pbc_liquid', 'pbc_dense', 'pbc_packed']
        counts = []
        for n in names:
            scene = Scene(skip_warmup=True)
            ctrl, _ = _make_controller_stub(scene)
            ctrl.run_demo_preset(n)
            counts.append(scene.simulation.count)
        assert counts == sorted(counts)
        assert len(set(counts)) == 4

    def test_pbc_packed_uses_higher_target_temp(self):
        """The packed preset (ρ*=1.0) bumps target_temp above the other
        three to keep the system fluid. Verifies the controller didn't
        accidentally pass the same T for all of them."""
        scene_a = Scene(skip_warmup=True)
        ctrl_a, _ = _make_controller_stub(scene_a)
        ctrl_a.run_demo_preset('pbc_dense')
        T_dense = scene_a.simulation.target_temp

        scene_b = Scene(skip_warmup=True)
        ctrl_b, _ = _make_controller_stub(scene_b)
        ctrl_b.run_demo_preset('pbc_packed')
        T_packed = scene_b.simulation.target_temp

        assert T_packed > T_dense


# =============================================================================
# Sim → UI sync after preset (prevents the per-frame UI read from clobbering
# the preset's physics knobs)
# =============================================================================

class TestRunDemoPresetSyncsUi:
    """The per-frame app loop in MODE_SIM does
        sim.gravity = ui.sliders['gravity'].val
        sim.target_temp = ui.sliders['temp'].val
        sim.damping = ui.sliders['damping'].val
        sim.use_thermostat = ui.buttons['thermostat'].active
        sim.use_boundaries = ui.buttons['boundaries'].active
    Without an active sim→UI push after a preset runs, the very next frame
    would overwrite gravity/temp/damping/thermostat/boundary the preset
    just set. These tests pin the push down."""

    def test_pbc_dense_pushes_gravity_zero_to_slider(self):
        """The PBC presets set sim.gravity=0; the gravity slider must
        reflect that, otherwise the next frame restores the old gravity."""
        scene = Scene(skip_warmup=True)
        ctrl, app = _make_controller_stub(scene, with_ui=True)
        # Sanity: slider starts at the default DEFAULT_GRAVITY
        assert app.ui.sliders['gravity'].val == pytest.approx(9.81)
        ctrl.run_demo_preset('pbc_dense')
        assert app.ui.sliders['gravity'].val == pytest.approx(0.0)

    def test_pbc_presets_set_boundaries_button_active(self):
        """The PBC presets set boundary_mode=PERIODIC (==2). The boolean
        boundaries button must read True afterward; the use_boundaries
        setter has a no-downgrade-from-PERIODIC rule, so as long as the
        button stays True the next frame can't drop us back to OPEN."""
        scene = Scene(skip_warmup=True)
        ctrl, app = _make_controller_stub(scene, with_ui=True)
        assert app.ui.buttons['boundaries'].active is False
        ctrl.run_demo_preset('pbc_dense')
        assert app.ui.buttons['boundaries'].active is True

    def test_pbc_dense_pushes_target_temp_to_slider(self):
        scene = Scene(skip_warmup=True)
        ctrl, app = _make_controller_stub(scene, with_ui=True)
        ctrl.run_demo_preset('pbc_dense')
        # pbc_dense uses target_temp=0.7
        assert app.ui.sliders['temp'].val == pytest.approx(0.7)

    def test_pbc_dense_pushes_thermostat_active(self):
        scene = Scene(skip_warmup=True)
        ctrl, app = _make_controller_stub(scene, with_ui=True)
        assert app.ui.buttons['thermostat'].active is False
        ctrl.run_demo_preset('pbc_dense')
        assert app.ui.buttons['thermostat'].active is True

    def test_existing_presets_also_sync_gravity(self):
        """The fix applies universally — Demixing/Micelles/Crystal Anneal
        also set sim.gravity=0 and were silently being clobbered by the
        UI read on subsequent frames. After the sync, all three presets'
        gravity sliders should read 0."""
        for name in ['demixing', 'micelles', 'crystal_anneal']:
            scene = Scene(skip_warmup=True)
            ctrl, app = _make_controller_stub(scene, with_ui=True)
            ctrl.run_demo_preset(name)
            assert app.ui.sliders['gravity'].val == pytest.approx(0.0), (
                f"{name}: gravity slider not synced after preset"
            )

    def test_existing_presets_set_boundaries_button(self):
        """Existing presets use REFLECTING (mode 1), still truthy — button
        should be active so the next-frame use_boundaries write stays
        in a non-OPEN mode."""
        for name in ['demixing', 'micelles', 'crystal_anneal']:
            scene = Scene(skip_warmup=True)
            ctrl, app = _make_controller_stub(scene, with_ui=True)
            ctrl.run_demo_preset(name)
            assert app.ui.buttons['boundaries'].active is True, (
                f"{name}: boundaries button not synced after preset"
            )

    def test_sync_safe_when_no_ui_attr(self):
        """The test stub built with with_ui=False has no `app.ui` attribute.
        run_demo_preset must not crash — the sync helper short-circuits."""
        scene = Scene(skip_warmup=True)
        ctrl, _ = _make_controller_stub(scene, with_ui=False)
        ctrl.run_demo_preset('demixing')  # No raise = pass
        assert scene.simulation.count > 0  # Confirm the preset ran

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
