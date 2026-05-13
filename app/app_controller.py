"""
AppController - UI Actions and Coordination

Handles high-level application logic, user actions, and UI coordination.
Extracts business logic out of FlowStateApp.

Architecture:
- Uses self.scene for orchestration (rebuild, undo/redo)
- Uses self.sketch for CAD operations (geometry, constraints)
- Uses self.sim for physics parameters only
"""

import pygame
import math
import core.config as config
import core.utils as utils

from ui.ui_widgets import MaterialDialog, RotationDialog, AnimationDialog, ContextMenu, SaveAsNewDialog, ConfirmDialog
from ui import icons
from model.geometry import Line, Circle, Point
from core.definitions import CONSTRAINT_DEFS
from core.sound_manager import SoundManager

# Import commands
from core.commands import (
    RemoveEntityCommand, RemoveConstraintCommand, CompositeCommand,
    ToggleAnchorCommand, ToggleInfiniteCommand, SetPhysicalCommand,
    SetMaterialCommand, SetDriverCommand, SetEntityDynamicCommand,
    ResizeWorldCommand,
)


class AppController:
    """
    Handles high-level application logic, user actions, and UI coordination.
    """
    def __init__(self, app):
        self.app = app
        self.sound_manager = SoundManager.get()

        # Modal stack for dialogs/menus (replaces individual properties)
        self._modal_stack = []
        self.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}

    # =========================================================================
    # Modal Stack Management
    # =========================================================================

    def push_modal(self, modal, modal_type=None):
        """
        Push a modal dialog onto the stack.

        Args:
            modal: The modal dialog instance
            modal_type: Optional string identifier (e.g., 'context_menu', 'prop_dialog')
        """
        # Reset interaction state on UI tree to prevent stale button clicks
        # from re-triggering when the modal closes
        if hasattr(self.app, 'ui') and hasattr(self.app.ui, 'root'):
            self.app.ui.root.reset_interaction_state()

        self._modal_stack.append({'modal': modal, 'type': modal_type})

    def pop_modal(self):
        """Pop and return the top modal from the stack.

        AIT-003: Now resets interaction state to prevent ghost inputs.
        This enforces Modal Stack Symmetry - both push and pop reset state.
        """
        if self._modal_stack:
            modal = self._modal_stack.pop()['modal']

            # Reset interaction state on UI tree to prevent stale button clicks
            # from triggering unintended actions after modal closes (AIT-003)
            if hasattr(self.app, 'ui') and hasattr(self.app.ui, 'root'):
                self.app.ui.root.reset_interaction_state()

            return modal
        return None

    def get_active_modal(self):
        """Get the currently active (top) modal, or None if stack is empty."""
        if self._modal_stack:
            return self._modal_stack[-1]['modal']
        return None

    def get_active_modal_type(self):
        """Get the type of the currently active modal, or None if stack is empty."""
        if self._modal_stack:
            return self._modal_stack[-1]['type']
        return None

    def is_modal_active(self):
        """Check if any modal is currently active."""
        return len(self._modal_stack) > 0

    def close_modal(self, modal=None):
        """
        Close a specific modal or the top modal.

        Args:
            modal: Specific modal to close, or None to close the top modal
        """
        if modal is None:
            self.pop_modal()
        else:
            # Find and remove specific modal
            self._modal_stack = [m for m in self._modal_stack if m['modal'] is not modal]

    def close_all_modals(self):
        """Close all active modals."""
        self._modal_stack.clear()

    # =========================================================================
    # Property Accessors (Clean SoC)
    # =========================================================================
    
    @property
    def session(self):
        return self.app.session
    
    @property
    def scene(self):
        return self.app.scene
    
    @property
    def sketch(self):
        return self.app.scene.sketch
    
    @property
    def sim(self):
        return self.app.scene.simulation
    
    @property
    def renderer(self):
        return self.app.renderer

    # =========================================================================
    # Undo/Redo
    # =========================================================================
    
    def action_undo(self):
        # Try CAD command undo first, fall back to physics snapshot
        if self.scene.can_undo():
            self.scene.undo()
            self.session.status.set("Undo")
        else:
            if self.sim.undo():
                self.session.status.set("Undo (particles)")
            else:
                self.session.status.set("Nothing to undo")
        self._sync_world_size_input()
        self.sound_manager.play_sound('click')

    def action_redo(self):
        # Try CAD command redo first, fall back to physics snapshot
        if self.scene.can_redo():
            self.scene.redo()
            self.session.status.set("Redo")
        else:
            if self.sim.redo():
                self.session.status.set("Redo (particles)")
            else:
                self.session.status.set("Nothing to redo")
        self._sync_world_size_input()
        self.sound_manager.play_sound('click')

    def _sync_world_size_input(self):
        """Push current sim.world_size into the Resize-World input field.

        InputField.set_value is gated on `not self.active`, so this is
        a no-op while the user is typing into the field — that's the
        correct behavior. Called after undo/redo so a ResizeWorldCommand
        being un/redone updates the displayed value to reflect reality.
        """
        if self.session.input_world is not None:
            self.session.input_world.set_value(self.sim.world_size)

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def action_reset(self):
        self.scene.new()
        self.session.status.set("Reset Simulation")
        self.sound_manager.play_sound('click')
        
    def action_clear_particles(self):
        self.sim.clear()
        self.scene.rebuild()  # Rebuild static atoms
        self.session.status.set("Particles Cleared")
        self.sound_manager.play_sound('click')

    def action_resize_world(self, size_str):
        try:
            val = float(size_str)
        except ValueError:
            self.session.status.set("Invalid Size")
            return

        if self.sim.count > 0:
            # Particles present — confirm before destructive reset.
            # Center the dialog on screen using the existing layout dict.
            screen_w = self.app.layout.get('W', 800)
            screen_h = self.app.layout.get('H', 600)
            dlg_w, dlg_h = 360, 180
            x = (screen_w - dlg_w) // 2
            y = (screen_h - dlg_h) // 2
            dialog = ConfirmDialog(
                x, y,
                title="Resize World",
                message=(
                    f"Resize the world to {val} and reset the simulation?\n"
                    f"All {self.sim.count} particles will be cleared and physics\n"
                    f"settings will return to defaults."
                ),
                destructive=True,
            )
            dialog.pending_value = val   # stashed for the dispatcher
            self.push_modal(dialog, 'confirm_resize_dialog')
        else:
            self._do_resize_world(val)

    def _do_resize_world(self, val):
        """Apply the resize through the Command queue so Ctrl+Z reverts it.

        ResizeWorldCommand captures the pre-resize physics state (positions,
        velocities, world_size, etc.), applies the destructive resize, and
        runs scene.rebuild() to restore Compiler-emitted atoms. Undo restores
        the captured state and re-emits atoms.

        Going through scene.execute() is what gives Ctrl+Z the right behavior:
        AppController.action_undo prefers scene.can_undo() over sim.undo(),
        so without a CAD-level command the resize would only be reversible
        when the CAD undo stack is empty.
        """
        self.scene.execute(ResizeWorldCommand(self.scene, val))
        self.session.status.set(f"World Resized: {val}")
        self.sound_manager.play_sound('click')

    def apply_resize_confirm(self, dialog):
        """Dispatcher hook for 'confirm_resize_dialog' completion.

        Polled from actions.update() once dialog.done is True. Routes the
        outcome based on which flag the dialog set.

        On cancel, restore the input_world field to the actual world
        size so the displayed value doesn't lie about the current state
        (the user typed a new value before clicking Resize World; if
        they cancel, the field should return to the real world_size).
        """
        if dialog.confirmed:
            self._do_resize_world(dialog.pending_value)
        else:
            self.session.status.set("Resize cancelled")
            if self.session.input_world is not None:
                self.session.input_world.set_value(self.sim.world_size)
        self.close_modal(dialog)

    # =========================================================================
    # Editor Actions
    # =========================================================================

    def toggle_ghost_mode(self):
        self.session.show_wall_atoms = not getattr(self.session, 'show_wall_atoms', True)
        if 'mode_ghost' in self.app.ui.buttons:
            btn = self.app.ui.buttons['mode_ghost']
            btn.active = not self.session.show_wall_atoms
            btn.text = "Mode: Blueprint" if not self.session.show_wall_atoms else "Mode: Physical"
            btn.cached_surf = None  # Force text re-render
        state = "Blueprint" if not self.session.show_wall_atoms else "Physical"
        self.session.status.set(f"View Mode: {state}")
        self.sound_manager.play_sound('click')
        
    def toggle_extend(self):
        if self.session.selection.walls:
            toggled = 0
            for idx in self.session.selection.walls:
                if idx < len(self.sketch.entities):
                    entity = self.sketch.entities[idx]
                    if isinstance(entity, Line) and entity.ref:
                        cmd = ToggleInfiniteCommand(self.sketch, idx)
                        self.scene.execute(cmd)
                        toggled += 1
            if toggled > 0:
                self.scene.rebuild()
                self.session.status.set(f"Toggled Infinite on {toggled} ref line(s)")
                self.sound_manager.play_sound('click')
            else:
                self.session.status.set("Select ref lines to toggle infinite")
                self.sound_manager.play_sound('error')
            
    def toggle_editor_play(self):
        self.session.editor_paused = not self.session.editor_paused
        btn = self.app.ui.buttons['editor_play']
        if self.session.editor_paused:
            btn.icon = icons.get_icon('anim_play')
            btn.tooltip = "Play Animation"
        else:
            btn.icon = icons.get_icon('anim_pause')
            btn.tooltip = "Pause Animation"
        self.sound_manager.play_sound('click')
        
    def toggle_show_constraints(self):
        self.session.show_constraints = not self.session.show_constraints
        btn = self.app.ui.buttons['show_const']
        if self.session.show_constraints:
            btn.icon = icons.get_icon('hide')
            btn.tooltip = "Hide Constraints"
        else:
            btn.icon = icons.get_icon('unhide')
            btn.tooltip = "Show Constraints"
        self.sound_manager.play_sound('click')

    def atomize_selected(self):
        if self.session.selection.walls:
            atomized = 0
            de_atomized = 0
            for idx in self.session.selection.walls:
                if idx < len(self.sketch.entities):
                    entity = self.sketch.entities[idx]
                    # Toggle the physical flag via Command
                    new_state = not getattr(entity, 'physical', False)
                    cmd = SetPhysicalCommand(self.sketch, idx, new_state)
                    self.scene.execute(cmd)
                    if new_state:
                        atomized += 1
                    else:
                        de_atomized += 1
            self.scene.rebuild()
            if atomized > 0 and de_atomized > 0:
                self.session.status.set(f"Atomized {atomized}, de-atomized {de_atomized}")
            elif atomized > 0:
                self.session.status.set(f"Atomized {atomized} entities")
            else:
                self.session.status.set(f"De-atomized {de_atomized} entities")
        else:
            self.session.status.set("Select entities to toggle atomization")
        self.sound_manager.play_sound('click')

    def action_delete_selection(self):
        """Delete selected entities using commands for proper undo/redo.

        Handle Points (those owned by a ProcessObject) cascade to their
        owner — deleting the center handle of a Source/Sink removes the
        whole ProcessObject. Without this dispatch, RemoveEntityCommand
        would orphan the ProcessObject in scene.process_objects with no
        handle in the sketch.
        """
        if not self.session.selection.walls:
            self.session.status.set("Nothing selected")
            return

        from model.process_objects import Source, Sink
        from core.source_commands import DeleteSourceCommand
        from core.sink_commands import DeleteSinkCommand

        # Sort indices descending so RemoveEntityCommand index math is
        # stable even if the cascade-delete shrinks the entity list.
        indices = sorted(self.session.selection.walls, reverse=True)
        cmds = []
        for idx in indices:
            if 0 <= idx < len(self.sketch.entities):
                entity = self.sketch.entities[idx]
                owner = self.scene.get_process_object_for_handle(entity)
                if owner is not None:
                    if isinstance(owner, Source):
                        cmds.append(DeleteSourceCommand(self.scene, owner))
                        continue
                    if isinstance(owner, Sink):
                        cmds.append(DeleteSinkCommand(self.scene, owner))
                        continue
            cmds.append(RemoveEntityCommand(self.sketch, idx))

        if len(cmds) == 1:
            self.scene.execute(cmds[0])
        else:
            self.scene.execute(CompositeCommand(cmds))

        self.session.selection.walls.clear()
        self.session.selection.points.clear()
        self.session.status.set(f"Deleted {len(indices)} entities")
        self.sound_manager.play_sound('click')

    def action_delete_constraint(self):
        """Delete a constraint by index stored in ctx_vars."""
        if self.ctx_vars['const'] != -1:
            if self.ctx_vars['const'] < len(self.sketch.constraints):
                cmd = RemoveConstraintCommand(self.sketch, self.ctx_vars['const'])
                self.scene.execute(cmd)
                self.session.status.set("Deleted Constraint")
                self.ctx_vars['const'] = -1
                self.sound_manager.play_sound('click')

    # =========================================================================
    # Dialogs
    # =========================================================================

    def apply_material_from_dialog(self, dialog):
        mat = dialog.get_result()
        # Add material to sketch (this is a definition, not an entity mutation)
        self.sketch.add_material(mat)

        targets = []
        if self.session.selection.walls:
            targets = list(self.session.selection.walls)
        elif self.ctx_vars['wall'] != -1:
            targets = [self.ctx_vars['wall']]

        # Apply material to each target via Command
        for idx in targets:
            if idx < len(self.sketch.entities):
                cmd = SetMaterialCommand(self.sketch, idx, mat.name)
                self.scene.execute(cmd)

        self.scene.rebuild()
        self.session.status.set(f"Material Applied: {mat.name}")
        self.sound_manager.play_sound('click')

        # Sync the right-panel MaterialPropertyWidget to reflect changes
        if hasattr(self.app, 'ui') and hasattr(self.app.ui, 'material_widget'):
            self.app.ui.material_widget.refresh_from_selection()

    def apply_rotation_from_dialog(self, dialog):
        # Legacy rotation via entity.anim is deprecated
        # Animation should be done via constraint drivers instead
        self.session.status.set("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')

    def apply_animation_from_dialog(self, dialog):
        if self.ctx_vars['const'] != -1:
            cmd = SetDriverCommand(self.sketch, self.ctx_vars['const'], dialog.get_values())
            self.scene.execute(cmd)
            self.session.status.set("Animation Set")
            self.sound_manager.play_sound('click')

    def open_material_dialog(self):
        if not self.session.selection.walls and self.ctx_vars['wall'] == -1:
            self.session.status.set("Select a wall first")
            return
        mx, my = pygame.mouse.get_pos()
        target_idx = -1
        if self.session.selection.walls:
            target_idx = list(self.session.selection.walls)[0]
        elif self.ctx_vars['wall'] != -1:
            target_idx = self.ctx_vars['wall']
        current_mat = "Wall"
        if target_idx != -1 and target_idx < len(self.sketch.entities):
            current_mat = self.sketch.entities[target_idx].material_id
        dialog = MaterialDialog(mx, my, self.sketch, current_mat)
        self.push_modal(dialog, 'prop_dialog')

    def open_source_properties_dialog(self):
        """Open the Source Properties dialog for the right-clicked Source handle.

        Resolves the Source from the entity at `ctx_vars['wall']` via the
        Scene's handle→owner map. If the entity isn't a Source handle,
        bails out with a status message rather than opening an empty
        dialog.
        """
        from ui.ui_widgets import SourcePropertiesDialog
        from model.process_objects import Source

        wall_idx = self.ctx_vars.get('wall', -1)
        if wall_idx == -1 or wall_idx >= len(self.sketch.entities):
            self.session.status.set("No Source under cursor")
            return
        entity = self.sketch.entities[wall_idx]
        owner = self.scene.get_process_object_for_handle(entity)
        if not isinstance(owner, Source):
            self.session.status.set("Not a Source")
            return

        mx, my = pygame.mouse.get_pos()
        dialog = SourcePropertiesDialog(mx, my, owner, self.sketch)
        self.push_modal(dialog, 'source_properties_dialog')

    def apply_source_properties_from_dialog(self, dialog):
        """Apply Source dialog values via the existing command pattern.

        Bundles radius + properties changes into one CompositeCommand so
        the whole edit collapses to a single undo.
        """
        from model.process_objects import SourceProperties
        from core.source_commands import (
            SetSourceRadiusCommand, SetSourcePropertiesCommand,
        )
        from core.commands import CompositeCommand

        if not getattr(dialog, 'apply', False):
            return  # Cancelled

        source = dialog.source
        vals = dialog.get_values()

        new_props = SourceProperties(
            material_name=vals['material_name'],
            flux=vals['flux'],
            temperature=vals['temperature'],
            injection_direction=source.properties.injection_direction,
            injection_spread=source.properties.injection_spread,
        )

        cmds = []
        if vals['radius'] != source.radius:
            cmds.append(SetSourceRadiusCommand(source, vals['radius']))
        cmds.append(SetSourcePropertiesCommand(source, new_props))

        if len(cmds) == 1:
            self.scene.execute(cmds[0])
        else:
            self.scene.execute(CompositeCommand(cmds))

        self.session.status.set(
            f"Source: {vals['material_name']}, r={vals['radius']:.1f}, flux={vals['flux']:.3f}"
        )
        self.sound_manager.play_sound('click')

    def open_molecule_builder_dialog(self, template=None):
        """Open the Molecule Builder dialog. If `template` is provided, the
        dialog starts populated with a copy of that template for editing;
        otherwise a fresh template is created.
        """
        from ui.molecule_builder_dialog import MoleculeBuilderDialog

        mx = self.app.layout['W'] // 2 - 260
        my = self.app.layout['H'] // 2 - 270
        dialog = MoleculeBuilderDialog(mx, my, self.sketch, template=template)
        self.push_modal(dialog, 'molecule_builder_dialog')

    def apply_molecule_builder_dialog(self, dialog):
        """Save the dialog's working template into Sketch.molecules.

        Saves on Apply (Save button), no-op on Cancel. After saving,
        refresh the right-panel palette so the new molecule appears in
        the dropdown without requiring a UI rebuild.
        """
        if not getattr(dialog, 'apply', False):
            return
        dialog.apply_to_sketch(self.sketch)
        # Refresh the right-panel palette dropdown so the new molecule shows up.
        ui = getattr(self.app, 'ui', None)
        palette = getattr(ui, 'molecule_palette', None) if ui is not None else None
        if palette is not None:
            palette.refresh()
        self.session.status.set(
            f"Saved molecule '{dialog.template.name}' "
            f"({len(dialog.template.atoms)} atoms, "
            f"{len(dialog.template.bonds)} bonds)"
        )
        self.sound_manager.play_sound('click')

    # =========================================================================
    # R3 Demo Presets (Menu: Demos → ...)
    # =========================================================================

    def _sync_sim_controls_to_ui(self) -> None:
        """Push the simulation's current physics knobs out to the left-panel
        sliders / buttons. Mirror image of the per-frame UI→sim read in
        flow_state_app.update (which reads the sliders into sim each frame
        in MODE_SIM). For one-shot demo presets the preset is the source of
        truth — without this call, the next frame's UI read would silently
        clobber any sim attribute the preset wrote (gravity, target_temp,
        damping, use_thermostat) using whatever value the slider happened
        to hold from the previous session. boundary_mode is special-cased:
        the `boundaries` button is a bool, so we leave it set True if any
        non-OPEN mode is active and rely on the use_boundaries setter's
        no-downgrade-from-PERIODIC semantics to preserve PBC across frames.

        Safe no-op if the controller's app has no `ui` attribute (test
        stub path).
        """
        ui = getattr(self.app, 'ui', None)
        if ui is None:
            return
        sliders = getattr(ui, 'sliders', None) or {}
        buttons = getattr(ui, 'buttons', None) or {}

        def _set_slider(key, val):
            sld = sliders.get(key)
            if sld is None:
                return
            # set_val honours hard walls + expands the soft range to fit.
            sld.set_val(float(val))

        def _set_button(key, active):
            btn = buttons.get(key)
            if btn is None:
                return
            btn.active = bool(active)

        _set_slider('gravity', self.sim.gravity)
        _set_slider('temp', self.sim.target_temp)
        _set_slider('damping', self.sim.damping)
        _set_slider('dt', self.sim.dt)
        _set_slider('skin', self.sim.skin_distance)
        _set_button('thermostat', self.sim.use_thermostat)
        # `boundaries` is a bool; PERIODIC (==2) reads as truthy via the
        # use_boundaries property and the setter refuses to downgrade
        # PERIODIC when given True, so this preserves the preset's choice
        # of OPEN/REFLECTING/PERIODIC unambiguously.
        _set_button('boundaries', self.sim.boundary_mode != 0)

    def run_demo_preset(self, preset_name: str) -> None:
        """Run a demo preset by name. Clears the simulation, seeds the
        initial state, and ensures the app is in simulation mode so the
        user can hit Play and watch.

        Recognised names: 'demixing', 'micelles', 'crystal_anneal',
        'pbc_sparse', 'pbc_liquid', 'pbc_dense', 'pbc_packed'.
        """
        from core import demo_presets

        # Ensure we're in simulation mode so the user can see the physics
        # immediately. switch_mode lives on FlowStateApp.
        if self.session.mode != config.MODE_SIM:
            self.app.switch_mode(config.MODE_SIM)

        if preset_name == 'demixing':
            info = demo_presets.preset_demixing(self.scene)
            self.session.status.set(
                f"Demixing demo: {info['n_polar']} Polar + "
                f"{info['n_nonpolar']} Nonpolar — hit Play"
            )
        elif preset_name == 'micelles':
            info = demo_presets.preset_micelles(self.scene)
            self.session.status.set(
                f"Micelles demo: {info['n_micelles']} pre-arranged proto-"
                f"micelles ({info['n_surfactant_molecules']} surfactants) "
                f"in {info['n_solvent']} Polar solvent — hit Play"
            )
        elif preset_name == 'crystal_anneal':
            info = demo_presets.preset_crystal_anneal(self.scene)
            self.session.status.set(
                f"Crystal anneal: {info['n_atoms']} {info['material']} atoms at T="
                f"{self.sim.target_temp:.1f} — lower target_temp to crystallise"
            )
        # PBC density-spectrum demos. Same physics (Polar LJ fluid, periodic
        # walls, thermostat on), four canonical reduced densities. Lead asked
        # for "a dense liquid that fills the entire space at different
        # densities" — these cover gas → liquid → dense → overcompressed.
        # T bumped to 1.2 at ρ*=1.0 so the packed run doesn't freeze on
        # contact; the other three share T=0.7 (typical LJ liquid range).
        elif preset_name == 'pbc_sparse':
            info = demo_presets.preset_pbc_liquid(
                self.scene, density=0.20, target_temp=0.7)
            self.session.status.set(
                f"PBC sparse: {info['n_atoms']} {info['material']} atoms "
                f"at ρ*={info['density_achieved']:.2f}, T={self.sim.target_temp:.1f}"
            )
        elif preset_name == 'pbc_liquid':
            info = demo_presets.preset_pbc_liquid(
                self.scene, density=0.50, target_temp=0.7)
            self.session.status.set(
                f"PBC liquid: {info['n_atoms']} {info['material']} atoms "
                f"at ρ*={info['density_achieved']:.2f}, T={self.sim.target_temp:.1f}"
            )
        elif preset_name == 'pbc_dense':
            info = demo_presets.preset_pbc_liquid(
                self.scene, density=0.80, target_temp=0.7)
            self.session.status.set(
                f"PBC dense: {info['n_atoms']} {info['material']} atoms "
                f"at ρ*={info['density_achieved']:.2f}, T={self.sim.target_temp:.1f}"
            )
        elif preset_name == 'pbc_packed':
            info = demo_presets.preset_pbc_liquid(
                self.scene, density=1.00, target_temp=1.2)
            self.session.status.set(
                f"PBC packed: {info['n_atoms']} {info['material']} atoms "
                f"at ρ*={info['density_achieved']:.2f}, T={self.sim.target_temp:.1f}"
            )
        else:
            self.session.status.set(f"Unknown preset: {preset_name}")
            return
        # Push the preset's physics knobs out to the UI so the next
        # MODE_SIM frame doesn't read stale slider values back over them.
        self._sync_sim_controls_to_ui()
        self.sound_manager.play_sound('click')

    # =========================================================================
    # Cross-ε Override Dialog (Menu: Tools → Cross-ε Overrides...)
    # =========================================================================

    def open_lj_override_dialog(self) -> None:
        """Open the LJ cross-ε override editor. Lets the user view and
        tune the per-pair ε table that the LJ kernel uses for cross-species
        interactions. Saving an override calls
        sketch.set_lj_cross_override(...) and pushes the updated matrix
        to the simulation so the change takes effect immediately.
        """
        from ui.lj_override_dialog import LjOverrideDialog

        mx = self.app.layout['W'] // 2 - 240
        my = self.app.layout['H'] // 2 - 220
        dialog = LjOverrideDialog(mx, my, self.sketch)
        self.push_modal(dialog, 'lj_override_dialog')

    def apply_lj_override_dialog(self, dialog) -> None:
        """After the LJ override dialog closes, push the updated eps_ij_matrix
        from the sketch to the simulation so the kernel sees the changes
        on the next step. The dialog mutates sketch.lj_cross_overrides
        directly; we just rebuild the matrix here."""
        if not getattr(dialog, 'apply', False):
            return
        self.sim.set_eps_ij_matrix(self.sketch.build_eps_ij_matrix())
        self.session.status.set("Cross-ε overrides updated")
        self.sound_manager.play_sound('click')

    def open_rotation_dialog(self):
        # Legacy rotation dialog - show message about using constraint drivers
        self.session.status.set("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')
        
    def open_animation_dialog(self):
        if self.ctx_vars['const'] != -1:
            c = self.sketch.constraints[self.ctx_vars['const']]
            driver = getattr(c, 'driver', None)
            dialog = AnimationDialog(
                self.app.layout['W'] // 2,
                self.app.layout['H'] // 2,
                driver
            )
            self.push_modal(dialog, 'anim_dialog')

    def open_save_as_new_dialog(self, suggested_name, existing_names):
        """Open the Save as New Material dialog."""
        # Center the dialog
        mx = self.app.layout['W'] // 2 - 140
        my = self.app.layout['H'] // 2 - 70
        dialog = SaveAsNewDialog(mx, my, suggested_name, existing_names)
        self.push_modal(dialog, 'save_as_new_dialog')

    def apply_save_as_new_from_dialog(self, dialog):
        """Apply the result from Save as New dialog."""
        new_name = dialog.get_name()
        # Find the material property widget and complete the save
        # Note: app uses self.ui, not self.ui_manager
        if hasattr(self.app, 'ui') and hasattr(self.app.ui, 'material_widget'):
            mat_widget = self.app.ui.material_widget
            if mat_widget:
                mat_widget.complete_save_as_new(new_name)

    # =========================================================================
    # Context Menus
    # =========================================================================

    def get_context_options(self, target_type, idx1, idx2=None):
        options = []
        if target_type == 'wall':
            options = ["Properties", "Atomize"]
            # Add dynamic toggle option based on current state
            if idx1 < len(self.sketch.entities):
                entity = self.sketch.entities[idx1]
                is_dynamic = getattr(entity, 'dynamic', False)
                options.append("Make Static" if is_dynamic else "Make Dynamic")
            options.append("Delete")
        elif target_type == 'point':
            w_idx, pt_idx = idx1, idx2
            entities = self.sketch.entities
            if w_idx < len(entities):
                w = entities[w_idx]
                # Process-object handle: swap in object-specific options.
                # Sinks stay simple (Lead direction); Sources get a
                # properties dialog covering material / radius / rate / temperature.
                owner = self.scene.get_process_object_for_handle(w)
                if owner is not None:
                    from model.process_objects import Source
                    if isinstance(owner, Source):
                        options.append("Source Properties...")
                    options.append("Delete")
                    return options
                is_anchored = False
                if isinstance(w, Line):
                    is_anchored = w.anchored[pt_idx]
                elif isinstance(w, Circle):
                    is_anchored = w.anchored[0]
                elif isinstance(w, Point):
                    is_anchored = w.anchored
                options.append("Un-Anchor" if is_anchored else "Anchor")
                options.append("Set Length...")
        elif target_type == 'constraint':
            options = ["Delete Constraint", "Animate..."]
        return options

    def handle_context_menu_action(self, action):
        if action == "Properties":
            self.open_material_dialog()
        elif action == "Source Properties...":
            self.open_source_properties_dialog()
        elif action == "Animate...":
            self.open_animation_dialog()
        elif action == "Delete":
            self.action_delete_selection()
        elif action == "Delete Constraint":
            self.action_delete_constraint()
        elif action == "Anchor" or action == "Un-Anchor":
            if self.ctx_vars['wall'] != -1 and self.ctx_vars['pt'] is not None:
                cmd = ToggleAnchorCommand(self.sketch, self.ctx_vars['wall'], self.ctx_vars['pt'])
                self.scene.execute(cmd)
                self.sound_manager.play_sound('click')
        elif action == "Atomize":
            if self.ctx_vars['wall'] != -1:
                self.session.selection.walls.add(self.ctx_vars['wall'])
                self.atomize_selected()
                self.session.selection.walls.clear()
        elif action == "Make Dynamic":
            if self.ctx_vars['wall'] != -1:
                cmd = SetEntityDynamicCommand(self.sketch, self.ctx_vars['wall'], True)
                self.scene.execute(cmd)
                self.sound_manager.play_sound('snap')
                self.session.status.set("Entity set to Dynamic (two-way coupling)")
        elif action == "Make Static":
            if self.ctx_vars['wall'] != -1:
                cmd = SetEntityDynamicCommand(self.sketch, self.ctx_vars['wall'], False)
                self.scene.execute(cmd)
                self.sound_manager.play_sound('snap')
                self.session.status.set("Entity set to Static (immovable)")
        # Note: Context menu is already closed by InputHandler before this method is called

    def spawn_context_menu(self, pos):
        mx, my = pos
        sim_x, sim_y = utils.screen_to_sim(
            mx, my,
            self.session.camera.zoom, self.session.camera.pan_x, self.session.camera.pan_y,
            self.sim.world_size, self.app.layout
        )

        # Check points FIRST (highest priority for right-click)
        point_map = utils.get_grouped_points(
            self.sketch.entities,
            self.session.camera.zoom, self.session.camera.pan_x, self.session.camera.pan_y,
            self.sim.world_size, self.app.layout
        )
        hit_pt = None
        base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            if math.hypot(mx - center_pos[0], my - center_pos[1]) <= base_r + (len(items) - 1) * step_r:
                hit_pt = items[0]
                break

        if hit_pt:
            self.ctx_vars['wall'] = hit_pt[0]
            self.ctx_vars['pt'] = hit_pt[1]
            opts = self.get_context_options('point', hit_pt[0], hit_pt[1])
            self.push_modal(ContextMenu(mx, my, opts), 'context_menu')
            return

        # Check constraints second
        if self.session.show_constraints:
            layout_data = self.renderer._calculate_constraint_layout(
                self.sketch.constraints, self.sketch.entities,
                self.session.camera.zoom, self.session.camera.pan_x, self.session.camera.pan_y,
                self.sim.world_size, self.app.layout
            )
            for item in layout_data:
                # Build rect from x, y (badge is roughly 40x20 pixels)
                badge_rect = pygame.Rect(item['x'] - 20, item['y'] - 10, 40, 20)
                if badge_rect.collidepoint(mx, my):
                    const_idx = item['const_idx']
                    self.ctx_vars['const'] = const_idx
                    opts = self.get_context_options('constraint', const_idx)
                    self.push_modal(ContextMenu(mx, my, opts), 'context_menu')
                    return

        # Check walls/entities using sketch's find_entity_at
        rad_sim = 5.0 / (((self.app.layout['MID_W'] - 50) / self.sim.world_size) * self.session.camera.zoom)
        hit_wall = self.sketch.find_entity_at(sim_x, sim_y, rad_sim)
        
        if hit_wall != -1:
            self.ctx_vars['wall'] = hit_wall
            opts = self.get_context_options('wall', hit_wall)
            self.push_modal(ContextMenu(mx, my, opts), 'context_menu')
        else:
            if self.session.mode == config.MODE_EDITOR: 
                self.app.change_tool(config.TOOL_SELECT)
                self.session.status.set("Switched to Select Tool")

    # =========================================================================
    # Constraint Handling
    # =========================================================================

    def trigger_constraint(self, ctype):
        """
        Trigger constraint creation workflow.

        Uses ConstraintBuilder for all constraint logic.
        AppController only handles UI orchestration (button states, sounds, tool switch).
        """
        builder = self.session.constraint_builder

        # Update UI button states
        for btn, c_val in self.app.input_handler.constraint_btn_map.items():
            btn.active = (c_val == ctype)

        # Initialize builder with current selection
        builder.start(
            ctype,
            initial_walls=list(self.session.selection.walls),
            initial_points=list(self.session.selection.points)
        )

        # Handle multi-apply constraints - apply to selected entities
        # This handles: H/V (unary), LENGTH (context-aware), PARALLEL/EQUAL (binary)
        if builder.is_multi_apply():
            walls = list(self.session.selection.walls)
            # Binary constraints need at least 2, unary needs at least 1
            min_required = 2 if builder.is_binary_multi() else 1
            if len(walls) >= min_required:
                cmd = builder.build_multi_command(self.sketch, walls)
                if cmd:
                    self.scene.execute(cmd)
                    self._clear_constraint_state(f"Applied {ctype} to {len(walls)} items")
                    self.sound_manager.play_sound('click')
                    return

        # Try to apply immediately with current selection
        cmd = builder.try_build_command(self.sketch)
        if cmd:
            self.scene.execute(cmd)
            self._clear_constraint_state(f"Applied {ctype}")
            self.sound_manager.play_sound('click')
            return

        # Not enough targets - enter pending mode
        self.session.selection.walls.clear()
        self.session.selection.points.clear()

        # Auto-switch to SelectTool so user can immediately pick targets
        self.app.change_tool(config.TOOL_SELECT)

        msg = CONSTRAINT_DEFS[ctype][0]['msg'] if ctype in CONSTRAINT_DEFS else "Select targets..."
        self.session.status.set(f"{ctype}: {msg}")
        self.sound_manager.play_sound('tool_select')

    def _clear_constraint_state(self, status_msg):
        """Helper to clear constraint UI state after successful application."""
        self.session.constraint_builder.reset()
        self.session.selection.walls.clear()
        self.session.selection.points.clear()
        for btn in self.app.input_handler.constraint_btn_map.keys():
            btn.active = False
        self.session.status.set(status_msg)

    # =========================================================================
    # Lifecycle Updates
    # =========================================================================
    
    def update(self, dt):
        """Update all active modals."""
        for entry in self._modal_stack:
            modal = entry['modal']
            modal_type = entry['type']
            if hasattr(modal, 'update'):
                modal.update(dt)
            # Handle save_as_new_dialog completion specially
            if modal_type == 'save_as_new_dialog' and hasattr(modal, 'done') and modal.done:
                self.apply_save_as_new_from_dialog(modal)
                self.close_modal(modal)
                break  # Modal stack was modified, exit loop
            # Handle confirm_resize_dialog completion (per CR-114 §5.3,
            # tu_scene cycle 7 note — converge all dispatch paths through
            # this per-frame poll rather than the event-driven dispatcher)
            if modal_type == 'confirm_resize_dialog' and hasattr(modal, 'done') and modal.done:
                self.apply_resize_confirm(modal)
                break  # Modal stack was modified, exit loop
            # Source properties dialog: apply on OK, dismiss on Cancel
            if modal_type == 'source_properties_dialog' and hasattr(modal, 'done') and modal.done:
                self.apply_source_properties_from_dialog(modal)
                self.close_modal(modal)
                break  # Modal stack was modified, exit loop
            # Molecule Builder dialog: apply on Save, dismiss on Cancel
            if modal_type == 'molecule_builder_dialog' and hasattr(modal, 'done') and modal.done:
                self.apply_molecule_builder_dialog(modal)
                self.close_modal(modal)
                break  # Modal stack was modified, exit loop
            # LJ cross-ε override dialog: push updated matrix to sim on close
            if modal_type == 'lj_override_dialog' and hasattr(modal, 'done') and modal.done:
                self.apply_lj_override_dialog(modal)
                self.close_modal(modal)
                break  # Modal stack was modified, exit loop

    def draw_overlays(self, screen, font):
        """Draw all active modals in stack order (bottom to top)."""
        for entry in self._modal_stack:
            modal = entry['modal']
            if hasattr(modal, 'draw'):
                modal.draw(screen, font)