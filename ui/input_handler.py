"""
InputHandler - Explicit Chain of Responsibility

Handles all keyboard and mouse input using a 5-layer protocol:
1. System (Quit, Resize)
2. Global (Hotkeys)
3. Modals (Context Menu, Dialogs)
4. HUD (Panel Tree - NOW INCLUDES SCENE)
"""

import pygame
import sys

import core.config as config
import core.utils as utils

from ui.ui_widgets import MaterialDialog, RotationDialog, AnimationDialog, ContextMenu
from model.constraints import Length
from core.session import InteractionState
from tkinter import simpledialog


class InputHandler:
    def __init__(self, controller):
        # The 'controller' is the FlowStateApp instance
        self.controller = controller
        
        # Direct references
        self.session = controller.session
        self.sim = controller.sim  
        self.ui = controller.ui
        self.layout = controller.layout
        
        # Tool button mappings
        self.tool_btn_map = {}
        tool_defs = {
            'brush': config.TOOL_BRUSH, 
            'select': config.TOOL_SELECT,
            'line': config.TOOL_LINE, 
            'rect': config.TOOL_RECT,
            'circle': config.TOOL_CIRCLE, 
            'point': config.TOOL_POINT,
            'ref': config.TOOL_REF,
            'source': config.TOOL_SOURCE, 
        }
        for key, val in tool_defs.items():
            if key in self.ui.tools:
                self.tool_btn_map[self.ui.tools[key]] = val
        
        # Constraint button mappings
        self.constraint_btn_map = {}
        constraint_defs = {
            'const_length': 'LENGTH', 'const_equal': 'EQUAL',
            'const_parallel': 'PARALLEL', 'const_perp': 'PERPENDICULAR',
            'const_coincident': 'COINCIDENT', 'const_collinear': 'COLLINEAR',
            'const_midpoint': 'MIDPOINT', 'const_angle': 'ANGLE',
            'const_horiz': 'HORIZONTAL', 'const_vert': 'VERTICAL'
        }
        for key, val in constraint_defs.items():
            if key in self.ui.buttons:
                self.constraint_btn_map[self.ui.buttons[key]] = val

        # Action mappings
        self.ui_action_map = {}
        
        def bind_action(btn_key, action):
            if btn_key in self.ui.buttons:
                self.ui_action_map[self.ui.buttons[btn_key]] = action

        bind_action('reset', self.controller.actions.action_reset)
        bind_action('clear', self.controller.actions.action_clear_particles)
        bind_action('undo', self.controller.actions.action_undo)
        bind_action('redo', self.controller.actions.action_redo)
        
        bind_action('atomize', self.controller.actions.atomize_selected)
        bind_action('auto_atomize', lambda: setattr(self.session, 'auto_atomize', self.ui.buttons['auto_atomize'].active))
        bind_action('mode_ghost', self.controller.actions.toggle_ghost_mode)
        bind_action('extend', self.controller.actions.toggle_extend)
        bind_action('editor_play', self.controller.actions.toggle_editor_play)
        bind_action('show_const', self.controller.actions.toggle_show_constraints)
        
        bind_action('discard_geo', lambda: self.controller.exit_editor_mode(None))
        bind_action('save_geo', self.controller.save_geometry)
        
        if 'resize' in self.ui.buttons and 'world' in self.ui.inputs:
            self.ui_action_map[self.ui.buttons['resize']] = lambda: self.controller.actions.action_resize_world(
                self.ui.inputs['world'].get_value(50.0)
            )

        # BIND CALLBACKS TO UI WIDGETS
        self._bind_callbacks()

    def _bind_callbacks(self):
        """Attach logic callbacks to buttons for automatic dispatch."""
        # Bind Actions
        for btn, action in self.ui_action_map.items():
            btn.callback = action
            
        # Bind Tools
        for btn, tid in self.tool_btn_map.items():
            # Use default argument 't=tid' to capture loop variable
            btn.callback = lambda t=tid: self.controller.change_tool(t)
            
        # Bind Constraints
        for btn, ctype in self.constraint_btn_map.items():
            btn.callback = lambda c=ctype: self.controller.actions.trigger_constraint(c)

    def handle_input(self):
        """The Main 4-Layer Explicit Chain."""
        # Update layout reference in case of resize
        self.layout = self.controller.layout

        for event in pygame.event.get():
            # Layer 1: System
            if self._attempt_handle_system(event): continue

            # Layer 2: Modals (Context, Dialogs) - BEFORE global hotkeys!
            # Dialogs need to capture keyboard input before hotkeys trigger
            if self._attempt_handle_modals(event): continue

            # Layer 3: Global Hotkeys
            if self._attempt_handle_global(event): continue

            # Focus Management: Check for focus loss before HUD handling
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._check_focus_loss(event.pos)

            # Layer 4: HUD (UI Tree - Now includes SceneViewport)
            if self._attempt_handle_hud(event): continue

    def _attempt_handle_system(self, event):
        """Layer 1: System Events."""
        if event.type == pygame.QUIT:
            self.controller.running = False
            return True
        elif event.type == pygame.VIDEORESIZE:
            self.controller.handle_resize(event.w, event.h)
            return True
        return False

    def _attempt_handle_global(self, event):
        """Layer 2: Global Hotkeys."""
        if event.type == pygame.KEYDOWN:
            
            # UI Scaling (Ctrl + / Ctrl -)
            if event.key == pygame.K_EQUALS and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.controller.set_ui_scale(config.UI_SCALE + 0.1)
                return True
            if event.key == pygame.K_MINUS and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.controller.set_ui_scale(config.UI_SCALE - 0.1)
                return True

            if event.key == pygame.K_ESCAPE:
                if self.session.placing_geo_data:
                    self.session.placing_geo_data = None
                    self.session.status.set("Placement Cancelled")
                    return True
                if self.session.current_tool:
                    self.session.current_tool.cancel()
                self.session.constraint_builder.pending_type = None
                self.session.selection.walls.clear()
                self.session.selection.points.clear()
                for btn in self.constraint_btn_map.keys():
                    btn.active = False
                self.session.status.set("Cancelled")
                return True
            if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.controller.actions.action_undo()
                return True
            if event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.controller.actions.action_redo()
                return True
            if event.key == pygame.K_DELETE:
                self.controller.actions.action_delete_selection() 
                return True
            
            # Tool hotkeys
            tools = {
                pygame.K_b: config.TOOL_BRUSH,
                pygame.K_v: config.TOOL_SELECT,
                pygame.K_l: config.TOOL_LINE,
                pygame.K_r: config.TOOL_RECT,
                pygame.K_c: config.TOOL_CIRCLE,
                pygame.K_p: config.TOOL_POINT
            }
            if event.key in tools:
                self.controller.change_tool(tools[event.key])
                return True
                
            if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                self.controller.change_tool(config.TOOL_SOURCE)
                return True

            # Solver Benchmarking Controls (F9/F10/F11)
            sketch = self.controller.scene.sketch

            if event.key == pygame.K_F9:
                # Toggle Numba solver
                sketch.use_numba = not sketch.use_numba
                mode = "ON" if sketch.use_numba else "OFF"
                self.session.status.set(f"Solver: Numba {mode}")
                return True

            if event.key == pygame.K_F10:
                # Decrease solver iterations
                sketch.solver_iterations = max(1, sketch.solver_iterations - 10)
                self.session.status.set(f"Solver: Iterations {sketch.solver_iterations}")
                return True

            if event.key == pygame.K_F11:
                # Increase solver iterations
                sketch.solver_iterations += 10
                self.session.status.set(f"Solver: Iterations {sketch.solver_iterations}")
                return True

        return False

    def _attempt_handle_modals(self, event):
        """Layer 3: Modals (Context Menu, Dialogs)."""
        actions = self.controller.actions

        # Get active modal from stack
        modal = actions.get_active_modal()
        if not modal:
            return False

        modal_type = actions.get_active_modal_type()

        # Click-outside-to-dismiss logic
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is inside modal
            click_inside = modal.rect.collidepoint(event.pos)

            # Special case: Check if dropdown is expanded (for dialogs with dropdowns)
            dropdown_expanded = hasattr(modal, 'dropdown') and modal.dropdown.expanded
            if dropdown_expanded:
                expanded_rect = modal.dropdown.get_expanded_rect()
                if expanded_rect.collidepoint(event.pos):
                    click_inside = True

            if not click_inside:
                # Click outside - dismiss modal
                if modal_type == 'save_as_new_dialog':
                    modal.cancelled = True
                    modal.done = True
                else:
                    actions.close_modal()
                return True

        # Let modal handle the event
        if modal.handle_event(event):
            # Handle modal-specific completion logic
            if modal_type == 'context_menu':
                action = getattr(modal, 'action', None)
                actions.close_modal()
                if action and action != "CLOSE":
                    actions.handle_context_menu_action(action)

            elif modal_type == 'prop_dialog':
                if getattr(modal, 'apply', False):
                    actions.apply_material_from_dialog(modal)
                    modal.apply = False
                if getattr(modal, 'done', False):
                    actions.close_modal()

            elif modal_type == 'anim_dialog':
                if getattr(modal, 'done', False):
                    if getattr(modal, 'apply', False):
                        actions.apply_animation_from_dialog(modal)
                    actions.close_modal()

            # save_as_new_dialog completion handled in actions.update()

            return True

        return False

    def _check_focus_loss(self, mouse_pos):
        """
        Check if a click occurred outside the focused element and trigger on_focus_lost.

        This is called before HUD handling to give the focused element a chance
        to handle focus loss (e.g., clear unsaved preview changes).
        """
        focused = self.session.focused_element
        if not focused:
            return

        # Don't process focus loss while a modal is active (borrowed focus)
        if self.controller.actions.is_modal_active():
            return

        # Check if click is inside the focused element
        # Use the element's rect for hit testing
        if focused.rect.collidepoint(mouse_pos):
            return  # Click is inside, keep focus

        # Click is outside - notify element and clear focus
        focused.on_focus_lost()
        self.session.focused_element = None

    def _acquire_focus(self, element):
        """
        Attempt to give focus to an element.

        Only elements that return True from wants_focus() will receive focus.
        """
        if element and hasattr(element, 'wants_focus') and element.wants_focus():
            self.session.focused_element = element

    def _attempt_handle_hud(self, event):
        """Layer 4: HUD Panel Tree."""
        # 1. Menu Bar Special Case (Extract Return Value)
        menu_result = self.ui.menu.handle_event(event)
        if menu_result:
            if isinstance(menu_result, str):
                self._dispatch_menu(menu_result)
            return True

        # 2. Generic Tree Delegation
        # This propagates to panels -> SceneViewport -> Scene
        if self.ui.root.handle_event(event):
            # Check if we should acquire focus for MaterialPropertyWidget
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Find which widget was clicked and if it wants focus
                self._check_focus_acquisition(event.pos)
            return True

        return False

    def _check_focus_acquisition(self, mouse_pos):
        """
        Check if a focusable widget was clicked and acquire focus for it.

        This searches the right panel for MaterialPropertyWidget (or other
        widgets that implement wants_focus()).
        """
        # Check if click is in the material widget
        if hasattr(self.ui, 'material_widget'):
            mat_widget = self.ui.material_widget
            # Use parent rect (right panel) for focus area, matching the widget's expectations
            focus_rect = mat_widget.rect
            if hasattr(mat_widget, 'parent') and mat_widget.parent:
                focus_rect = mat_widget.parent.rect

            if focus_rect.collidepoint(mouse_pos):
                self._acquire_focus(mat_widget)

    def _dispatch_menu(self, selection):
        """Handle menu item selection."""
        if selection == "New Simulation":
            self.controller.new_scene()
        elif selection == "New Model":
            self.controller.new_scene()
            self.controller.switch_mode(config.MODE_EDITOR)
        elif selection == "Open...":
            self.controller.load_scene()
        elif selection == "Save":
            if self.session.current_sim_filepath:
                self.controller.save_scene(self.session.current_sim_filepath)
            else:
                self.controller.save_scene()
        elif selection == "Save As...":
            self.controller.save_scene()
        elif selection == "Import Geometry":
            self.controller.import_geometry()