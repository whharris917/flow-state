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
            
            # Layer 2: Global Hotkeys
            if self._attempt_handle_global(event): continue
            
            # Layer 3: Modals (Context, Dialogs)
            if self._attempt_handle_modals(event): continue
            
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
        return False

    def _attempt_handle_modals(self, event):
        """Layer 3: Modals (Context Menu, Dialogs)."""
        # Context Menu
        if self.controller.actions.context_menu:
            menu = self.controller.actions.context_menu

            # Click-outside-to-dismiss: Check BEFORE delegating to menu
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not menu.rect.collidepoint(event.pos):
                    # Click outside menu - dismiss and consume event
                    self.controller.actions.context_menu = None
                    return True

            # Let menu handle the event (clicks inside, motion, etc.)
            if menu.handle_event(event):
                action = menu.action
                self.controller.actions.context_menu = None
                if action and action != "CLOSE":
                    self.controller.actions.handle_context_menu_action(action)
                return True
        
        # Property Dialog (MaterialDialog)
        if self.controller.actions.prop_dialog:
            dialog = self.controller.actions.prop_dialog

            # Click-outside-to-dismiss (but not if dropdown is expanded)
            if event.type == pygame.MOUSEBUTTONDOWN:
                dropdown_expanded = hasattr(dialog, 'dropdown') and dialog.dropdown.expanded
                if not dialog.rect.collidepoint(event.pos) and not dropdown_expanded:
                    # Also check if click is in expanded dropdown area
                    if not (dropdown_expanded and dialog.dropdown.get_expanded_rect().collidepoint(event.pos)):
                        self.controller.actions.prop_dialog = None
                        return True

            if dialog.handle_event(event):
                # Check for apply (with or without closing)
                if dialog.apply:
                    self.controller.actions.apply_material_from_dialog(dialog)
                    dialog.apply = False  # Reset so we don't re-apply
                if dialog.done:
                    self.controller.actions.prop_dialog = None
                return True

        # Rotation Dialog
        if self.controller.actions.rot_dialog:
            dialog = self.controller.actions.rot_dialog

            # Click-outside-to-dismiss
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not dialog.rect.collidepoint(event.pos):
                    self.controller.actions.rot_dialog = None
                    return True

            if dialog.handle_event(event):
                if dialog.done:
                    self.controller.actions.rot_dialog = None
                return True

        # Animation Dialog
        if self.controller.actions.anim_dialog:
            dialog = self.controller.actions.anim_dialog

            # Click-outside-to-dismiss
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not dialog.rect.collidepoint(event.pos):
                    self.controller.actions.anim_dialog = None
                    return True

            if dialog.handle_event(event):
                if dialog.done:
                    if dialog.apply:
                        self.controller.actions.apply_animation_from_dialog(dialog)
                    self.controller.actions.anim_dialog = None
                return True
                
        return False

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
            return True
            
        return False

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