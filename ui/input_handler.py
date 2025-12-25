"""
InputHandler - Central Input Processing

Handles all keyboard and mouse input, delegating to:
- Tools for scene interaction
- UI widgets for panel interaction
- Controller for menu actions
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
        self.sim = controller.sim  # For physics parameters (world_size, etc.)
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
            'source': config.TOOL_SOURCE,  # ProcessObject: Particle emitter
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

        # Action mappings - delegate to Controller (AppController)
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

    def handle_input(self):
        self.layout = self.controller.layout
        
        # Build UI list based on what is initialized
        physics_elements = [
            self.ui.buttons.get('play'), self.ui.buttons.get('clear'), 
            self.ui.buttons.get('reset'), self.ui.buttons.get('undo'), 
            self.ui.buttons.get('redo'), self.ui.buttons.get('thermostat'), 
            self.ui.buttons.get('boundaries'),
            *self.ui.sliders.values()
        ]
        
        editor_elements = [
            self.ui.buttons.get('mode_ghost'), self.ui.buttons.get('atomize'), 
            self.ui.buttons.get('save_geo'), self.ui.buttons.get('discard_geo'), 
            self.ui.buttons.get('editor_play'), self.ui.buttons.get('show_const'),
            self.ui.buttons.get('resize'), self.ui.inputs.get('world'),
            self.ui.buttons.get('extend'),
            *self.ui.tools.values(), 
            *self.constraint_btn_map.keys()
        ]
        
        # Filter None values
        ui_list = [el for el in physics_elements + editor_elements if el is not None]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.controller.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.controller.handle_resize(event.w, event.h)
            
            if self._handle_keys(event):
                continue
            
            tool_switched = False
            for btn, tid in self.tool_btn_map.items():
                if btn.handle_event(event): 
                    self.controller.change_tool(tid)
                    tool_switched = True
            if tool_switched:
                continue

            if self._handle_menus(event):
                continue
            if self._handle_dialogs(event):
                continue

            ui_interacted = False
            for el in ui_list:
                if el.handle_event(event):
                    ui_interacted = True
                    if el in self.constraint_btn_map:
                        self.controller.actions.trigger_constraint(self.constraint_btn_map[el])
                    elif el in self.ui_action_map:
                        self.ui_action_map[el]()
            
            mouse_on_ui = (
                event.type == pygame.MOUSEBUTTONDOWN and 
                (event.pos[0] > self.layout['RIGHT_X'] or 
                 event.pos[0] < self.layout['LEFT_W'] or 
                 event.pos[1] < config.TOP_MENU_H)
            )
            
            if not mouse_on_ui and not ui_interacted:
                self._handle_scene_mouse(event)

    def _handle_keys(self, event):
        if event.type == pygame.KEYDOWN:
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
            if event.key == pygame.K_b:
                self.controller.change_tool(config.TOOL_BRUSH)
                return True
            if event.key == pygame.K_v:
                self.controller.change_tool(config.TOOL_SELECT)
                return True
            if event.key == pygame.K_l:
                self.controller.change_tool(config.TOOL_LINE)
                return True
            if event.key == pygame.K_r:
                self.controller.change_tool(config.TOOL_RECT)
                return True
            if event.key == pygame.K_c:
                self.controller.change_tool(config.TOOL_CIRCLE)
                return True
            if event.key == pygame.K_p:
                self.controller.change_tool(config.TOOL_POINT)
                return True
            if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                # Shift+S for Source tool
                self.controller.change_tool(config.TOOL_SOURCE)
                return True
                
        return False

    def _handle_menus(self, event):
        result = self.ui.menu.handle_event(event)
        if result:
            # If result is a string, it's a menu action
            if isinstance(result, str):
                self._dispatch_menu(result)
                return True
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and self.ui.menu.active_menu:
            if self.ui.menu.dropdown_rect and self.ui.menu.dropdown_rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.ui.menu.dropdown_rect.y - 5
                idx = rel_y // 30
                opts = self.ui.menu.items[self.ui.menu.active_menu]
                if 0 <= idx < len(opts):
                    self._dispatch_menu(opts[idx])
            self.ui.menu.active_menu = None
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

    def _handle_dialogs(self, event):
        """Handle active dialog events."""
        if self.controller.actions.context_menu:
            if self.controller.actions.context_menu.handle_event(event):
                action = self.controller.actions.context_menu.action
                self.controller.actions.context_menu = None
                if action and action != "CLOSE":
                    self.controller.actions.handle_context_menu_action(action)
                return True
        
        if self.controller.actions.prop_dialog:
            if self.controller.actions.prop_dialog.handle_event(event):
                return True
            if self.controller.actions.prop_dialog.done:
                self.controller.actions.prop_dialog = None
                return True
        
        if self.controller.actions.rot_dialog:
            if self.controller.actions.rot_dialog.handle_event(event):
                return True
            if self.controller.actions.rot_dialog.done:
                self.controller.actions.rot_dialog = None
                return True
        
        if self.controller.actions.anim_dialog:
            if self.controller.actions.anim_dialog.handle_event(event):
                return True
            if self.controller.actions.anim_dialog.done:
                self.controller.actions.anim_dialog = None
                return True
        
        return False

    def _handle_scene_mouse(self, event):
        """Handle mouse events in the viewport."""
        # Let the current tool handle it first
        if self.session.current_tool:
            if self.session.current_tool.handle_event(event, self.layout):
                return
        
        # Panning with middle mouse button
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            self.session.state = InteractionState.PANNING
            return
        
        if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            if self.session.state == InteractionState.PANNING:
                self.session.state = InteractionState.IDLE
            return
        
        if event.type == pygame.MOUSEMOTION:
            if self.session.state == InteractionState.PANNING:
                self.session.camera.apply_pan(event.rel[0], event.rel[1])
                return
        
        # Zooming with scroll wheel
        if event.type == pygame.MOUSEWHEEL:
            self.session.camera.apply_zoom(event.y)
            return
        
        # Right-click for context menu
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.controller.actions.spawn_context_menu(event.pos)
            return
