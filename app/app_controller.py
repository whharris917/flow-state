import pygame
import core.config as config

import math
import core.utils as utils

from ui.ui_widgets import MaterialDialog, RotationDialog, AnimationDialog, ContextMenu
from model.constraints import Length
from model.geometry import Line, Circle
from ui.tools import PointTool
from core.definitions import CONSTRAINT_DEFS
from core.sound_manager import SoundManager

# Phase 8b: Import commands
from core.commands import RemoveEntityCommand, RemoveConstraintCommand, CompositeCommand


class AppController:
    """
    Handles high-level application logic, user actions, and UI coordination.
    Extracts business logic out of FlowStateApp.
    """
    def __init__(self, app):
        self.app = app
        self.sim = app.sim
        self.session = app.session
        self.sketch = app.sketch
        self.renderer = app.renderer
        self.sound_manager = SoundManager.get()
        
        # State for dialogs/menus
        self.context_menu = None
        self.prop_dialog = None
        self.rot_dialog = None
        self.anim_dialog = None
        self.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}

    # --- SIMULATION CONTROL ---
    
    def action_undo(self):
        # Try command-based undo first, fall back to snapshot-based
        if self.app.scene.can_undo():
            self.app.scene.undo()
            self.session.set_status("Undo")
        else:
            self.sim.undo()
            self.session.set_status("Undo (snapshot)")
        self.sound_manager.play_sound('click')

    def action_redo(self):
        # Try command-based redo first, fall back to snapshot-based
        if self.app.scene.can_redo():
            self.app.scene.redo()
            self.session.set_status("Redo")
        else:
            self.sim.redo()
            self.session.set_status("Redo (snapshot)")
        self.sound_manager.play_sound('click')

    def action_reset(self):
        self.sim.reset_simulation()
        self.app.scene.commands.clear()  # Clear command history on reset
        self.session.set_status("Reset Simulation")
        self.sound_manager.play_sound('click')
        
    def action_clear_particles(self):
        self.sim.clear_particles()
        self.session.set_status("Particles Cleared")
        self.sound_manager.play_sound('click')

    def action_resize_world(self, size_str):
        try:
            val = float(size_str)
            self.sim.resize_world(val)
            self.session.set_status(f"World Resized: {val}")
            self.sound_manager.play_sound('click')
        except ValueError:
            self.session.set_status("Invalid Size")

    # --- EDITOR ACTIONS ---

    def toggle_ghost_mode(self):
        self.session.show_wall_atoms = not getattr(self.session, 'show_wall_atoms', True)
        if 'mode_ghost' in self.app.ui.buttons:
            self.app.ui.buttons['mode_ghost'].active = not self.session.show_wall_atoms 
        state = "Hidden" if not self.session.show_wall_atoms else "Visible"
        self.session.set_status(f"Wall Atoms: {state}")
        self.sound_manager.play_sound('click')
        
    def toggle_extend(self):
        if self.session.selected_walls:
            for idx in self.session.selected_walls:
                if idx < len(self.sim.walls) and isinstance(self.sim.walls[idx], Line): 
                    if not hasattr(self.sim.walls[idx], 'infinite'): self.sim.walls[idx].infinite = False
                    self.sim.walls[idx].infinite = not self.sim.walls[idx].infinite
            self.sim.rebuild_static_atoms(); self.session.set_status("Toggled Extend")
            self.sound_manager.play_sound('click')
            
    def toggle_editor_play(self):
        self.session.editor_paused = not self.session.editor_paused
        self.app.ui.buttons['editor_play'].text = "Play" if self.session.editor_paused else "Pause"
        self.app.ui.buttons['editor_play'].cached_surf = None
        self.sound_manager.play_sound('click')
        
    def toggle_show_constraints(self):
        self.session.show_constraints = not self.session.show_constraints
        self.app.ui.buttons['show_const'].text = "Show Cnstr" if not self.session.show_constraints else "Hide Cnstr"
        self.app.ui.buttons['show_const'].cached_surf = None
        self.sound_manager.play_sound('click')

    def atomize_selected(self):
        if self.session.selected_walls:
            count = 0
            for idx in self.session.selected_walls:
                if idx < len(self.sim.walls):
                    self.sim.update_wall_props(idx, {'physical': True})
                    count += 1
            self.sim.rebuild_static_atoms()
            self.session.set_status(f"Atomized {count} entities")
            self.sound_manager.play_sound('click')
        else:
            self.sim.rebuild_static_atoms()
            self.session.set_status("Atomized All Geometry")
            self.sound_manager.play_sound('click')

    def action_delete_selection(self):
        """
        Delete selected entities using commands for proper undo/redo.
        Phase 8b: Uses RemoveEntityCommand.
        """
        deleted_count = 0
        
        if self.session.selected_walls:
            # Delete in reverse order to preserve indices
            indices_to_delete = sorted(list(self.session.selected_walls), reverse=True)
            
            if len(indices_to_delete) == 1:
                # Single delete - simple command
                idx = indices_to_delete[0]
                cmd = RemoveEntityCommand(self.sketch, idx)
                self.app.scene.execute(cmd)
                deleted_count = 1
            else:
                # Multiple deletes - use composite command
                # Note: indices shift as we delete, so we need to be careful
                # RemoveEntityCommand stores the entity data, so we can delete in reverse
                # and restore in forward order
                commands = []
                for idx in indices_to_delete:
                    commands.append(RemoveEntityCommand(self.sketch, idx))
                
                composite = CompositeCommand(commands, "Delete Selection")
                self.app.scene.execute(composite)
                deleted_count = len(indices_to_delete)
            
            self.session.selected_walls.clear()
            self.session.selected_points.clear()
            self.sound_manager.play_sound('click')
            self.session.set_status(f"Deleted {deleted_count} Items")
            
        elif self.ctx_vars['wall'] != -1:
            # Context menu delete
            cmd = RemoveEntityCommand(self.sketch, self.ctx_vars['wall'])
            self.app.scene.execute(cmd)
            self.ctx_vars['wall'] = -1
            self.session.set_status("Deleted Item")
            self.sound_manager.play_sound('click')

    def action_delete_constraint(self):
        """
        Delete constraint using command for proper undo/redo.
        Phase 8b: Uses RemoveConstraintCommand.
        """
        if self.ctx_vars['const'] != -1:
            if self.ctx_vars['const'] < len(self.sim.constraints):
                cmd = RemoveConstraintCommand(self.sketch, self.ctx_vars['const'])
                self.app.scene.execute(cmd)
                self.session.set_status("Deleted Constraint")
                self.ctx_vars['const'] = -1
                self.sound_manager.play_sound('click')

    # --- DIALOGS ---

    def apply_material_from_dialog(self, dialog):
        mat = dialog.get_result()
        self.sketch.add_material(mat)
        
        targets = []
        if self.session.selected_walls: targets = list(self.session.selected_walls)
        elif self.ctx_vars['wall'] != -1: targets = [self.ctx_vars['wall']]
            
        for idx in targets:
             if idx < len(self.sim.walls):
                 self.sim.update_wall_props(idx, {'material_id': mat.name})
        
        self.session.set_status(f"Material Applied: {mat.name}")
        self.sound_manager.play_sound('click')

    def apply_rotation_from_dialog(self, dialog):
        # Legacy rotation via entity.anim is removed.
        # Animation should be done via constraint drivers instead.
        # This method is kept for compatibility but does nothing.
        self.session.set_status("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')

    def apply_animation_from_dialog(self, dialog):
        if self.ctx_vars['const'] != -1:
            self.sketch.set_driver(self.ctx_vars['const'], dialog.get_values())
            self.session.set_status("Animation Set")
            self.sound_manager.play_sound('click')

    def open_material_dialog(self):
        if not self.session.selected_walls and self.ctx_vars['wall'] == -1: 
            self.session.set_status("Select a wall first")
            return
        mx, my = pygame.mouse.get_pos()
        target_idx = -1
        if self.session.selected_walls: target_idx = list(self.session.selected_walls)[0]
        elif self.ctx_vars['wall'] != -1: target_idx = self.ctx_vars['wall']
        current_mat = "Default"
        if target_idx != -1 and target_idx < len(self.sim.walls):
             current_mat = self.sim.walls[target_idx].material_id
        self.prop_dialog = MaterialDialog(mx, my, self.sketch, current_mat)

    def open_rotation_dialog(self):
        # Legacy rotation dialog - show message about using constraint drivers
        self.session.set_status("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')
        
    def open_animation_dialog(self):
        if self.ctx_vars['const'] != -1:
             c = self.sim.constraints[self.ctx_vars['const']]
             driver = getattr(c, 'driver', None)
             self.anim_dialog = AnimationDialog(self.app.layout['W']//2, self.app.layout['H']//2, driver)

    # --- CONTEXT MENUS ---

    def get_context_options(self, target_type, idx1, idx2=None):
        options = []
        if target_type == 'wall':
            options = ["Properties", "Atomize", "Delete"]
        elif target_type == 'point':
            w_idx, pt_idx = idx1, idx2
            walls = self.sim.walls
            if w_idx < len(walls):
                w = walls[w_idx]
                is_anchored = False
                if isinstance(w, Line): is_anchored = w.anchored[pt_idx]
                elif isinstance(w, Circle): is_anchored = w.anchored[0]
                elif isinstance(w, PointTool): is_anchored = w.anchored 
                options.append("Un-Anchor" if is_anchored else "Anchor")
                options.append("Set Length...") 
        elif target_type == 'constraint':
            options = ["Delete Constraint", "Animate..."]
        return options

    def handle_context_menu_action(self, action):
        if action == "Properties": self.open_material_dialog()
        elif action == "Animate...": self.open_animation_dialog()
        elif action == "Delete": self.action_delete_selection()
        elif action == "Delete Constraint": self.action_delete_constraint()
        elif action == "Anchor" or action == "Un-Anchor":
            if self.ctx_vars['wall'] != -1 and self.ctx_vars['pt'] is not None:
                self.sim.toggle_anchor(self.ctx_vars['wall'], self.ctx_vars['pt'])
                self.sound_manager.play_sound('click')
        elif action == "Atomize":
            if self.ctx_vars['wall'] != -1:
                self.session.selected_walls.add(self.ctx_vars['wall'])
                self.atomize_selected()
                self.session.selected_walls.clear()
        self.context_menu = None

    def spawn_context_menu(self, pos):
        mx, my = pos
        sim_x, sim_y = utils.screen_to_sim(mx, my, self.session.zoom, self.session.pan_x, self.session.pan_y, self.sim.world_size, self.app.layout)
        
        if self.session.show_constraints:
            layout_data = self.renderer._calculate_constraint_layout(
                self.sim.constraints, self.sim.walls, 
                self.session.zoom, self.session.pan_x, self.session.pan_y, 
                self.sim.world_size, self.app.layout
            )
            for item in layout_data:
                if item['rect'].collidepoint(mx, my):
                    const_idx = item['index']
                    self.ctx_vars['const'] = const_idx
                    opts = self.get_context_options('constraint', const_idx)
                    self.context_menu = ContextMenu(mx, my, opts)
                    return

        point_map = utils.get_grouped_points(self.sim.walls, self.session.zoom, self.session.pan_x, self.session.pan_y, self.sim.world_size, self.app.layout)
        hit_pt = None; base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            if math.hypot(mx - center_pos[0], my - center_pos[1]) <= base_r + (len(items) - 1) * step_r: hit_pt = items[0]; break

        if hit_pt:
            self.ctx_vars['wall'] = hit_pt[0]; self.ctx_vars['pt'] = hit_pt[1]
            opts = self.get_context_options('point', hit_pt[0], hit_pt[1])
            self.context_menu = ContextMenu(mx, my, opts)
            if self.session.pending_constraint: self.handle_pending_constraint_click(pt_idx=hit_pt)
            return

        rad_sim = 5.0 / (((self.app.layout['MID_W'] - 50) / self.sim.world_size) * self.session.zoom)
        hit_wall = self.sketch.find_entity_at(sim_x, sim_y, rad_sim)
        
        if hit_wall != -1:
            if self.session.pending_constraint: self.handle_pending_constraint_click(wall_idx=hit_wall)
            else:
                self.ctx_vars['wall'] = hit_wall
                opts = self.get_context_options('wall', hit_wall)
                self.context_menu = ContextMenu(mx, my, opts)
        else:
            if self.session.mode == config.MODE_EDITOR: 
                self.app.change_tool(config.TOOL_SELECT)
                self.session.set_status("Switched to Select Tool")

    # --- CONSTRAINT TRIGGERS ---

    def trigger_constraint(self, ctype):
        from core.commands import AddConstraintCommand
        
        for btn, c_val in self.app.input_handler.constraint_btn_map.items(): btn.active = (c_val == ctype)
        is_multi = False
        if ctype in CONSTRAINT_DEFS and CONSTRAINT_DEFS[ctype][0].get('multi'):
            walls = list(self.session.selected_walls)
            if walls:
                count = 0
                for w_idx in walls:
                    if self.sim.attempt_apply_constraint(ctype, [w_idx], []): count += 1
                if count > 0:
                    self.session.set_status(f"Applied {ctype} to {count} items")
                    self.session.selected_walls.clear(); self.session.selected_points.clear()
                    self.sim.apply_constraints(); is_multi = True
                    self.sound_manager.play_sound('click')
        if is_multi: return
        
        walls = list(self.session.selected_walls); pts = list(self.session.selected_points)
        if self._try_apply_constraint_smart(ctype, walls, pts): return

        self.session.pending_constraint = ctype
        self.session.pending_targets_walls = list(self.session.selected_walls)
        self.session.pending_targets_points = list(self.session.selected_points)
        self.session.selected_walls.clear(); self.session.selected_points.clear()
        msg = CONSTRAINT_DEFS[ctype][0]['msg'] if ctype in CONSTRAINT_DEFS else "Select targets..."
        self.session.set_status(f"{ctype}: {msg}")
        self.sound_manager.play_sound('tool_select')

    def _try_apply_constraint_smart(self, ctype, walls, pts):
        if self.sim.attempt_apply_constraint(ctype, walls, pts):
            self.session.set_status(f"Applied {ctype}")
            self.session.selected_walls.clear(); self.session.selected_points.clear()
            self.session.pending_constraint = None
            for btn in self.app.input_handler.constraint_btn_map.keys(): btn.active = False
            self.sim.apply_constraints()
            self.sound_manager.play_sound('click')
            return True
        if ctype in CONSTRAINT_DEFS:
            rules = CONSTRAINT_DEFS[ctype]
            for r in rules:
                if len(walls) >= r['w'] and len(pts) >= r['p']:
                    sub_w = walls[:r['w']]
                    sub_p = pts[:r['p']]
                    if self.sim.attempt_apply_constraint(ctype, sub_w, sub_p):
                         self.session.set_status(f"Applied {ctype} (Auto-Trimmed)")
                         self.session.selected_walls.clear(); self.session.selected_points.clear()
                         self.session.pending_constraint = None
                         for btn in self.app.input_handler.constraint_btn_map.keys(): btn.active = False
                         self.sim.apply_constraints()
                         self.sound_manager.play_sound('click')
                         return True
        return False

    def handle_pending_constraint_click(self, wall_idx=None, pt_idx=None):
        if not self.session.pending_constraint: return
        if wall_idx is not None and wall_idx not in self.session.pending_targets_walls: self.session.pending_targets_walls.append(wall_idx)
        if pt_idx is not None and pt_idx not in self.session.pending_targets_points: self.session.pending_targets_points.append(pt_idx)
        
        if self._try_apply_constraint_smart(self.session.pending_constraint, self.session.pending_targets_walls, self.session.pending_targets_points):
            return
            
        ctype = self.session.pending_constraint; msg = CONSTRAINT_DEFS[ctype][0]['msg']
        self.session.set_status(f"{ctype} ({len(self.session.pending_targets_walls)}P, {len(self.session.pending_targets_points)}P): {msg}")
        self.sound_manager.play_sound('click')
    
    # --- Lifecycle Updates ---
    
    def update(self, dt):
        if self.prop_dialog: self.prop_dialog.update(dt)
        if self.rot_dialog: self.rot_dialog.update(dt)
        if self.anim_dialog: self.anim_dialog.update(dt)

    def draw_overlays(self, screen, font):
        if self.context_menu: self.context_menu.draw(screen, font)
        if self.prop_dialog: self.prop_dialog.draw(screen, font)
        if self.rot_dialog: self.rot_dialog.draw(screen, font)
        if self.anim_dialog: self.anim_dialog.draw(screen, font)