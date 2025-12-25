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

from ui.ui_widgets import MaterialDialog, RotationDialog, AnimationDialog, ContextMenu
from ui import icons
from model.geometry import Line, Circle, Point
from core.definitions import CONSTRAINT_DEFS
from core.sound_manager import SoundManager

# Import commands
from core.commands import RemoveEntityCommand, RemoveConstraintCommand, CompositeCommand


class AppController:
    """
    Handles high-level application logic, user actions, and UI coordination.
    """
    def __init__(self, app):
        self.app = app
        self.sound_manager = SoundManager.get()
        
        # State for dialogs/menus
        self.context_menu = None
        self.prop_dialog = None
        self.rot_dialog = None
        self.anim_dialog = None
        self.ctx_vars = {'wall': -1, 'pt': None, 'const': -1}

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
        self.sound_manager.play_sound('click')

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
            self.sim.resize_world(val)
            self.session.status.set(f"World Resized: {val}")
            self.sound_manager.play_sound('click')
        except ValueError:
            self.session.status.set("Invalid Size")

    # =========================================================================
    # Editor Actions
    # =========================================================================

    def toggle_ghost_mode(self):
        self.session.show_wall_atoms = not getattr(self.session, 'show_wall_atoms', True)
        if 'mode_ghost' in self.app.ui.buttons:
            self.app.ui.buttons['mode_ghost'].active = not self.session.show_wall_atoms 
        state = "Hidden" if not self.session.show_wall_atoms else "Visible"
        self.session.status.set(f"Wall Atoms: {state}")
        self.sound_manager.play_sound('click')
        
    def toggle_extend(self):
        if self.session.selection.walls:
            for idx in self.session.selection.walls:
                if idx < len(self.sketch.entities):
                    entity = self.sketch.entities[idx]
                    if isinstance(entity, Line):
                        if not hasattr(entity, 'infinite'):
                            entity.infinite = False
                        entity.infinite = not entity.infinite
            self.scene.rebuild()
            self.session.status.set("Toggled Extend")
            self.sound_manager.play_sound('click')
            
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
            count = 0
            for idx in self.session.selection.walls:
                if idx < len(self.sketch.entities):
                    self.sketch.update_entity(idx, physical=True)
                    count += 1
            self.scene.rebuild()
            self.session.status.set(f"Atomized {count} entities")
        else:
            self.scene.rebuild()
            self.session.status.set("Atomized All Geometry")
        self.sound_manager.play_sound('click')

    def action_delete_selection(self):
        """Delete selected entities using commands for proper undo/redo."""
        if not self.session.selection.walls:
            self.session.status.set("Nothing selected")
            return
            
        # Create composite command for multi-delete
        indices = sorted(self.session.selection.walls, reverse=True)
        cmds = [RemoveEntityCommand(self.sketch, idx) for idx in indices]
        
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
        self.sketch.add_material(mat)
        
        targets = []
        if self.session.selection.walls:
            targets = list(self.session.selection.walls)
        elif self.ctx_vars['wall'] != -1:
            targets = [self.ctx_vars['wall']]
            
        for idx in targets:
            if idx < len(self.sketch.entities):
                self.sketch.update_entity(idx, material_id=mat.name)
        
        self.scene.rebuild()
        self.session.status.set(f"Material Applied: {mat.name}")
        self.sound_manager.play_sound('click')

    def apply_rotation_from_dialog(self, dialog):
        # Legacy rotation via entity.anim is deprecated
        # Animation should be done via constraint drivers instead
        self.session.status.set("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')

    def apply_animation_from_dialog(self, dialog):
        if self.ctx_vars['const'] != -1:
            self.sketch.set_driver(self.ctx_vars['const'], dialog.get_values())
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
        current_mat = "Default"
        if target_idx != -1 and target_idx < len(self.sketch.entities):
            current_mat = self.sketch.entities[target_idx].material_id
        self.prop_dialog = MaterialDialog(mx, my, self.sketch, current_mat)

    def open_rotation_dialog(self):
        # Legacy rotation dialog - show message about using constraint drivers
        self.session.status.set("Use constraint drivers for animation (right-click constraint > Animate)")
        self.sound_manager.play_sound('error')
        
    def open_animation_dialog(self):
        if self.ctx_vars['const'] != -1:
            c = self.sketch.constraints[self.ctx_vars['const']]
            driver = getattr(c, 'driver', None)
            self.anim_dialog = AnimationDialog(
                self.app.layout['W'] // 2, 
                self.app.layout['H'] // 2, 
                driver
            )

    # =========================================================================
    # Context Menus
    # =========================================================================

    def get_context_options(self, target_type, idx1, idx2=None):
        options = []
        if target_type == 'wall':
            options = ["Properties", "Atomize", "Delete"]
        elif target_type == 'point':
            w_idx, pt_idx = idx1, idx2
            entities = self.sketch.entities
            if w_idx < len(entities):
                w = entities[w_idx]
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
        elif action == "Animate...":
            self.open_animation_dialog()
        elif action == "Delete":
            self.action_delete_selection()
        elif action == "Delete Constraint":
            self.action_delete_constraint()
        elif action == "Anchor" or action == "Un-Anchor":
            if self.ctx_vars['wall'] != -1 and self.ctx_vars['pt'] is not None:
                self.sketch.toggle_anchor(self.ctx_vars['wall'], self.ctx_vars['pt'])
                self.scene.rebuild()
                self.sound_manager.play_sound('click')
        elif action == "Atomize":
            if self.ctx_vars['wall'] != -1:
                self.session.selection.walls.add(self.ctx_vars['wall'])
                self.atomize_selected()
                self.session.selection.walls.clear()
        self.context_menu = None

    def spawn_context_menu(self, pos):
        mx, my = pos
        sim_x, sim_y = utils.screen_to_sim(
            mx, my, 
            self.session.camera.zoom, self.session.camera.pan_x, self.session.camera.pan_y, 
            self.sim.world_size, self.app.layout
        )
        
        # Check constraints first
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
                    self.context_menu = ContextMenu(mx, my, opts)
                    return

        # Check points using grouped point map
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
            self.context_menu = ContextMenu(mx, my, opts)
            if self.session.constraint_builder.pending_type:
                self.handle_pending_constraint_click(pt_idx=hit_pt)
            return

        # Check walls/entities using sketch's find_entity_at
        rad_sim = 5.0 / (((self.app.layout['MID_W'] - 50) / self.sim.world_size) * self.session.camera.zoom)
        hit_wall = self.sketch.find_entity_at(sim_x, sim_y, rad_sim)
        
        if hit_wall != -1:
            if self.session.constraint_builder.pending_type:
                self.handle_pending_constraint_click(wall_idx=hit_wall)
            else:
                self.ctx_vars['wall'] = hit_wall
                opts = self.get_context_options('wall', hit_wall)
                self.context_menu = ContextMenu(mx, my, opts)
        else:
            if self.session.mode == config.MODE_EDITOR: 
                self.app.change_tool(config.TOOL_SELECT)
                self.session.status.set("Switched to Select Tool")

    # =========================================================================
    # Constraint Handling
    # =========================================================================

    def trigger_constraint(self, ctype):
        from core.commands import AddConstraintCommand
        
        # Update UI button states
        for btn, c_val in self.app.input_handler.constraint_btn_map.items():
            btn.active = (c_val == ctype)
        
        # Handle multi-apply constraints (H/V)
        is_multi = False
        if ctype in CONSTRAINT_DEFS and CONSTRAINT_DEFS[ctype][0].get('multi'):
            walls = list(self.session.selection.walls)
            if walls:
                count = 0
                for w_idx in walls:
                    if self.sketch.attempt_apply_constraint(ctype, [w_idx], []):
                        count += 1
                if count > 0:
                    self.scene.rebuild()
                    self.session.status.set(f"Applied {ctype} to {count} items")
                    self.session.selection.walls.clear()
                    self.session.selection.points.clear()
                    is_multi = True
                    self.sound_manager.play_sound('click')
        if is_multi:
            return
        
        # Try to apply with current selection
        walls = list(self.session.selection.walls)
        pts = list(self.session.selection.points)
        if self._try_apply_constraint_smart(ctype, walls, pts):
            return

        # Enter pending constraint mode
        self.session.constraint_builder.pending_type = ctype
        self.session.constraint_builder.target_walls = list(self.session.selection.walls)
        self.session.constraint_builder.target_points = list(self.session.selection.points)
        self.session.selection.walls.clear()
        self.session.selection.points.clear()
        msg = CONSTRAINT_DEFS[ctype][0]['msg'] if ctype in CONSTRAINT_DEFS else "Select targets..."
        self.session.status.set(f"{ctype}: {msg}")
        self.sound_manager.play_sound('tool_select')

    def _try_apply_constraint_smart(self, ctype, walls, pts):
        """Try to apply a constraint with the given selection."""
        if self.sketch.attempt_apply_constraint(ctype, walls, pts):
            self.scene.rebuild()
            self.session.status.set(f"Applied {ctype}")
            self.session.selection.walls.clear()
            self.session.selection.points.clear()
            self.session.constraint_builder.pending_type = None
            for btn in self.app.input_handler.constraint_btn_map.keys():
                btn.active = False
            self.sound_manager.play_sound('click')
            return True
        
        # Try auto-trimming selection
        if ctype in CONSTRAINT_DEFS:
            rules = CONSTRAINT_DEFS[ctype]
            for r in rules:
                if len(walls) >= r['w'] and len(pts) >= r['p']:
                    sub_w = walls[:r['w']]
                    sub_p = pts[:r['p']]
                    if self.sketch.attempt_apply_constraint(ctype, sub_w, sub_p):
                        self.scene.rebuild()
                        self.session.status.set(f"Applied {ctype} (Auto-Trimmed)")
                        self.session.selection.walls.clear()
                        self.session.selection.points.clear()
                        self.session.constraint_builder.pending_type = None
                        for btn in self.app.input_handler.constraint_btn_map.keys():
                            btn.active = False
                        self.sound_manager.play_sound('click')
                        return True
        return False

    def handle_pending_constraint_click(self, wall_idx=None, pt_idx=None):
        if not self.session.constraint_builder.pending_type:
            return
        if wall_idx is not None and wall_idx not in self.session.constraint_builder.target_walls:
            self.session.constraint_builder.target_walls.append(wall_idx)
        if pt_idx is not None and pt_idx not in self.session.constraint_builder.target_points:
            self.session.constraint_builder.target_points.append(pt_idx)
        
        if self._try_apply_constraint_smart(
            self.session.constraint_builder.pending_type, 
            self.session.constraint_builder.target_walls, 
            self.session.constraint_builder.target_points
        ):
            return
            
        ctype = self.session.constraint_builder.pending_type
        msg = CONSTRAINT_DEFS[ctype][0]['msg']
        nw = len(self.session.constraint_builder.target_walls)
        np_count = len(self.session.constraint_builder.target_points)
        self.session.status.set(f"{ctype} ({nw}W, {np_count}P): {msg}")
        self.sound_manager.play_sound('click')
    
    # =========================================================================
    # Lifecycle Updates
    # =========================================================================
    
    def update(self, dt):
        if self.prop_dialog:
            self.prop_dialog.update(dt)
        if self.rot_dialog:
            self.rot_dialog.update(dt)
        if self.anim_dialog:
            self.anim_dialog.update(dt)

    def draw_overlays(self, screen, font):
        if self.context_menu:
            self.context_menu.draw(screen, font)
        if self.prop_dialog:
            self.prop_dialog.draw(screen, font)
        if self.rot_dialog:
            self.rot_dialog.draw(screen, font)
        if self.anim_dialog:
            self.anim_dialog.draw(screen, font)