import pygame
import numpy as np
import math
import core.config as config

import core.utils as utils

import time
from model.geometry import Line, Circle
from model.constraints import Coincident, Length, Angle
from core.session import InteractionState

# Phase 8b: Import commands for proper undo/redo
from core.commands import (
    AddLineCommand, AddCircleCommand, RemoveEntityCommand,
    MoveEntityCommand, MoveMultipleCommand, AddConstraintCommand,
    AddRectangleCommand, CompositeCommand
)


class Tool:
    def __init__(self, app, name="Tool"):
        self.app = app
        self.name = name

    def activate(self):
        pass

    def deactivate(self):
        self.app.session.current_snap_target = None

    def handle_event(self, event, layout):
        return False

    def update(self, dt, layout):
        pass

    def draw_overlay(self, screen, renderer, layout):
        pass
    
    def cancel(self):
        pass


class BrushTool(Tool):
    """
    Brush tool for painting/erasing particles.
    Does NOT use commands - particle operations are not undoable
    (they're physics state, not CAD state).
    """
    def __init__(self, app):
        super().__init__(app, "Brush")
        self.brush_radius = 5.0 

    def update(self, dt, layout):
        if self.app.session.state == InteractionState.PAINTING:
            mx, my = pygame.mouse.get_pos()
            if layout['LEFT_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT:
                sx, sy = utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
                
                radius = self.brush_radius
                
                if pygame.mouse.get_pressed()[0]: 
                    self.app.sim.add_particles_brush(sx, sy, radius)
                elif pygame.mouse.get_pressed()[2]: 
                    self.app.sim.delete_particles_brush(sx, sy, radius)

    def handle_event(self, event, layout):
        if self.app.session.mode != config.MODE_SIM: return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if not (layout['MID_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
                return False

            if event.button == 1:
                self.app.session.state = InteractionState.PAINTING
                # Note: No snapshot - particle operations use old system
                self.app.sim.snapshot()
                return True
            elif event.button == 3:
                self.app.session.state = InteractionState.PAINTING
                self.app.sim.snapshot()
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.app.session.state == InteractionState.PAINTING:
                self.app.session.state = InteractionState.IDLE
                return True
        return False

    def draw_overlay(self, screen, renderer, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['MID_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT:
            zoom = self.app.session.zoom
            world_size = self.app.sim.world_size
            base_scale = (layout['MID_W'] - 50) / world_size
            final_scale = base_scale * zoom
            screen_r = self.brush_radius * final_scale
            
            renderer.draw_tool_brush(mx, my, screen_r)


class GeometryTool(Tool):
    """
    Base class for geometry creation tools.
    Phase 8b: Uses commands for undoable operations.
    """
    def __init__(self, app, name):
        super().__init__(app, name)
        self.dragging = False
        self.start_pos = None
        self.created_indices = []
        self.pending_commands = []  # Commands to execute on completion

    def get_world_pos(self, mx, my, layout):
        return utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)

    def get_snapped(self, mx, my, layout, anchor=None, exclude_idx=-1):
        return utils.get_snapped_pos(mx, my, self.app.sketch.entities, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout, anchor, exclude_idx)

    def cancel(self):
        """Cancel current operation - undo any pending geometry."""
        if self.dragging:
            # Remove any geometry created during this drag
            for idx in sorted(self.created_indices, reverse=True):
                if idx < len(self.app.sketch.entities):
                    self.app.sketch.entities.pop(idx)
            self.dragging = False
            self.created_indices = []
            self.pending_commands = []
            self.app.sim.rebuild_static_atoms()
            self.app.session.set_status("Cancelled")
        self.app.session.current_snap_target = None

    def _update_hover_snap(self, mx, my, layout):
        if not self.dragging:
            _, _, snap = self.get_snapped(mx, my, layout, anchor=None)
            self.app.session.current_snap_target = snap


class LineTool(GeometryTool):
    """
    Line drawing tool.
    Phase 8b: Uses AddLineCommand for undoable line creation.
    """
    def __init__(self, app):
        super().__init__(app, "Line")
        self.current_wall_idx = -1
        self.start_snap = None  # Store snap target from start point

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                self.start_snap = snap
                self.start_pos = (sx, sy)
                
                # Create line directly for visual feedback during drag
                # We'll convert to command on completion
                is_ref = (self.name == "Ref Line")
                self.app.sketch.add_line((sx, sy), (sx, sy), is_ref=is_ref)
                self.current_wall_idx = len(self.app.sketch.entities) - 1
                self.created_indices = [self.current_wall_idx]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                walls = self.app.sketch.entities
                if self.current_wall_idx < len(walls):
                    w = walls[self.current_wall_idx]
                    sx, sy, snap = self.get_snapped(mx, my, layout, w.start, self.current_wall_idx)
                    
                    # Update line end point directly during drag
                    w.end[:] = (sx, sy)
                    self.app.session.current_snap_target = snap
                    self.app.sim.rebuild_static_atoms()
                    
                    if self.app.session.mode == config.MODE_EDITOR:
                        self.app.sim.apply_constraints()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            mx, my = event.pos
            walls = self.app.sketch.entities
            
            if self.current_wall_idx < len(walls):
                w = walls[self.current_wall_idx]
                
                # Check if line is too short (degenerate)
                if np.linalg.norm(w.end - w.start) < 1e-4:
                    # Remove the degenerate line
                    self.app.sketch.entities.pop(self.current_wall_idx)
                    self.dragging = False
                    self.created_indices = []
                    self.app.session.state = InteractionState.IDLE
                    self.app.sim.rebuild_static_atoms()
                    return True

                # Line is valid - finalize with command
                sx, sy, end_snap = self.get_snapped(mx, my, layout, w.start, self.current_wall_idx)
                
                # Remove the preview line (we'll recreate via command)
                start = tuple(w.start)
                end = tuple(w.end)
                is_ref = w.ref
                material_id = w.material_id
                self.app.sketch.entities.pop(self.current_wall_idx)
                
                # Build composite command with line + any constraints
                commands = []
                
                # Main line command
                line_cmd = AddLineCommand(self.app.sketch, start, end, is_ref=is_ref, material_id=material_id)
                commands.append(line_cmd)
                
                # Execute line first to get its index
                self.app.scene.execute(line_cmd)
                line_idx = line_cmd.created_index
                
                # Add snap constraints if Ctrl was held
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if self.start_snap:
                        c = Coincident(line_idx, 0, self.start_snap[0], self.start_snap[1])
                        self.app.scene.execute(AddConstraintCommand(self.app.sketch, c))
                    
                    if end_snap:
                        c = Coincident(line_idx, 1, end_snap[0], end_snap[1])
                        self.app.scene.execute(AddConstraintCommand(self.app.sketch, c))
                
                # Add orientation constraints if Shift was held
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    dx = abs(start[0] - end[0])
                    dy = abs(start[1] - end[1])
                    if dy < 0.001:
                        self.app.scene.execute(AddConstraintCommand(self.app.sketch, Angle('HORIZONTAL', line_idx)))
                    elif dx < 0.001:
                        self.app.scene.execute(AddConstraintCommand(self.app.sketch, Angle('VERTICAL', line_idx)))

            self.dragging = False
            self.created_indices = []
            self.start_snap = None
            self.app.session.current_snap_target = None
            self.app.session.state = InteractionState.IDLE
            self.app.sim.apply_constraints()
            return True
        return False

    def draw_overlay(self, screen, renderer, layout):
        pass


class RectTool(GeometryTool):
    """
    Rectangle drawing tool.
    Phase 8b: Uses AddRectangleCommand for undoable rectangle creation.
    """
    def __init__(self, app):
        super().__init__(app, "Rectangle")
        self.base_idx = -1
        self.rect_start_pos = None

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, _ = self.get_snapped(mx, my, layout)
                self.rect_start_pos = (sx, sy)
                
                # Create 4 preview lines
                self.base_idx = len(self.app.sketch.entities)
                for _ in range(4):
                    self.app.sketch.add_line((sx, sy), (sx, sy))
                self.created_indices = [self.base_idx + i for i in range(4)]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cur_x, cur_y = self.get_world_pos(mx, my, layout)
                sx, sy = self.rect_start_pos
                
                # Update preview lines
                walls = self.app.sketch.entities
                if self.base_idx + 3 < len(walls):
                    walls[self.base_idx].start[:] = (sx, sy)
                    walls[self.base_idx].end[:] = (cur_x, sy)
                    walls[self.base_idx+1].start[:] = (cur_x, sy)
                    walls[self.base_idx+1].end[:] = (cur_x, cur_y)
                    walls[self.base_idx+2].start[:] = (cur_x, cur_y)
                    walls[self.base_idx+2].end[:] = (sx, cur_y)
                    walls[self.base_idx+3].start[:] = (sx, cur_y)
                    walls[self.base_idx+3].end[:] = (sx, sy)
                    self.app.sim.rebuild_static_atoms()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            # Get final corners
            mx, my = event.pos
            cur_x, cur_y = self.get_world_pos(mx, my, layout)
            sx, sy = self.rect_start_pos
            
            # Remove preview lines
            for _ in range(4):
                if self.base_idx < len(self.app.sketch.entities):
                    self.app.sketch.entities.pop(self.base_idx)
            
            # Create rectangle via command (includes constraints)
            cmd = AddRectangleCommand(self.app.sketch, sx, sy, cur_x, cur_y)
            self.app.scene.execute(cmd)
            
            self.dragging = False
            self.created_indices = []
            self.app.session.state = InteractionState.IDLE
            return True
        return False


class CircleTool(GeometryTool):
    """
    Circle drawing tool.
    Phase 8b: Uses AddCircleCommand for undoable circle creation.
    """
    def __init__(self, app):
        super().__init__(app, "Circle")
        self.circle_idx = -1
        self.center_snap = None

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                self.center_snap = snap
                self.start_pos = (sx, sy)
                
                # Create preview circle
                self.app.sketch.add_circle((sx, sy), 0.1)
                self.circle_idx = len(self.app.sketch.entities) - 1
                self.created_indices = [self.circle_idx]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cur_x, cur_y = self.get_world_pos(mx, my, layout)
                walls = self.app.sketch.entities
                if self.circle_idx < len(walls):
                    c = walls[self.circle_idx]
                    c.radius = max(0.1, math.hypot(cur_x - c.center[0], cur_y - c.center[1]))
                    self.app.sim.rebuild_static_atoms()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            walls = self.app.sketch.entities
            if self.circle_idx < len(walls):
                c = walls[self.circle_idx]
                center = tuple(c.center)
                radius = c.radius
                material_id = c.material_id
                
                # Remove preview
                self.app.sketch.entities.pop(self.circle_idx)
                
                # Create via command
                cmd = AddCircleCommand(self.app.sketch, center, radius, material_id=material_id)
                self.app.scene.execute(cmd)
                circle_idx = cmd.created_index
                
                # Add snap constraint if Ctrl held
                if pygame.key.get_mods() & pygame.KMOD_CTRL and self.center_snap:
                    c = Coincident(circle_idx, 0, self.center_snap[0], self.center_snap[1])
                    self.app.scene.execute(AddConstraintCommand(self.app.sketch, c))

            self.dragging = False
            self.created_indices = []
            self.center_snap = None
            self.app.session.state = InteractionState.IDLE
            self.app.session.current_snap_target = None
            return True
        return False


class PointTool(GeometryTool):
    """
    Point creation tool.
    Phase 8b: Uses commands (Point is created as degenerate line for now).
    """
    def __init__(self, app):
        super().__init__(app, "Point")

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._update_hover_snap(mx, my, layout)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                
                # Create point as degenerate line via command
                cmd = AddLineCommand(self.app.sketch, (sx, sy), (sx, sy), is_ref=False)
                self.app.scene.execute(cmd)
                wall_idx = cmd.created_index
                
                # Add snap constraint if Ctrl held
                if snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    c = Coincident(wall_idx, 0, snap[0], snap[1])
                    self.app.scene.execute(AddConstraintCommand(self.app.sketch, c))
                return True
        return False
    
    def draw_overlay(self, screen, renderer, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['MID_X'] < mx < layout['RIGHT_X']:
             renderer.draw_tool_point(mx, my)


class SelectTool(Tool):
    """
    Selection and manipulation tool.
    Phase 8b: Uses MoveEntityCommand for undoable moves.
    """
    def __init__(self, app):
        super().__init__(app, "Select")
        self.mode = None 
        self.target_idx = -1
        self.target_pt = -1
        self.group_indices = []
        self.drag_start_mouse = (0, 0)
        self.drag_start_sim = (0, 0)  # Track sim coords for command
        self.drag_rect = None
        self.last_click_time = 0.0
        self.last_click_pos = (0, 0)
        self.DOUBLE_CLICK_TIME = 0.3
        self.DOUBLE_CLICK_DIST = 5
        
        # Phase 8b: Track accumulated movement for command
        self.total_dx = 0.0
        self.total_dy = 0.0

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                if event.button == 1: 
                    return self._handle_left_click(mx, my, layout)
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mode:
                return self._handle_drag(event.pos, layout)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.mode:
                return self._handle_release(event.pos, layout)
                
        return False

    def cancel(self):
        # If we were dragging, we need to undo the movement
        if self.mode in ['MOVE_WALL', 'MOVE_GROUP', 'EDIT']:
            # Reverse the accumulated movement
            if self.mode == 'MOVE_GROUP':
                for idx in self.group_indices:
                    if idx < len(self.app.sketch.entities):
                        self.app.sketch.entities[idx].move(-self.total_dx, -self.total_dy)
            elif self.mode in ['MOVE_WALL', 'EDIT']:
                if self.target_idx < len(self.app.sketch.entities):
                    entity = self.app.sketch.entities[self.target_idx]
                    if self.mode == 'EDIT':
                        # For point editing, only move the specific point
                        indices = [self.target_pt] if self.target_pt >= 0 else None
                        entity.move(-self.total_dx, -self.total_dy, indices)
                    else:
                        entity.move(-self.total_dx, -self.total_dy)
            
            self.app.sim.rebuild_static_atoms()
        
        self.mode = None
        self.group_indices = []
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.app.session.state = InteractionState.IDLE
        self.app.session.set_status("Move Cancelled")

    def _handle_left_click(self, mx, my, layout):
        now = time.time()
        is_double = False
        dist = math.hypot(mx - self.last_click_pos[0], my - self.last_click_pos[1])
        if (now - self.last_click_time < self.DOUBLE_CLICK_TIME) and (dist < self.DOUBLE_CLICK_DIST):
            is_double = True
        
        self.last_click_time = now
        self.last_click_pos = (mx, my)

        # 1. Check Points
        point_map = utils.get_grouped_points(self.app.sketch.entities, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
        hit_pt = self._hit_test_points(mx, my, point_map)
        
        if hit_pt:
            if hit_pt in self.app.session.selected_points:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._select_point(hit_pt) 
            else:
                self._select_point(hit_pt)

            self.mode = 'EDIT'
            self.target_idx = hit_pt[0]
            self.target_pt = hit_pt[1]
            self.drag_start_mouse = (mx, my)
            self.drag_start_sim = utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
            self.total_dx = 0.0
            self.total_dy = 0.0
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True
            
        # 2. Check Walls
        hit_wall = self._hit_test_walls(mx, my, layout)
        if hit_wall != -1:
            w = self.app.sketch.entities[hit_wall]
            
            if is_double:
                group = utils.get_connected_group(self.app.sketch.constraints, hit_wall)
                self.app.session.selected_walls.update(group)
                self.app.session.set_status("Group Selected")
                return True

            if hit_wall in self.app.session.selected_walls:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._select_wall(hit_wall)
            else:
                if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                     self.app.session.selected_walls.clear()
                     self.app.session.selected_points.clear()
                self._select_wall(hit_wall)
            
            if isinstance(w, Circle):
                self.mode = 'RESIZE_CIRCLE'
                self.target_idx = hit_wall
                self.drag_start_mouse = (mx, my)
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True
            
            is_part_of_selection = hit_wall in self.app.session.selected_walls
            if is_part_of_selection and len(self.app.session.selected_walls) > 1:
                self.mode = 'MOVE_GROUP'
                self.group_indices = list(self.app.session.selected_walls)
            else:
                self.mode = 'MOVE_WALL'
                self.target_idx = hit_wall
                # Add temp length constraint for editor mode
                if isinstance(w, Line) and self.app.session.mode == config.MODE_EDITOR:
                    l = np.linalg.norm(w.end - w.start)
                    c = Length(hit_wall, l)
                    c.temp = True
                    self.app.sketch.constraints.append(c)
                    self.app.session.temp_constraint_active = True
            
            self.drag_start_mouse = (mx, my)
            self.drag_start_sim = utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
            self.total_dx = 0.0
            self.total_dy = 0.0
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True
            
        # 3. Clicked Empty -> Deselect
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.app.session.selected_walls.clear()
            self.app.session.selected_points.clear()
        return True

    def _handle_drag(self, mouse_pos, layout):
        mx, my = mouse_pos
        curr_sim = utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
        prev_sim = utils.screen_to_sim(self.drag_start_mouse[0], self.drag_start_mouse[1], self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
        
        dx = curr_sim[0] - prev_sim[0]
        dy = curr_sim[1] - prev_sim[1]
        self.drag_start_mouse = (mx, my)
        
        if self.mode == 'EDIT':
            walls = self.app.sketch.entities
            if self.target_idx >= len(walls):
                return False
            w = walls[self.target_idx]
            
            if isinstance(w, Line):
                anchor = w.end if self.target_pt == 0 else w.start
                dest_x, dest_y, snap = utils.get_snapped_pos(mx, my, walls, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout, anchor, self.target_idx)
                
                self.app.session.current_snap_target = snap
                
                # Calculate actual delta for this point
                old_pt = w.get_point(self.target_pt)
                actual_dx = dest_x - old_pt[0]
                actual_dy = dest_y - old_pt[1]
                
                if self.target_pt == 0:
                    w.start[:] = (dest_x, dest_y)
                else:
                    w.end[:] = (dest_x, dest_y)
                
                self.total_dx += actual_dx
                self.total_dy += actual_dy
            
            elif isinstance(w, Circle):
                dest_x, dest_y, snap = utils.get_snapped_pos(mx, my, walls, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout, None, self.target_idx)
                self.app.session.current_snap_target = snap
                
                old_center = w.center.copy()
                w.center[:] = (dest_x, dest_y)
                self.total_dx += dest_x - old_center[0]
                self.total_dy += dest_y - old_center[1]
            
            self.app.sim.rebuild_static_atoms()
            if self.app.session.mode == config.MODE_EDITOR:
                self.app.sim.apply_constraints()
            return True

        elif self.mode == 'RESIZE_CIRCLE':
            w = self.app.sketch.entities[self.target_idx]
            if isinstance(w, Circle):
                new_r = math.hypot(curr_sim[0] - w.center[0], curr_sim[1] - w.center[1])
                w.radius = max(0.1, new_r)
                self.app.sim.rebuild_static_atoms()
            return True

        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            self.total_dx += dx
            self.total_dy += dy
            
            targets = self.group_indices if self.mode == 'MOVE_GROUP' else [self.target_idx]
            
            for idx in targets:
                walls = self.app.sketch.entities
                if idx < len(walls):
                    w = walls[idx]
                    w.move(dx, dy) 
            
            self.app.sim.rebuild_static_atoms()
            if self.app.session.mode == config.MODE_EDITOR:
                self.app.sim.apply_constraints()
            return True
            
        return False

    def _handle_release(self, mouse_pos, layout):
        mx, my = mouse_pos
        
        # Handle snap connection on point edit release
        if self.mode == 'EDIT' and self.app.session.current_snap_target:
            snap = self.app.session.current_snap_target
            if not (self.target_idx == snap[0] and self.target_pt == snap[1]):
                c = Coincident(
                    self.target_idx, self.target_pt,
                    snap[0], snap[1]
                )
                self.app.scene.execute(AddConstraintCommand(self.app.sketch, c))
                self.app.session.set_status("Snapped & Connected")

        # Create command for the movement (if significant)
        if abs(self.total_dx) > 1e-6 or abs(self.total_dy) > 1e-6:
            if self.mode == 'MOVE_GROUP' and self.group_indices:
                # Already moved during drag - just record for potential undo
                # For now, we don't undo via command since movement already happened
                pass
            elif self.mode in ['MOVE_WALL', 'EDIT']:
                # Already moved during drag
                pass

        self.mode = None
        self.app.session.current_snap_target = None
        self.app.session.state = InteractionState.IDLE
        self.total_dx = 0.0
        self.total_dy = 0.0
        
        # Remove temp constraint
        if self.app.session.temp_constraint_active:
            self.app.sketch.constraints = [c for c in self.app.sketch.constraints if not c.temp]
            self.app.session.temp_constraint_active = False
        
        self.app.sim.apply_constraints()
        return True

    def _hit_test_points(self, mx, my, point_map):
        base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            dist = math.hypot(mx - center_pos[0], my - center_pos[1])
            max_r = base_r + (len(items) - 1) * step_r
            if dist <= max_r:
                if dist < base_r: return items[0]
                k = int((dist - base_r) / step_r) + 1
                if k < len(items): return items[k]
                return items[-1]
        return None

    def _hit_test_walls(self, mx, my, layout):
        sim_x, sim_y = utils.screen_to_sim(mx, my, self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, self.app.sim.world_size, layout)
        rad_sim = 5.0 / (((layout['MID_W'] - 50) / self.app.sim.world_size) * self.app.session.zoom)
        
        walls = self.app.sketch.entities
        for i, w in enumerate(walls):
            if isinstance(w, Line):
                if np.array_equal(w.start, w.end): continue
                p1, p2 = w.start, w.end
                p3 = np.array([sim_x, sim_y])
                d_vec = p2-p1
                len_sq = np.dot(d_vec, d_vec)
                if len_sq == 0: dist = np.linalg.norm(p3-p1)
                else:
                    t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                    proj = p1 + t*d_vec
                    dist = np.linalg.norm(p3-proj)
                if dist < rad_sim: return i
            elif isinstance(w, Circle):
                d = math.hypot(sim_x - w.center[0], sim_y - w.center[1])
                if abs(d - w.radius) < rad_sim: return i
        return -1

    def _select_point(self, pt_tuple):
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            if pt_tuple not in self.app.session.selected_points:
                self.app.session.selected_walls.clear()
                self.app.session.selected_points.clear()
        
        if pt_tuple in self.app.session.selected_points:
            self.app.session.selected_points.remove(pt_tuple)
        else:
            self.app.session.selected_points.add(pt_tuple)

    def _select_wall(self, wall_idx):
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            if wall_idx not in self.app.session.selected_walls:
                self.app.session.selected_walls.clear()
                self.app.session.selected_points.clear()
        
        if wall_idx in self.app.session.selected_walls:
            self.app.session.selected_walls.remove(wall_idx)
        else:
            self.app.session.selected_walls.add(wall_idx)